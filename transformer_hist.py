import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import transformer as base

from transformer import (
    RESULTS_DIR,
    SCENARIO_SAMPLES,
    RollingWindowResult,
    add_scenario_noise,
    build_sequence_arrays,
    compute_directional_accuracy,
    drop_near_constant_columns,
    fine_tune_cnn,
    merge_all_series,
    plot_predictions,
    plot_reconstructed_levels,
    plot_residuals,
    timesfm_scenarios,
    ForecastModel,
    train_cnn_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Historical-only weekly training (no TimesFM predictions).")
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon-weeks", type=int, default=2)
    parser.add_argument("--step-weeks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scenario-k", type=int, default=SCENARIO_SAMPLES)
    parser.add_argument("--cnn-hidden", nargs="+", type=int, default=[256, 128], help="Hidden layer sizes for the MLP head.")
    parser.add_argument("--tfm-layers", type=int, default=2, help="Number of transformer layers.")
    parser.add_argument("--weight-decay", type=float, default=1e-6, help="Weight decay for CNN training.")
    parser.add_argument("--run-name", type=str, default="hist_only")
    parser.add_argument("--grid", action="store_true", help="Run a hyperparameter grid instead of a single configuration.")
    return parser.parse_args()


def walkforward_hist_pipeline(args):
    df = merge_all_series()
    true_cols = sorted([c for c in df.columns if c.endswith("_true")])
    df = df.sort_values("date").set_index("date")
    df = df[true_cols + ["sp500_close"]].dropna()
    df["weekly_close"] = df["sp500_close"].resample("W-FRI").last().ffill()
    df["date"] = df.index
    df["y_ret"] = np.log(df["weekly_close"]).diff()
    df = df.dropna(subset=["y_ret"]).reset_index(drop=True)
    y = df["y_ret"].values.astype(np.float32)

    split_idx = int(len(df) * 0.6)
    val_size = max(int(len(df) * 0.06), 8)
    train_end = split_idx - val_size
    if train_end <= args.context_len:
        raise ValueError("Not enough data for requested context length.")

    keep_cols = drop_near_constant_columns(df.iloc[:train_end][true_cols])
    df = df[keep_cols + ["date", "sp500_close", "y_ret"]]

    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end][keep_cols].to_numpy())
    true_scaled = scaler.transform(df[keep_cols].to_numpy())

    residual_std = np.maximum(true_scaled[:train_end].std(axis=0), 1e-6)

    train_seq, train_targets = build_sequence_arrays(true_scaled, y, args.context_len, args.context_len, train_end)
    val_seq, val_targets = build_sequence_arrays(true_scaled, y, args.context_len, train_end, split_idx)

    model = ForecastModel(
        feature_dim=true_scaled.shape[1],
        dropout=args.dropout,
        hidden_dims=args.cnn_hidden,
        transformer_layers=args.tfm_layers,
    )
    train_cnn_model(
        model,
        train_seq,
        train_targets,
        val_seq,
        val_targets,
        residual_std,
        args.epochs,
        args.batch_size,
        args.patience,
        args.lr,
        weight_decay=args.weight_decay,
    )

    rolling_preds = []
    rolling_metrics: List[RollingWindowResult] = []
    cursor = split_idx
    window_idx = 1
    fine_tune_lr = args.lr * 0.3
    seq_len = args.context_len

    while cursor < len(df):
        future_end = min(cursor + args.horizon_weeks, len(df))
        if future_end - cursor <= 0:
            break

        # Build future sequences using only historical context (last known block)
        base_context = true_scaled[cursor - seq_len : cursor]
        window_len = future_end - cursor
        scenario_seq = [base_context.copy() for _ in range(window_len)]
        future_dates = df["date"].iloc[cursor:future_end].values
        y_future = y[cursor:future_end]

        scenario_preds = timesfm_scenarios(np.array(scenario_seq), residual_std, args.scenario_k)
        preds_samples = []
        model.eval()
        with torch.no_grad():
            for scenario in scenario_preds:
                tensor = torch.tensor(scenario, dtype=torch.float32)
                preds = model(tensor).squeeze(-1).numpy()
                preds_samples.append(preds)
        preds_samples = np.stack(preds_samples)
        pred_mean = preds_samples.mean(axis=0)
        pred_p10 = np.percentile(preds_samples, 10, axis=0)
        pred_p50 = np.percentile(preds_samples, 50, axis=0)
        pred_p90 = np.percentile(preds_samples, 90, axis=0)

        rmse = np.sqrt(mean_squared_error(y_future, pred_mean))
        r2 = r2_score(y_future, pred_mean)
        dir_acc = compute_directional_accuracy(y_future, pred_mean)
        start_dt = pd.to_datetime(future_dates[0])
        end_dt = pd.to_datetime(future_dates[-1])
        rolling_metrics.append(
            RollingWindowResult(
                window=window_idx,
                start_date=start_dt,
                end_date=end_dt,
                rmse=rmse,
                r2=r2,
                direction_acc=dir_acc,
            )
        )
        print(
            f"[Hist Window {window_idx}] {start_dt.date()} → {end_dt.date()} | "
            f"RMSE {rmse:.4f} | R^2 {r2:.4f} | DirAcc {dir_acc:.3f}"
        )

        rolling_preds.append(
            pd.DataFrame(
                {
                    "date": future_dates,
                    "y_true": y_future,
                    "y_pred_mean": pred_mean,
                    "y_pred_p10": pred_p10,
                    "y_pred_p50": pred_p50,
                    "y_pred_p90": pred_p90,
                }
            )
        )
        fine_seq, fine_targets = build_sequence_arrays(true_scaled, y, seq_len, cursor, future_end)
        fine_tune_cnn(
            model,
            fine_seq,
            fine_targets,
            fine_tune_lr,
            args.fine_tune_epochs,
            residual_std,
            weight_decay=args.weight_decay,
        )

        cursor += args.step_weeks
        window_idx += 1

    if not rolling_preds:
        raise RuntimeError("No rolling predictions produced.")

    results_df = pd.concat(rolling_preds, ignore_index=True).sort_values("date")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = RESULTS_DIR / "rolling_predictions_hist.csv"
    results_df.to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame(
        {
            "window": [m.window for m in rolling_metrics],
            "start_date": [m.start_date for m in rolling_metrics],
            "end_date": [m.end_date for m in rolling_metrics],
            "rmse": [m.rmse for m in rolling_metrics],
            "r2": [m.r2 for m in rolling_metrics],
            "directional_accuracy": [m.direction_acc for m in rolling_metrics],
        }
    )
    metrics_path = RESULTS_DIR / "rolling_metrics_hist.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved historical rolling predictions -> {preds_path}")
    print(f"Saved historical metrics -> {metrics_path}")

    plot_predictions(results_df, args.horizon_weeks)
    plot_residuals(results_df, args.horizon_weeks)
    plot_reconstructed_levels(results_df, df[["date", "sp500_close"]].copy(), args.horizon_weeks)

    overall_rmse = float(np.sqrt(mean_squared_error(results_df["y_true"], results_df["y_pred_mean"])))
    overall_r2 = float(r2_score(results_df["y_true"], results_df["y_pred_mean"]))
    overall_dir = compute_directional_accuracy(results_df["y_true"].values, results_df["y_pred_mean"].values)
    log_row = pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "context_len": args.context_len,
                "horizon_weeks": args.horizon_weeks,
                "step_weeks": args.step_weeks,
                "epochs": args.epochs,
                "dropout": args.dropout,
                "lr": args.lr,
                "scenario_k": args.scenario_k,
                "cnn_hidden": "-".join(map(str, args.cnn_hidden)),
                "tfm_layers": args.tfm_layers,
                "rmse": overall_rmse,
                "r2": overall_r2,
                "directional_accuracy": overall_dir,
                "mode": "historical_only",
            }
        ]
    )
    log_path = RESULTS_DIR / "hyperparam_runs_hist.csv"
    log_row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(
        f"[Hist Hyperparam Log] {args.run_name} -> RMSE={overall_rmse:.4f} R^2={overall_r2:.4f} "
        f"DirAcc={overall_dir:.3f}"
    )


def main():
    args = parse_args()
    log_path = RESULTS_DIR / "hyperparam_runs_hist.csv"
    if args.grid:
        context_lens = [12, 16, 20]
        dropouts = [0.1, 0.2, 0.3]
        lrs = [1e-4, 5e-5]
        scenario_ks = [3, 5]
        cnn_hidden_options = [[128, 64], [256, 128]]
        tfm_layers_options = [1, 2]
        run_idx = 1
        for c in context_lens:
            for d in dropouts:
                for lr in lrs:
                    for k in scenario_ks:
                        for hidden in cnn_hidden_options:
                            for tfm_layers in tfm_layers_options:
                                print(
                                    f"\n[Grid] Run {run_idx} | context={c} dropout={d} lr={lr} "
                                    f"scenario_k={k} hidden={hidden} tfm_layers={tfm_layers}"
                                )
                                run_args = argparse.Namespace(
                                    context_len=c,
                                    epochs=args.epochs,
                                    patience=args.patience,
                                    fine_tune_epochs=args.fine_tune_epochs,
                                    batch_size=args.batch_size,
                                    horizon_weeks=args.horizon_weeks,
                                    step_weeks=args.step_weeks,
                                    dropout=d,
                                    lr=lr,
                                    scenario_k=k,
                                    cnn_hidden=hidden,
                                    tfm_layers=tfm_layers,
                                    weight_decay=args.weight_decay,
                                    run_name=f"{args.run_name}_grid{run_idx}",
                                    grid=False,
                                )
                                walkforward_hist_pipeline(run_args)
                                run_idx += 1
        if log_path.exists():
            df = pd.read_csv(log_path)
            best_idx = df["directional_accuracy"].idxmax()
            best_row = df.loc[best_idx]
            print("\n[Grid Result] Best configuration:")
            for col, val in best_row.items():
                print(f"  {col}: {val}")
    else:
        walkforward_hist_pipeline(args)


if __name__ == "__main__":
    main()
