import argparse
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from transformer import (
    RESULTS_DIR,
    RollingWindowResult,
    align_true_pred_columns,
    build_sequence_arrays,
    compute_directional_accuracy,
    drop_near_constant_columns,
    merge_all_series,
    plot_predictions,
    plot_reconstructed_levels,
    plot_residuals,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid Ridge regression baseline using TimesFM predictions.")
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--horizon-weeks", type=int, default=2)
    parser.add_argument("--step-weeks", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization strength.")
    parser.add_argument("--run-name", type=str, default="ridge_hybrid")
    return parser.parse_args()


def flatten_sequences(seqs: np.ndarray) -> np.ndarray:
    if len(seqs) == 0:
        return np.empty((0, 0), dtype=np.float32)
    return seqs.reshape(seqs.shape[0], -1)


def train_ridge(X: np.ndarray, y: np.ndarray, alpha: float) -> Ridge:
    model = Ridge(alpha=alpha)
    model.fit(X, y)
    return model


def walkforward_ridge_pipeline(args):
    df = merge_all_series()
    true_cols, pred_cols = align_true_pred_columns(df)
    df = df.sort_values("date").reset_index(drop=True)

    if "y_ret" not in df.columns:
        raise ValueError("Target y_ret missing after merge.")
    y = df["y_ret"].values.astype(np.float32)

    split_idx = int(len(df) * 0.6)
    val_size = max(int(len(df) * 0.06), 8)
    train_end = split_idx - val_size
    if train_end <= args.context_len:
        raise ValueError("Not enough data for requested context length.")

    train_true = df.iloc[:train_end][true_cols]
    keep_cols = drop_near_constant_columns(train_true)
    keep_pred = [c for c in pred_cols if c.replace("_pred", "_true") in keep_cols]
    df = df[keep_cols + keep_pred + ["date", "sp500_close", "y_ret"]]
    true_cols, pred_cols = align_true_pred_columns(df)

    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end][true_cols].to_numpy())
    true_scaled = scaler.transform(df[true_cols].to_numpy())
    pred_scaled = scaler.transform(df[pred_cols].to_numpy())

    train_seq, train_targets = build_sequence_arrays(true_scaled, y, args.context_len, args.context_len, train_end)
    if len(train_seq) == 0:
        raise RuntimeError("No training sequences available for Ridge baseline.")
    X_train = flatten_sequences(train_seq)
    ridge = train_ridge(X_train, train_targets, args.alpha)

    rolling_preds: List[pd.DataFrame] = []
    rolling_metrics: List[RollingWindowResult] = []
    cursor = split_idx
    window_idx = 1
    train_limit = train_end

    while cursor < len(df):
        future_end = min(cursor + args.horizon_weeks, len(df))
        if future_end - cursor <= 0:
            break

        future_seq, y_future = build_sequence_arrays(pred_scaled, y, args.context_len, cursor, future_end)
        if len(future_seq) == 0:
            break
        X_pred = flatten_sequences(future_seq)
        pred_vals = ridge.predict(X_pred)
        start_idx = max(args.context_len, cursor)
        future_dates = df["date"].iloc[start_idx:future_end].values

        rmse = np.sqrt(mean_squared_error(y_future, pred_vals))
        r2 = r2_score(y_future, pred_vals)
        dir_acc = compute_directional_accuracy(y_future, pred_vals)
        rolling_metrics.append(
            RollingWindowResult(
                window=window_idx,
                start_date=pd.to_datetime(future_dates[0]),
                end_date=pd.to_datetime(future_dates[-1]),
                rmse=rmse,
                r2=r2,
                direction_acc=dir_acc,
            )
        )
        rolling_preds.append(
            pd.DataFrame(
                {
                    "date": future_dates,
                    "y_true": y_future,
                    "y_pred_mean": pred_vals,
                    "y_pred_p10": pred_vals,
                    "y_pred_p50": pred_vals,
                    "y_pred_p90": pred_vals,
                }
            )
        )

        new_seq, new_targets = build_sequence_arrays(true_scaled, y, args.context_len, train_limit, future_end)
        if len(new_seq):
            X_new = flatten_sequences(new_seq)
            X_train = np.concatenate([X_train, X_new], axis=0)
            train_targets = np.concatenate([train_targets, new_targets], axis=0)
            ridge = train_ridge(X_train, train_targets, args.alpha)
            train_limit = future_end

        cursor += args.step_weeks
        window_idx += 1

    if not rolling_preds:
        raise RuntimeError("No rolling predictions produced.")

    results_df = pd.concat(rolling_preds, ignore_index=True).sort_values("date")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = RESULTS_DIR / "rolling_predictions_ridge.csv"
    metrics_path = RESULTS_DIR / "rolling_metrics_ridge.csv"
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
    metrics_df.to_csv(metrics_path, index=False)

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
                "alpha": args.alpha,
                "rmse": overall_rmse,
                "r2": overall_r2,
                "directional_accuracy": overall_dir,
                "mode": "ridge_hybrid",
            }
        ]
    )
    log_path = RESULTS_DIR / "hyperparam_runs_ridge.csv"
    log_row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)

    print(
        f"[Ridge Hybrid] {args.run_name} -> RMSE={overall_rmse:.4f} "
        f"R^2={overall_r2:.4f} DirAcc={overall_dir:.3f} | saved {preds_path}"
    )


def main():
    args = parse_args()
    walkforward_ridge_pipeline(args)


if __name__ == "__main__":
    main()
