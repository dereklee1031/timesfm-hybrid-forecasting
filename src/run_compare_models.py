import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

import transformer as tt


def _extract_arg(arg_list, flag, default=None):
    if flag not in arg_list:
        return default
    idx = arg_list.index(flag)
    if idx == -1 or idx + 1 >= len(arg_list):
        return default
    value = arg_list[idx + 1]
    try:
        if "." in value or "e" in value.lower():
            return float(value)
        return int(value)
    except ValueError:
        return value


def run_transformer(transformer_args: list[str]):
    cmd = ["python3", "src/transformer.py", *transformer_args]
    print(f"[Runner] Executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    pred_path = Path("results/rolling_predictions.csv")
    metrics_path = Path("results/rolling_metrics.csv")
    if not pred_path.exists():
        raise FileNotFoundError(f"{pred_path} missing after transformer run.")
    preds_df = pd.read_csv(pred_path, parse_dates=["date"]).sort_values("date")
    metrics_df = pd.read_csv(metrics_path)
    return preds_df, metrics_df


def compute_overall_stats(df: pd.DataFrame, pred_col: str) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(df["y_true"], df[pred_col])))
    r2 = float(r2_score(df["y_true"], df[pred_col]))
    dir_acc = float(np.mean(np.sign(df["y_true"]) == np.sign(df[pred_col])))
    return {"rmse": rmse, "r2": r2, "directional_accuracy": dir_acc}


def run_ridge(alpha: float, context_len: int, horizon_weeks: int, step_weeks: int):
    df = tt.merge_all_series()
    true_cols, pred_cols = tt.align_true_pred_columns(df)
    y = df["y_ret"].values.astype(np.float32)

    split_idx = int(len(df) * 0.6)
    val_size = max(int(len(df) * 0.06), 8)
    train_end = split_idx - val_size
    if train_end <= context_len:
        raise ValueError("Not enough data for requested context length.")

    keep_cols = tt.drop_near_constant_columns(df.iloc[:train_end][true_cols])
    df = df[keep_cols + [c for c in pred_cols if c.replace("_pred", "_true") in keep_cols] + ["date", "y_ret"]]
    true_cols, pred_cols = tt.align_true_pred_columns(df)

    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end][true_cols].to_numpy())
    true_scaled = scaler.transform(df[true_cols].to_numpy())
    pred_scaled = scaler.transform(df[pred_cols].to_numpy())

    lagged_X, lagged_y = tt.build_lagged_features(true_scaled, y, context_len, train_end)
    if len(lagged_X) == 0:
        raise RuntimeError("Ridge baseline has no training rows.")
    ridge = Ridge(alpha=alpha).fit(lagged_X, lagged_y)

    predictions = []
    metrics = []
    cursor = split_idx
    window_idx = 1
    predicted_indices = set()

    while cursor < len(df):
        future_end = min(cursor + horizon_weeks, len(df))
        scenario_features = []
        scenario_indices = []
        scenario_targets = []
        for idx in range(cursor, future_end):
            if idx < context_len or idx in predicted_indices:
                continue
            scenario_features.append(pred_scaled[idx])
            scenario_indices.append(idx)
            scenario_targets.append(y[idx])
        if not scenario_features:
            break
        future_features = np.array(scenario_features)
        future_dates = df["date"].iloc[scenario_indices].values
        y_future = np.array(scenario_targets)

        preds = tt.baseline_predict_sequence(
            ridge,
            true_scaled[:cursor],
            y[:cursor],
            future_features,
        )

        rmse = float(np.sqrt(mean_squared_error(y_future, preds)))
        r2 = float(r2_score(y_future, preds))
        dir_acc = float(np.mean(np.sign(y_future) == np.sign(preds)))
        metrics.append(
            {
                "window": window_idx,
                "start_date": future_dates[0],
                "end_date": future_dates[-1],
                "rmse": rmse,
                "r2": r2,
                "directional_accuracy": dir_acc,
            }
        )
        predictions.append(
            pd.DataFrame(
                {
                    "date": future_dates,
                    "y_true": y_future,
                    "y_pred_ridge": preds,
                }
            )
        )
        predicted_indices.update(scenario_indices)
        cursor += step_weeks
        window_idx += 1

    if not predictions:
        raise RuntimeError("Ridge produced no predictions.")

    preds_df = pd.concat(predictions, ignore_index=True).sort_values("date")
    metrics_df = pd.DataFrame(metrics)
    preds_path = Path("results/ridge_predictions.csv")
    metrics_path = Path("results/ridge_metrics.csv")
    preds_df.to_csv(preds_path, index=False)
    metrics_df.to_csv(metrics_path, index=False)
    print(f"[Ridge] Saved predictions -> {preds_path}")
    print(f"[Ridge] Saved metrics -> {metrics_path}")
    return preds_df, metrics_df


def main():
    parser = argparse.ArgumentParser(description="Run train_transformer and Ridge, compare results.")
    parser.add_argument("--skip-transformer", action="store_true", help="Assume transformer outputs already exist.")
    parser.add_argument("--transformer-args", nargs=argparse.REMAINDER, default=[], help="Arguments passed to train_transformer.py")
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--context-len", type=int, default=52)
    parser.add_argument("--horizon-weeks", type=int, default=20)
    parser.add_argument("--step-weeks", type=int, default=1)
    args = parser.parse_args()

    if args.skip_transformer:
        transformer_preds = pd.read_csv("results/rolling_predictions.csv", parse_dates=["date"]).sort_values("date")
        print("[Runner] Skipping transformer run; using existing results/rolling_predictions.csv")
    else:
        transformer_preds, _ = run_transformer(args.transformer_args)

    ridge_preds, _ = run_ridge(args.ridge_alpha, args.context_len, args.horizon_weeks, args.step_weeks)

    transformer_stats = compute_overall_stats(transformer_preds, "y_pred_mean")
    ridge_stats = compute_overall_stats(ridge_preds, "y_pred_ridge")

    summary = pd.DataFrame(
        [
            {"model": "Transformer", **transformer_stats},
            {"model": "Ridge", **ridge_stats},
        ]
    )
    summary_path = Path("results/model_comparison.csv")
    summary.to_csv(summary_path, index=False)
    print("\n=== Model Comparison ===")
    print(summary.to_string(index=False))
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()
