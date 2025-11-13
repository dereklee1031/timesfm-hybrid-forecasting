"""Baseline evaluation utilities for the forecasting project.

This script implements a simple persistence / lag-based baseline that mimics
the rolling 10-day backtest we run for TimesFM. It predicts that the next
`horizon` values will match the most recent observed value (AR(1) persistence)
and rolls forward with the same stride, so we can quantify how much lift
the two-stage pipeline provides over this naive benchmark.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rolling_persistence(series: pd.Series, context_len: int, horizon: int, stride: int):
    preds, trues, dates = [], [], []
    for i in range(context_len, len(series) - horizon, stride):
        context_slice = series.iloc[i - context_len : i]
        last_val = context_slice.iloc[-1]
        true_future = series.iloc[i : i + horizon].values
        preds.extend([last_val] * horizon)
        trues.extend(true_future)
        dates.extend(series.index[i : i + horizon])
    return np.array(preds), np.array(trues), dates


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate naive rolling baseline for price series.")
    parser.add_argument("--data-path", type=str, default="data/raw/sp500.csv")
    parser.add_argument("--value-col", type=str, default="sp500_close")
    parser.add_argument("--context", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=100)  # ~2 months of trading days
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output", type=str, default="results/baseline_persistence.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.data_path, parse_dates=["date"]).sort_values("date")
    series = df.set_index("date")[args.value_col].dropna()
    preds, trues, dates = rolling_persistence(series, args.context, args.horizon, args.stride)

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    print(f"[Baseline Persistence] RMSE={rmse:.4f}  MAE={mae:.4f}  ({len(preds)} predictions)")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "date": dates,
        "y_true": trues,
        "y_pred": preds,
    }).to_csv(out_path, index=False)
    print(f"Saved baseline predictions -> {out_path}")

    plt.figure(figsize=(10, 4))
    plt.plot(dates, trues, label="True")
    plt.plot(dates, preds, label="Baseline", linestyle="--")
    plt.title("Baseline Persistence vs True Values")
    plt.xlabel("Date")
    plt.ylabel(args.value_col)
    plt.legend()
    plt.tight_layout()
    plot_path = out_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved baseline plot -> {plot_path}")


if __name__ == "__main__":
    main()
