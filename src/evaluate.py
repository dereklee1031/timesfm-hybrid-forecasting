"""Baseline evaluation utilities for the forecasting project.

This script implements a simple persistence / lag-based baseline that mimics
the rolling 10-day backtest we run for TimesFM. It predicts that the next
`horizon` values will match the most recent observed value (AR(1) persistence)
and rolls forward with the same stride, so we can quantify how much lift
the two-stage pipeline provides over this naive benchmark.
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


HYBRID_DEFAULT_START = "2014-05-19"
HYBRID_DEFAULT_END = "2024-12-31"


# Named regime-shift windows (inclusive bounds) based on major market stress events.
STRESS_WINDOWS = {
    "dotcom": ("2000-03-10", "2002-10-09"),
    "gfc": ("2007-07-01", "2009-06-30"),
    "flash_crash": ("2010-04-01", "2010-07-31"),
    "euro_debt": ("2011-07-01", "2012-06-30"),
    "covid": ("2020-02-01", "2020-09-30"),
    "inflation": ("2021-11-01", "2022-12-31"),
}


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


def run_window(series: pd.Series, context_len: int, horizon: int, stride: int):
    """Run persistence baseline on the provided window and guard against empty results."""
    preds, trues, dates = rolling_persistence(series, context_len, horizon, stride)
    if len(preds) == 0:
        raise ValueError(
            "Evaluation window is too short for the requested context/horizon/stride configuration."
        )
    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    return preds, trues, dates, rmse, mae


def compute_directional_accuracy(series_true: np.ndarray, series_pred: np.ndarray) -> float:
    """Directional accuracy based on consecutive changes in the series."""
    if len(series_true) < 2 or len(series_pred) < 2:
        return float("nan")
    true_diff = np.diff(series_true)
    pred_diff = np.diff(series_pred)
    min_len = min(len(true_diff), len(pred_diff))
    if min_len == 0:
        return float("nan")
    true_sign = np.sign(true_diff[:min_len])
    pred_sign = np.sign(pred_diff[:min_len])
    return float(np.mean(true_sign == pred_sign))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate naive rolling baseline for price series.")
    parser.add_argument("--data-path", type=str, default="data/raw/sp500.csv")
    parser.add_argument("--value-col", type=str, default="sp500_close")
    parser.add_argument("--context", type=int, default=16)
    parser.add_argument("--horizon", type=int, default=14)  # ~2 weeks of trading days
    parser.add_argument("--stride", type=int, default=7)
    parser.add_argument("--output", type=str, default="results/baseline_persistence.csv")
    parser.add_argument(
        "--window-start",
        type=str,
        default=HYBRID_DEFAULT_START,
        help="Inclusive start date for the main evaluation window (default aligns with hybrid training).",
    )
    parser.add_argument(
        "--window-end",
        type=str,
        default=HYBRID_DEFAULT_END,
        help="Inclusive end date for the main evaluation window (default aligns with hybrid training).",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Ignore --window-start/--window-end and use the entire series for the baseline evaluation.",
    )
    parser.add_argument("--stress-start", type=str, help="Inclusive start date for stress window (YYYY-MM-DD).")
    parser.add_argument("--stress-end", type=str, help="Inclusive end date for stress window (YYYY-MM-DD).")
    parser.add_argument(
        "--stress-output",
        type=str,
        help="Optional CSV output path for stress-window predictions. Defaults to appending _stress to --output.",
    )
    parser.add_argument(
        "--stress-label",
        type=str,
        default="stress",
        help="Suffix used for logging and filenames when stress window is enabled.",
    )
    parser.add_argument(
        "--stress-only",
        action="store_true",
        help="Skip the full-series baseline and evaluate only the requested stress windows.",
    )
    if STRESS_WINDOWS:
        parser.add_argument(
            "--stress-preset",
            action="append",
            choices=sorted(STRESS_WINDOWS.keys()),
            help="Run additional evaluations for one or more named stress regimes (can be provided multiple times).",
        )
    return parser.parse_args()


def subset_series(series: pd.Series, start: Optional[str], end: Optional[str]) -> pd.Series:
    """Slice series to requested window while keeping date index."""
    start_ts = pd.to_datetime(start) if start else series.index.min()
    end_ts = pd.to_datetime(end) if end else series.index.max()
    window = series.loc[start_ts:end_ts]
    if window.empty:
        raise ValueError(
            f"Stress window returned no data. Start={start_ts.date()} End={end_ts.date()} within [{series.index.min().date()}, {series.index.max().date()}]."
        )
    return window


def evaluate_and_save(
    series: pd.Series,
    context: int,
    horizon: int,
    stride: int,
    value_col: str,
    output_path: Path,
    label: str,
):
    preds, trues, dates, rmse, mae = run_window(series, context, horizon, stride)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df = pd.DataFrame(
        {
            "date": dates,
            "y_true": trues,
            "y_pred": preds,
        }
    )
    results_df.sort_values("date").to_csv(output_path, index=False)
    print(f"[{label}] Saved predictions -> {output_path}")

    plot_df = (
        results_df.groupby("date", as_index=False)
        .agg({"y_true": "last", "y_pred": "mean"})
        .sort_values("date")
    )
    dir_acc = compute_directional_accuracy(plot_df["y_true"].to_numpy(), plot_df["y_pred"].to_numpy())
    dir_str = f"{dir_acc:.4f}" if np.isfinite(dir_acc) else "nan"
    print(f"[{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  DirAcc={dir_str}  ({len(preds)} predictions)")
    plt.figure(figsize=(10, 4))
    plt.plot(plot_df["date"], plot_df["y_true"], label="True")
    plt.plot(plot_df["date"], plot_df["y_pred"], label="Baseline", linestyle="--")
    plt.title(f"Baseline Persistence vs True Values ({label})")
    plt.xlabel("Date")
    plt.ylabel(value_col)
    plt.legend()
    plt.tight_layout()
    plot_path = output_path.with_suffix(".png")
    plt.savefig(plot_path, dpi=300)
    print(f"[{label}] Saved plot -> {plot_path}")


def main():
    args = parse_args()
    if args.stress_only and not (args.stress_start or args.stress_end or args.stress_preset):
        raise ValueError("--stress-only requires --stress-start/--stress-end or --stress-preset.")
    df = pd.read_csv(args.data_path, parse_dates=["date"]).sort_values("date")
    full_series = df.set_index("date")[args.value_col].dropna()
    if args.full_history:
        eval_series = full_series
    else:
        eval_series = subset_series(full_series, args.window_start, args.window_end)
    out_path = Path(args.output)
    if not args.stress_only:
        evaluate_and_save(
            series=eval_series,
            context=args.context,
            horizon=args.horizon,
            stride=args.stride,
            value_col=args.value_col,
            output_path=out_path,
            label="Baseline",
        )

    stress_start = args.stress_start
    stress_end = args.stress_end
    if stress_start or stress_end:
        stress_series = subset_series(full_series, stress_start, stress_end)
        if args.stress_output:
            stress_out = Path(args.stress_output)
        else:
            stress_out = out_path.with_name(f"{out_path.stem}_{args.stress_label}{out_path.suffix}")
        evaluate_and_save(
            series=stress_series,
            context=args.context,
            horizon=args.horizon,
            stride=args.stride,
            value_col=args.value_col,
            output_path=stress_out,
            label=f"Stress ({args.stress_label})",
        )
    elif args.stress_output:
        print("[Stress] Ignored --stress-output because no stress window was specified.")

    if args.stress_preset:
        for preset in args.stress_preset:
            start, end = STRESS_WINDOWS[preset]
            preset_series = subset_series(full_series, start, end)
            preset_out = out_path.with_name(f"{out_path.stem}_{preset}{out_path.suffix}")
            evaluate_and_save(
                series=preset_series,
                context=args.context,
                horizon=args.horizon,
                stride=args.stride,
                value_col=args.value_col,
                output_path=preset_out,
                label=f"Stress ({preset})",
            )


if __name__ == "__main__":
    main()
