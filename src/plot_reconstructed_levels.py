import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_predictions(pred_path: Path) -> pd.DataFrame:
    df = pd.read_csv(pred_path, parse_dates=["date"]).sort_values("date")
    for col in ["y_true", "y_pred_mean"]:
        if col not in df.columns:
            raise ValueError(f"{pred_path} missing required column '{col}'.")
    return df


def load_sp500(base_path: Path) -> pd.DataFrame:
    df = pd.read_csv(base_path, parse_dates=["date"]).sort_values("date")
    if "sp500_close" not in df.columns:
        raise ValueError(f"{base_path} must include 'sp500_close'.")
    return df.set_index("date")


def reconstruct_levels(pred_df: pd.DataFrame, sp500_df: pd.DataFrame) -> pd.DataFrame:
    start_date = pred_df["date"].min()
    if start_date < sp500_df.index.min():
        raise ValueError(f"Start date {start_date.date()} precedes available sp500 data.")
    base_series = sp500_df.reindex(sp500_df.index.union([start_date])).sort_index().ffill()
    base_price = base_series.loc[start_date, "sp500_close"]
    levels = pred_df[["date"]].copy()
    levels["cum_true"] = np.exp(pred_df["y_true"].cumsum()) * base_price
    levels["cum_pred"] = np.exp(pred_df["y_pred_mean"].cumsum()) * base_price
    return levels


def plot_levels(levels_df: pd.DataFrame, output_path: Path):
    plt.figure(figsize=(12, 5))
    plt.plot(levels_df["date"], levels_df["cum_true"], label="True (reconstructed)")
    plt.plot(levels_df["date"], levels_df["cum_pred"], label="Predicted (reconstructed)")
    plt.title("Reconstructed S&P 500 Levels from Weekly Log Returns")
    plt.ylabel("Pseudo Level")
    plt.xlabel("Date")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved reconstructed level plot -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Reconstruct pseudo S&P 500 levels from log-return predictions.")
    parser.add_argument("--predictions", type=str, default="results/rolling_predictions.csv")
    parser.add_argument("--sp500", type=str, default="data/raw/sp500.csv")
    parser.add_argument("--output", type=str, default="results/transformer_levels.png")
    args = parser.parse_args()

    pred_df = load_predictions(Path(args.predictions))
    sp500_df = load_sp500(Path(args.sp500))
    levels_df = reconstruct_levels(pred_df, sp500_df)
    levels_df.to_csv(Path(args.output).with_suffix(".csv"), index=False)
    plot_levels(levels_df, Path(args.output))


if __name__ == "__main__":
    main()
