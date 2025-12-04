import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timesfm
from sklearn.metrics import mean_absolute_error, mean_squared_error
from timesfm import ForecastConfig

from fred_utils import load_series as load_fred_series


MODEL_NAME = "google/timesfm-2.5-200m-pytorch"


@dataclass(frozen=True)
class MacroTarget:
    name: str
    data_path: Path
    value_col: str
    output_prefix: str
    fred_series: str | None = None


TARGETS: dict[str, MacroTarget] = {
    "dgs10": MacroTarget("dgs10", Path("data/raw/fred_10y.csv"), "dgs10", "dgs10"),
    "baa10ym": MacroTarget("baa10ym", Path("data/raw/baa10ym.csv"), "baa10ym", "baa10ym"),
    "cpiaucsl": MacroTarget("cpiaucsl", Path("data/raw/cpiaucsl.csv"), "cpiaucsl", "cpiaucsl"),
    "dcoilwtico": MacroTarget("dcoilwtico", Path("data/raw/dcoilwtico.csv"), "dcoilwtico", "dcoilwtico"),
    "fedfunds": MacroTarget("fedfunds", Path("data/raw/fedfunds.csv"), "fedfunds", "fedfunds"),
    "indpro": MacroTarget("indpro", Path("data/raw/indpro.csv"), "indpro", "indpro"),
    "t10y2y": MacroTarget("t10y2y", Path("data/raw/t10y2y.csv"), "t10y2y", "t10y2y"),
    "unrate": MacroTarget("unrate", Path("data/raw/unrate.csv"), "unrate", "unrate"),
    "gs2": MacroTarget("gs2", Path("data/raw/gs2.csv"), "gs2", "gs2", fred_series="GS2"),
}


def load_csv_series(path: Path, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a date column.")
    col = value_col.lower()
    if col not in df.columns:
        candidates = [c for c in df.columns if c != "date"]
        if len(candidates) != 1:
            raise ValueError(f"Unable to identify value column in {path}")
        col = candidates[0]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df.set_index("date")[col].astype(float).dropna()


def load_target_series(target: MacroTarget) -> pd.Series:
    if target.fred_series:
        return load_fred_series(target.data_path, target.fred_series, target.value_col)
    return load_csv_series(target.data_path, target.value_col)


def load_timesfm_model(context_len: int, horizon: int) -> timesfm.TimesFM_2p5_200M_torch:
    print(f"[TimesFM] Loading {MODEL_NAME} ...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(MODEL_NAME)
    config = ForecastConfig(
        max_context=context_len,
        max_horizon=horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=False,
    )
    model.compile(config)
    print("[TimesFM] Model ready.")
    return model


def run_backtest(
    model: timesfm.TimesFM_2p5_200M_torch,
    series: pd.Series,
    target: MacroTarget,
    context_len: int,
    horizon: int,
    stride: int,
    output_dir: Path,
):
    preds, trues, pred_dates = [], [], []
    print(f"[Backtest] {target.name} | len={len(series)} context={context_len} horizon={horizon} stride={stride}")
    for i in range(context_len, len(series) - horizon, stride):
        context_slice = series.iloc[i - context_len : i].values
        true_future = series.iloc[i : i + horizon].values
        forecast, _ = model.forecast(horizon=horizon, inputs=[context_slice])
        preds.extend(forecast[0])
        trues.extend(true_future)
        pred_dates.extend(series.index[i : i + horizon])

    if not preds:
        raise RuntimeError(f"No predictions generated for {target.name}; check context/horizon settings.")

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    print(f"[Results] {target.name} -> RMSE={rmse:.4f}  MAE={mae:.4f}  ({len(preds)} predictions)")

    output_dir.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(
        {
            "date": pred_dates,
            f"{target.output_prefix}_true": trues,
            f"{target.output_prefix}_pred": preds,
        }
    )
    out_csv = output_dir / f"{target.output_prefix}_backtest.csv"
    results.to_csv(out_csv, index=False)
    print(f"[Save] CSV -> {out_csv}")

    plt.figure(figsize=(10, 4))
    plt.plot(results["date"], results[f"{target.output_prefix}_true"], label="True")
    plt.plot(results["date"], results[f"{target.output_prefix}_pred"], label="Predicted", linestyle="--")
    plt.title(f"TimesFM Rolling Backtest ({target.output_prefix.upper()})")
    plt.legend()
    plt.tight_layout()
    fig_path = output_dir / f"{target.output_prefix}_backtest.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"[Save] Figure -> {fig_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Batch TimesFM predictions for macro indicators.")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        help=f"Which targets to run (choices: {', '.join(sorted(TARGETS.keys()))} or 'all').",
    )
    parser.add_argument("--context-len", type=int, default=512)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="data/predictions")
    return parser.parse_args()


def main():
    args = parse_args()
    if "all" in args.targets:
        selected: List[str] = sorted(TARGETS.keys())
    else:
        for t in args.targets:
            if t not in TARGETS:
                raise ValueError(f"Unknown target '{t}'. Valid options: {', '.join(sorted(TARGETS.keys()))}")
        selected = args.targets

    model = load_timesfm_model(args.context_len, args.horizon)
    output_dir = Path(args.output_dir)
    for name in selected:
        target = TARGETS[name]
        series = load_target_series(target)
        run_backtest(model, series, target, args.context_len, args.horizon, args.stride, output_dir)


if __name__ == "__main__":
    main()
