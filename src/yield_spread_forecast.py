import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import timesfm
from timesfm import ForecastConfig

DATA_PATH = Path("data/raw/t10y2y.csv")
VALUE_COL = "t10y2y"
OUTPUT_PREFIX = "t10y2y"


def load_series(path: Path, value_col: str) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError("CSV must contain a date column.")
    col = value_col.lower()
    if col not in df.columns:
        candidates = [c for c in df.columns if c != "date"]
        if len(candidates) != 1:
            raise ValueError(f"Unable to identify value column in {path}")
        col = candidates[0]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df.set_index("date")[col].astype(float).dropna()


def main() -> None:
    series = load_series(DATA_PATH, VALUE_COL)
    print(series.head())

    print("Loading TimesFM model...")
    model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
    print("Model loaded.")

    context_len = 512
    horizon = 10
    config = ForecastConfig(
        max_context=context_len,
        max_horizon=horizon,
        normalize_inputs=True,
        use_continuous_quantile_head=False,
    )
    model.compile(config)
    print("Model compiled.")

    stride = 20
    preds, trues, pred_dates = [], [], []
    for i in range(context_len, len(series) - horizon, stride):
        context_slice = series.iloc[i - context_len : i].values
        true_future = series.iloc[i : i + horizon].values
        forecast, _ = model.forecast(horizon=horizon, inputs=[context_slice])
        preds.extend(forecast[0])
        trues.extend(true_future)
        pred_dates.extend(series.index[i : i + horizon])

    rmse = np.sqrt(mean_squared_error(trues, preds))
    mae = mean_absolute_error(trues, preds)
    print(f"[Results] RMSE={rmse:.4f}  MAE={mae:.4f}  ({len(preds)} predictions)\n")

    Path("data/predictions").mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({
        "date": pred_dates,
        f"{OUTPUT_PREFIX}_true": trues,
        f"{OUTPUT_PREFIX}_pred": preds,
    })
    out_csv = Path("data/predictions") / f"{OUTPUT_PREFIX}_backtest.csv"
    results.to_csv(out_csv, index=False)
    print(f"Saved backtest predictions -> {out_csv}")

    plt.figure(figsize=(10, 4))
    plt.plot(results["date"], results[f"{OUTPUT_PREFIX}_true"], label="True")
    plt.plot(results["date"], results[f"{OUTPUT_PREFIX}_pred"], label="Predicted", linestyle="--")
    plt.title(f"TimesFM Rolling Backtest ({OUTPUT_PREFIX.upper()})")
    plt.legend()
    plt.tight_layout()
    fig_path = Path("data/predictions") / f"{OUTPUT_PREFIX}_backtest.png"
    plt.savefig(fig_path, dpi=300)
    print(f"Saved backtest figure -> {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
