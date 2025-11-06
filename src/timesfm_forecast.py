import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import timesfm
from timesfm import ForecastConfig

#Load Fred data 
df = pd.read_csv("data/raw/fred_10y.csv", parse_dates=["date"])
df = df.sort_values("date")
series = df.set_index("date")["dgs10"].dropna()
print(series.head())

#model loading and configuration
print("Loading TimesFM model...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch")
print("Model loaded.")

#later we can vary these parameters
context_len = 512      # how many past data points to use
horizon = 10           # forecast 10 future days

config = ForecastConfig(
    max_context=context_len,
    max_horizon=horizon,
    normalize_inputs=True,
    use_continuous_quantile_head=False,
)
model.compile(config)
print("Model compiled.")

#Rolling walk-forward evaluation

#more parameters we can vary
stride = 20  #how far to roll the window each time (≈ monthly)
preds, trues, pred_dates = [], [], []

print("[Backtest] Starting rolling evaluation...")
for i in range(context_len, len(series) - horizon, stride):
    # Slice last `context_len` points as input
    context_slice = series.iloc[i - context_len : i].values
    true_future = series.iloc[i : i + horizon].values

    # make forecast
    forecast, _ = model.forecast(horizon=horizon, inputs=[context_slice])

    # store results
    preds.extend(forecast[0])
    trues.extend(true_future)
    pred_dates.extend(series.index[i : i + horizon])

# Compute backtest metrics
rmse = np.sqrt(mean_squared_error(trues, preds))
mae = mean_absolute_error(trues, preds)
print(f"[Results] RMSE={rmse:.4f}  MAE={mae:.4f}  ({len(preds)} predictions)\n")

# Save results of backtest
results = pd.DataFrame({
    "date": pred_dates,
    "dgs10_true": trues,
    "dgs10_pred": preds,
})
Path("data/predictions").mkdir(parents=True, exist_ok=True)
results.to_csv("data/predictions/dgs10_backtest.csv", index=False)
print("Saved backtest predictions -> data/predictions/dgs10_backtest.csv")

# Plot results
plt.figure(figsize=(10, 4))
plt.plot(results["date"], results["dgs10_true"], label="True")
plt.plot(results["date"], results["dgs10_pred"], label="Predicted", linestyle="--")
plt.title("TimesFM Rolling Backtest (10-day horizon)")
plt.legend(); plt.tight_layout()

fig_path = Path("data/predictions/dgs10_backtest.png")
plt.savefig(fig_path, dpi=300)
print(f"Saved backtest figure -> {fig_path}")
plt.show()
