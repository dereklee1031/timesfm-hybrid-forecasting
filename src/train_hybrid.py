import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

#Load data actual and predicted datasets 
InterestR_df = pd.read_csv("data/predictions/dgs10_backtest.csv", parse_dates=["date"])
InterestR_df = InterestR_df.sort_values("date").dropna()

credit_spread_df = pd.read_csv("data/predictions/baa10ym_backtest.csv", parse_dates=["date"])
credit_spread_df = credit_spread_df.sort_values("date").dropna()

inflation_df = pd.read_csv("data/predictions/cpiaucsl_backtest.csv", parse_dates=["date"])
inflation_df = inflation_df.sort_values("date").dropna()

oil_price_df = pd.read_csv("data/predictions/dcoilwtico_backtest.csv", parse_dates=["date"])
oil_price_df = oil_price_df.sort_values("date").dropna()

fed_funds_df = pd.read_csv("data/predictions/fedfunds_backtest.csv", parse_dates=["date"])
fed_funds_df = fed_funds_df.sort_values("date").dropna()

industrial_prod_df = pd.read_csv("data/predictions/indpro_backtest.csv", parse_dates=["date"])
industrial_prod_df = industrial_prod_df.sort_values("date").dropna()

yield_spread_df = pd.read_csv("data/predictions/t10y2y_backtest.csv", parse_dates=["date"])
yield_spread_df = yield_spread_df.sort_values("date").dropna()

unemployment_rate_df = pd.read_csv("data/predictions/unrate_backtest.csv", parse_dates=["date"])
unemployment_rate_df = unemployment_rate_df.sort_values("date").dropna()

#Load SP500 data
sp500_df = pd.read_csv("data/raw/sp500.csv", parse_dates=["date"])
sp500_df = sp500_df.sort_values("date").dropna()

#Merge datasets focus on daiy and monthly data
# Put all dataframes into a list for easy iteration
macro_dfs = [
    InterestR_df,
    credit_spread_df,
    inflation_df,
    oil_price_df,
    fed_funds_df,
    industrial_prod_df,
    yield_spread_df,
    unemployment_rate_df
]

# Resample to daily frequency and forward-fill
for i in range(len(macro_dfs)):
    df = macro_dfs[i].copy()
    df = df.set_index("date").resample("D").ffill().reset_index()
    macro_dfs[i] = df

# Reassign the resampled versions back
(
    InterestR_df,
    credit_spread_df,
    inflation_df,
    oil_price_df,
    fed_funds_df,
    industrial_prod_df,
    yield_spread_df,
    unemployment_rate_df
) = macro_dfs

from functools import reduce

dfs = [
    InterestR_df,
    credit_spread_df,
    inflation_df,
    oil_price_df,
    fed_funds_df,
    industrial_prod_df,
    yield_spread_df,
    unemployment_rate_df,
    sp500_df
]

df_merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs)
df_merged = df_merged.sort_values("date").fillna(method="ffill").dropna().reset_index(drop=True)

# from functools import reduce

# df_merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="inner"), dfs)
# df_merged = df_merged.sort_values("date").dropna().reset_index(drop=True)

print("Merged shape:", df_merged.shape)
print(df_merged.head())

print("After merge:", df_merged.shape)
print("Date range:", df_merged["date"].min(), "→", df_merged["date"].max())
print(df_merged.tail())

# Collect column names
true_features = [col for col in df_merged.columns if col.endswith("_true")]
pred_features = [col for col in df_merged.columns if col.endswith("_pred")]

print("True features:", true_features)
print("Pred features:", pred_features)

input_dim = len(true_features)

class MLPRegressor(nn.Module):
    def __init__(self, input_dim):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

num_epochs = 200
patience = 15
horizon = 200  # rolling prediction horizon in days

total_rows = len(df_merged)
if total_rows <= horizon:
    raise ValueError("Not enough rows to support the requested rolling horizon.")

initial_train_size = max(horizon, int(total_rows * 0.6))
if initial_train_size + horizon > total_rows:
    initial_train_size = total_rows - horizon
if initial_train_size <= 0:
    raise ValueError("Insufficient history for rolling forecast setup.")

print(f"Rolling forecast configuration -> initial train rows: {initial_train_size}, horizon: {horizon}")

rolling_predictions = []
rolling_truth = []
rolling_dates = []

forecast_starts = list(range(initial_train_size, total_rows - horizon + 1, horizon))
if not forecast_starts:
    forecast_starts = [initial_train_size]

for window_idx, forecast_start in enumerate(forecast_starts, start=1):
    train_slice = df_merged.iloc[:forecast_start]
    future_slice = df_merged.iloc[forecast_start:forecast_start + horizon]

    val_size = max(1, int(0.1 * len(train_slice)))
    if len(train_slice) - val_size < 1:
        val_size = len(train_slice) - 1
    if val_size <= 0:
        raise ValueError("Increase history length; validation split would be empty.")

    train_sub = train_slice.iloc[:-val_size]
    val_sub = train_slice.iloc[-val_size:]

    X_train = train_sub[true_features].values
    y_train = train_sub["sp500_close"].values
    X_val = val_sub[true_features].values
    y_val = val_sub["sp500_close"].values

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))
    X_val_scaled = scaler_X.transform(X_val)
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
    X_val_t = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_t = torch.tensor(y_val_scaled, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset = TensorDataset(X_val_t, y_val_t)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    model = MLPRegressor(input_dim=input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    best_val_loss = np.inf
    patience_counter = 0
    best_state_dict = None

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(Xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * Xb.size(0)
        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        with torch.no_grad():
            val_preds = model(X_val_t)
            val_loss = criterion(val_preds, y_val_t).item()

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            patience_counter = 0
            best_state_dict = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    X_future = future_slice[pred_features].values
    y_future = future_slice["sp500_close"].values
    X_future_scaled = scaler_X.transform(X_future)
    X_future_t = torch.tensor(X_future_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        future_pred_scaled = model(X_future_t).numpy()

    future_pred = scaler_y.inverse_transform(future_pred_scaled).ravel()

    window_rmse = np.sqrt(mean_squared_error(y_future, future_pred))
    window_r2 = r2_score(y_future, future_pred)
    start_date = future_slice["date"].iloc[0].date()
    end_date = future_slice["date"].iloc[-1].date()
    print(f"Window {window_idx}: {start_date} → {end_date} | RMSE {window_rmse:.4f} | R^2 {window_r2:.4f}")

    rolling_predictions.append(future_pred)
    rolling_truth.append(y_future)
    rolling_dates.append(future_slice["date"].values)

all_pred = np.concatenate(rolling_predictions)
all_true = np.concatenate(rolling_truth)
all_dates = pd.to_datetime(np.concatenate(rolling_dates))

overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
overall_r2 = r2_score(all_true, all_pred)
print(f"Rolling horizon {horizon} days | Overall RMSE: {overall_rmse:.4f}, R^2: {overall_r2:.4f}")

results_df = pd.DataFrame(
    {
        "date": all_dates,
        "sp500_true": all_true,
        "sp500_pred": all_pred,
    }
)
results_df.to_csv("results/rolling_predictions.csv", index=False)

plt.figure(figsize=(12, 5))
plt.plot(results_df["date"], results_df["sp500_true"], label="True")
plt.plot(results_df["date"], results_df["sp500_pred"], label="Predicted")
plt.title(f"Rolling S&P500 Forecast (horizon={horizon} days)")
plt.legend()
plt.tight_layout()
plt.show()
