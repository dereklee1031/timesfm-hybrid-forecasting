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


def make_tensor_dataset(df, feature_cols, scaler_X, scaler_y, fit_scalers=False):
    X = df[feature_cols].values
    y = df["sp500_close"].values.reshape(-1, 1)
    if fit_scalers:
        scaler_X.fit(X)
        scaler_y.fit(y)
    X_scaled = scaler_X.transform(X)
    y_scaled = scaler_y.transform(y)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_scaled, dtype=torch.float32)
    return TensorDataset(X_tensor, y_tensor)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_samples = 0
    for Xb, yb in loader:
        optimizer.zero_grad()
        preds = model(Xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * Xb.size(0)
        total_samples += Xb.size(0)
    return total_loss / max(total_samples, 1)


def evaluate_loss(model, loader, criterion):
    if loader is None:
        return None
    model.eval()
    total_loss = 0.0
    total_samples = 0
    with torch.no_grad():
        for Xb, yb in loader:
            preds = model(Xb)
            loss = criterion(preds, yb)
            total_loss += loss.item() * Xb.size(0)
            total_samples += Xb.size(0)
    return total_loss / max(total_samples, 1)


def train_with_early_stopping(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience):
    best_state = None
    best_val = np.inf
    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_loss(model, val_loader, criterion)
        print(f"[Initial Training] Epoch {epoch:03d} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}")
        if val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered during initial training.")
                break
    return best_state


def fine_tune_model(model, optimizer, loader, criterion, epochs):
    for epoch in range(epochs):
        fine_tune_loss = train_one_epoch(model, loader, criterion, optimizer)
    print(f"[Fine-tune] Trained for {epochs} epochs | Last loss {fine_tune_loss:.6f}")

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
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.model(x)


num_epochs = 200
patience = 15
fine_tune_epochs = 50
horizon_days = 100  # rolling prediction horizon in trading days

total_rows = len(df_merged)
if total_rows <= horizon_days:
    raise ValueError("Not enough rows to support the requested rolling horizon.")

initial_train_size = max(horizon_days, int(total_rows * 0.6))
if initial_train_size + horizon_days > total_rows:
    initial_train_size = total_rows - horizon_days
if initial_train_size <= 0:
    raise ValueError("Insufficient data to leave a full forecasting buffer.")

print(
    f"Config -> initial training rows: {initial_train_size}, horizon: {horizon_days} days, "
    f"fine-tune epochs per block: {fine_tune_epochs}"
)

initial_history = df_merged.iloc[:initial_train_size]
val_size = max(30, int(0.1 * len(initial_history)))
if val_size >= len(initial_history):
    val_size = max(1, len(initial_history) // 5)

train_sub = initial_history.iloc[:-val_size]
val_sub = initial_history.iloc[-val_size:]

scaler_X = StandardScaler()
scaler_y = StandardScaler()

train_dataset = make_tensor_dataset(train_sub, true_features, scaler_X, scaler_y, fit_scalers=True)
val_dataset = make_tensor_dataset(val_sub, true_features, scaler_X, scaler_y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = MLPRegressor(input_dim=input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

best_state = train_with_early_stopping(
    model, train_loader, val_loader, criterion, optimizer, num_epochs, patience
)
if best_state is not None:
    model.load_state_dict(best_state)

rolling_predictions = []
rolling_truth = []
rolling_dates = []

cursor = initial_train_size
window_idx = 1

while cursor < total_rows:
    future_end = min(cursor + horizon_days, total_rows)
    future_slice = df_merged.iloc[cursor:future_end]
    if future_slice.empty:
        break

    X_future = future_slice[pred_features].values
    X_future_scaled = scaler_X.transform(X_future)
    X_future_t = torch.tensor(X_future_scaled, dtype=torch.float32)

    model.eval()
    with torch.no_grad():
        future_pred_scaled = model(X_future_t).numpy()
    future_pred = scaler_y.inverse_transform(future_pred_scaled).ravel()
    y_future = future_slice["sp500_close"].values

    window_rmse = np.sqrt(mean_squared_error(y_future, future_pred))
    window_r2 = r2_score(y_future, future_pred)
    start_date = future_slice["date"].iloc[0].date()
    end_date = future_slice["date"].iloc[-1].date()
    print(
        f"[Window {window_idx}] {start_date} → {end_date} | "
        f"RMSE {window_rmse:.4f} | R^2 {window_r2:.4f}"
    )

    rolling_predictions.append(future_pred)
    rolling_truth.append(y_future)
    rolling_dates.append(future_slice["date"].values)

    revealed_dataset = make_tensor_dataset(future_slice, true_features, scaler_X, scaler_y)
    revealed_loader = DataLoader(revealed_dataset, batch_size=32, shuffle=True)
    fine_tune_model(model, optimizer, revealed_loader, criterion, fine_tune_epochs)

    cursor = future_end
    window_idx += 1

if not rolling_predictions:
    raise RuntimeError("No rolling predictions were produced; check horizon and data length.")

all_pred = np.concatenate(rolling_predictions)
all_true = np.concatenate(rolling_truth)
all_dates = pd.to_datetime(np.concatenate(rolling_dates))

overall_rmse = np.sqrt(mean_squared_error(all_true, all_pred))
overall_r2 = r2_score(all_true, all_pred)
print(f"Rolling horizon {horizon_days} days | Overall RMSE: {overall_rmse:.4f}, R^2: {overall_r2:.4f}")

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
plt.title(f"Rolling S&P500 Forecast (horizon={horizon_days} days)")
plt.legend()
plt.tight_layout()
plt.show()
