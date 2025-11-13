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

#Load data FRED actual and predicted
FRED_df = pd.read_csv("data/predictions/dgs10_backtest.csv", parse_dates=["date"])
FRED_df = FRED_df.sort_values("date").dropna()

#Load SP500 data
sp500_df = pd.read_csv("data/raw/sp500.csv", parse_dates=["date"])
sp500_df = sp500_df.sort_values("date").dropna()

#Merge datasets on date
df_merged = pd.merge(FRED_df, sp500_df[['date', 'sp500_close']], on='date', how='inner')
print(df_merged.head())

#Train-val-test split
n = len(df_merged)
train_end = int(0.95 * n)
val_end = int(0.975 * n)
train_df = df_merged.iloc[:train_end]
val_df = df_merged.iloc[train_end:val_end]
test_df = df_merged.iloc[val_end:]

print(f"Train: {len(train_df)} rows, Val: {len(val_df)} rows, Test: {len(test_df)} rows")

#Assign inputs and targets
X_train = train_df[["dgs10_true"]].values
y_train = train_df["sp500_close"].values

#For validation/test, use predicted values
X_val = val_df[["dgs10_pred"]].values
y_val = val_df["sp500_close"].values

X_test = test_df[["dgs10_pred"]].values
y_test = test_df["sp500_close"].values

#Scale features and target
scaler_X = StandardScaler()
scaler_y = StandardScaler()

#Fit scalers on train data only
X_train_scaled = scaler_X.fit_transform(X_train)
y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1))

#Apply same transformation to val/test sets
X_val_scaled = scaler_X.transform(X_val)
y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1))

X_test_scaled = scaler_X.transform(X_test)
y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1))

#Convert to tensors
X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_t = torch.tensor(y_train_scaled, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_scaled, dtype=torch.float32)
y_val_t   = torch.tensor(y_val_scaled, dtype=torch.float32)
X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_t  = torch.tensor(y_test_scaled, dtype=torch.float32)

#Create TensorDatasets
train_dataset = TensorDataset(X_train_t, y_train_t)
val_dataset   = TensorDataset(X_val_t, y_val_t)
test_dataset  = TensorDataset(X_test_t, y_test_t)

#Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

#Feedforward neural network
class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(1, 64), #1 feature input for now
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.model(x)

model = MLPRegressor()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

#Training loop
num_epochs = 200
patience = 15
best_val_loss = np.inf
patience_counter = 0

train_losses, val_losses = [], []

for epoch in range(num_epochs):
    # --- Training ---
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

    # --- Validation ---
    model.eval()
    with torch.no_grad():
        val_preds = model(X_val_t)
        val_loss = criterion(val_preds, y_val_t).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

    # Early stopping
    if val_loss < best_val_loss - 1e-6:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# Load best model
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

# Predict
with torch.no_grad():
    y_pred_scaled = model(X_test_t).numpy()

# Inverse scale
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test_scaled)

# Evaluate metrics
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)
print(f"Test RMSE: {rmse:.4f}, R^2: {r2:.4f}")

# Plot
plt.figure(figsize=(10,5))
plt.plot(y_true, label='True')
plt.plot(y_pred, label='Predicted')
plt.title("S&P500 Prediction (Test Set)")
plt.legend()
plt.show()

