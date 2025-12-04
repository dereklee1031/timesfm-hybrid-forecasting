import argparse
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from matplotlib.ticker import MaxNLocator
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


BASES = ["dgs10", "baa10ym", "cpiaucsl", "dcoilwtico", "fedfunds", "indpro", "t10y2y", "unrate"]
MONTHLY_BASES = {"baa10ym", "cpiaucsl", "fedfunds", "indpro", "unrate"}
PUBLICATION_LAGS_DAYS = {
    "cpiaucsl_true": 14,
    "indpro_true": 14,
    "unrate_true": 7,
    "fedfunds_true": 7,
    "baa10ym_true": 7,
    "dgs10_true": 0,
    "t10y2y_true": 0,
    "dcoilwtico_true": 0,
}
WEEKLY_FREQ = "W-FRI"
RESULTS_DIR = Path("results")
SCENARIO_SAMPLES = 5
NEAR_ZERO_STD = 1e-12


def resample_to_weekly_and_ffill(df: pd.DataFrame) -> pd.DataFrame:
    return df.set_index("date").resample(WEEKLY_FREQ).ffill().reset_index()


def apply_publication_lags(df: pd.DataFrame) -> pd.DataFrame:
    shifted = df.set_index("date")
    for col, lag_days in PUBLICATION_LAGS_DAYS.items():
        if col not in shifted.columns:
            continue
        weeks = max(int(np.round(lag_days / 7)), 0)
        if weeks > 0:
            shifted[col] = shifted[col].shift(weeks)
    return shifted.reset_index()


def compute_5day_log_return(sp500: pd.DataFrame) -> pd.DataFrame:
    daily = sp500.set_index("date").sort_index()
    daily["log_price"] = np.log(daily["sp500_close"])
    daily["y_ret"] = daily["log_price"] - daily["log_price"].shift(5)
    weekly = daily[["sp500_close", "y_ret"]].resample(WEEKLY_FREQ).last()
    weekly = weekly.dropna().reset_index()
    return weekly


def load_base_series(base: str) -> pd.DataFrame:
    path = Path("data/predictions") / f"{base}_backtest.csv"
    df = pd.read_csv(path, parse_dates=["date"]).sort_values("date")
    df = resample_to_weekly_and_ffill(df)
    df = apply_publication_lags(df)
    return df


def merge_all_series() -> pd.DataFrame:
    frames = [load_base_series(base) for base in BASES]
    sp500 = pd.read_csv("data/raw/sp500.csv", parse_dates=["date"]).sort_values("date")
    sp500_weekly = compute_5day_log_return(sp500)
    dfs = frames + [sp500_weekly]
    merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs)
    merged = merged.sort_values("date")
    merged = merged.ffill().dropna().reset_index(drop=True)
    merged["oil_log_true"] = np.log1p(merged["dcoilwtico_true"])
    merged["oil_log_pred"] = np.log1p(merged["dcoilwtico_pred"])
    merged = merged.dropna().reset_index(drop=True)
    return merged


def align_true_pred_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    true_cols = sorted([c for c in df.columns if c.endswith("_true")])
    pred_cols = sorted([c for c in df.columns if c.endswith("_pred")])
    true_bases = [c.replace("_true", "") for c in true_cols]
    pred_bases = [c.replace("_pred", "") for c in pred_cols]
    if true_bases != pred_bases:
        raise ValueError(f"Mismatch true vs pred columns: {true_bases} vs {pred_bases}")
    return true_cols, pred_cols


def drop_near_constant_columns(train_true: pd.DataFrame) -> List[str]:
    mask = train_true.std(axis=0) >= NEAR_ZERO_STD
    return [col for col, keep in zip(train_true.columns, mask) if keep]


def build_sequence_arrays(
    features: np.ndarray,
    targets: np.ndarray,
    seq_len: int,
    start_idx: int,
    end_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    sequences, t = [], []
    start = max(seq_len, start_idx)
    for idx in range(start, end_idx):
        seq = features[idx - seq_len : idx]
        sequences.append(seq)
        t.append(targets[idx])
    if not sequences:
        return np.empty((0, seq_len, features.shape[1])), np.empty((0,))
    return np.array(sequences, dtype=np.float32), np.array(t, dtype=np.float32)


def build_lagged_features(
    features: np.ndarray,
    targets: np.ndarray,
    start_idx: int,
    end_idx: int,
    max_lag: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    rows, labels = [], []
    for idx in range(max(start_idx, max_lag), end_idx):
        feat_lags = [features[idx - lag] for lag in range(1, max_lag + 1)]
        ret_lags = [targets[idx - lag] for lag in range(1, max_lag + 1)]
        rows.append(np.concatenate(feat_lags + [np.array(ret_lags)]))
        labels.append(targets[idx])
    if not rows:
        return np.empty((0, features.shape[1] * max_lag + max_lag)), np.empty((0,))
    return np.array(rows, dtype=np.float32), np.array(labels, dtype=np.float32)


def baseline_predict_sequence(
    model,
    feature_history: np.ndarray,
    target_history: np.ndarray,
    future_features: np.ndarray,
    max_lag: int = 4,
) -> np.ndarray:
    if len(feature_history) < max_lag or len(target_history) < max_lag:
        return np.full(len(future_features), np.nan, dtype=np.float32)
    preds: List[float] = []
    hist_returns = list(target_history[-max_lag:])
    feat_hist = list(feature_history[-max_lag:])
    for feat in future_features:
        row = np.concatenate(feat_hist[::-1][:max_lag] + [np.array(hist_returns[::-1][:max_lag])])
        pred = model.predict(row.reshape(1, -1))[0]
        preds.append(pred)
        hist_returns.append(pred)
        if len(hist_returns) > max_lag:
            hist_returns = hist_returns[-max_lag:]
        feat_hist.append(feat)
        if len(feat_hist) > max_lag:
            feat_hist = feat_hist[-max_lag:]
    return np.array(preds)


class CNNStem(nn.Module):
    def __init__(self, in_channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=5, padding=0)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=0)
        self.residual = nn.Conv1d(in_channels, 64, kernel_size=1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq, features)
        x_t = x.transpose(1, 2)
        res = self.residual(x_t)
        # causal padding so we never use future timesteps
        pad1 = (int(self.conv1.kernel_size[0]) - 1, 0)
        h = F.pad(x_t, pad1)
        h = self.act(self.conv1(h))
        h = self.dropout(h)
        pad2 = (int(self.conv2.kernel_size[0]) - 1, 0)
        h = F.pad(h, pad2)
        h = self.act(self.conv2(h))
        h = h + res
        h = h.transpose(1, 2)
        return self.norm(h)


class TransformerHead(nn.Module):
    def __init__(self, d_model: int, num_layers: int, dropout: float, nhead: int = 4):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class MLPHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], dropout: float):
        super().__init__()
        layers: List[nn.Module] = []
        in_dim = input_dim
        for hidden in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x[:, -1, :]
        return self.net(pooled)


class ForecastModel(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        dropout: float,
        hidden_dims: List[int],
        transformer_layers: int,
    ):
        super().__init__()
        self.stem = CNNStem(feature_dim, dropout)
        self.transformer_head = TransformerHead(d_model=64, num_layers=transformer_layers, dropout=dropout)
        self.head = MLPHead(64, hidden_dims, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        h = self.transformer_head(h)
        return self.head(h)


def add_scenario_noise(sequences: np.ndarray, residual_std: np.ndarray) -> np.ndarray:
    noise = np.random.normal(scale=residual_std.reshape(1, 1, -1), size=sequences.shape)
    return sequences + noise.astype(np.float32)


def timesfm_scenarios(future_seq: np.ndarray, residual_std: np.ndarray, k: int) -> np.ndarray:
    scenarios = []
    for _ in range(k):
        scenarios.append(add_scenario_noise(future_seq, residual_std))
    return np.stack(scenarios)


def train_cnn_model(
    model: nn.Module,
    train_seq: np.ndarray,
    train_targets: np.ndarray,
    val_seq: np.ndarray,
    val_targets: np.ndarray,
    residual_std: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    lr: float,
    use_huber: bool = False,
    weight_decay: float = 1e-4,
):
    criterion = nn.SmoothL1Loss(beta=0.005) if use_huber else nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state = None
    best_val = np.inf
    patience_ctr = 0

    val_loader = None
    if len(val_seq) > 0:
        val_ds = TensorDataset(
            torch.tensor(val_seq, dtype=torch.float32), torch.tensor(val_targets, dtype=torch.float32).unsqueeze(-1)
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for epoch in range(1, epochs + 1):
        scenario_seq = add_scenario_noise(train_seq, residual_std)
        train_ds = TensorDataset(
            torch.tensor(scenario_seq, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).unsqueeze(-1),
        )
        loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        model.train()
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        train_loss = total_loss / max(total, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            v_loss = 0.0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    v_loss += loss.item() * xb.size(0)
                    v_total += xb.size(0)
            val_loss = v_loss / max(v_total, 1)

        print(f"[CNN Train] Epoch {epoch:03d} | Train {train_loss:.6f} | Val {val_loss:.6f}")
        if val_loss is not None and val_loss < best_val - 1e-5:
            patience_ctr = 0
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping for CNN.")
                break

    if best_state:
        model.load_state_dict(best_state)


def fine_tune_cnn(
    model: nn.Module,
    seq: np.ndarray,
    targets: np.ndarray,
    lr: float,
    epochs: int,
    residual_std: np.ndarray,
    weight_decay: float = 1e-4,
):
    if len(seq) == 0:
        return
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    fine_ds = TensorDataset(
        torch.tensor(add_scenario_noise(seq, residual_std), dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32).unsqueeze(-1),
    )
    loader = DataLoader(fine_ds, batch_size=32, shuffle=True)
    for epoch in range(epochs):
        total_loss = 0.0
        total = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * xb.size(0)
            total += xb.size(0)
        print(f"[Fine-tune] Epoch {epoch+1:02d} | Loss {total_loss / max(total, 1):.6f}")


@dataclass
class RollingWindowResult:
    window: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    rmse: float
    r2: float
    direction_acc: float


def plot_predictions(results_df: pd.DataFrame, horizon: int):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results_df["date"], results_df["y_true"], label="True")
    ax.plot(results_df["date"], results_df["y_pred_mean"], label="Predicted")
    ax.set_title("Weekly 5-day Log Returns")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(10))
    fig.tight_layout()
    path = RESULTS_DIR / f"transformer_weekly_returns_h{horizon}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved prediction plot -> {path}")


def plot_residuals(results_df: pd.DataFrame, horizon: int):
    residuals = results_df["y_true"] - results_df["y_pred_mean"]
    rolling_rmse = residuals.rolling(20).apply(lambda x: np.sqrt(np.mean(x**2)), raw=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(results_df["date"], residuals, label="Residuals")
    ax.plot(results_df["date"], rolling_rmse, label="20-week RMSE")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Residuals & 20-week Rolling RMSE")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(10))
    fig.tight_layout()
    path = RESULTS_DIR / f"transformer_residuals_h{horizon}.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"Saved residual plot -> {path}")


def plot_reconstructed_levels(results_df: pd.DataFrame, price_df: pd.DataFrame, horizon: int):
    merged = results_df.merge(price_df[["date", "sp500_close"]], on="date", how="left").sort_values("date")
    if merged["sp500_close"].isna().any():
        raise ValueError("Missing sp500_close for some prediction dates; ensure price_df covers the forecast span.")
    base_price = merged["sp500_close"].iloc[0]
    merged["level_true"] = base_price * np.exp(merged["y_true"].cumsum())
    merged["level_pred"] = base_price * np.exp(merged["y_pred_mean"].cumsum())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    levels_path = RESULTS_DIR / f"transformer_levels_h{horizon}.csv"
    merged[["date", "level_true", "level_pred", "sp500_close"]].to_csv(levels_path, index=False)
    plt.figure(figsize=(12, 5))
    plt.plot(merged["date"], merged["sp500_close"], label="Actual SP500 (weekly)")
    plt.plot(merged["date"], merged["level_true"], label="Reconstructed True", linestyle="--")
    plt.plot(merged["date"], merged["level_pred"], label="Reconstructed Pred", linestyle="--")
    plt.title("SP500 Levels vs Reconstructed Paths")
    plt.xlabel("Date")
    plt.ylabel("Level")
    plt.legend()
    plt.tight_layout()
    plot_path = RESULTS_DIR / f"transformer_levels_h{horizon}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Saved reconstructed level plot -> {plot_path} (data -> {levels_path})")


def parse_args():
    parser = argparse.ArgumentParser(description="Weekly leakage-free training with baselines + CNN transformer.")
    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--horizon-weeks", type=int, default=2)
    parser.add_argument("--step-weeks", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scenario-k", type=int, default=SCENARIO_SAMPLES)
    parser.add_argument("--cnn-hidden", nargs="+", type=int, default=[256, 128], help="Hidden layer sizes for the MLP head.")
    parser.add_argument("--tfm-layers", type=int, default=2, help="Number of transformer layers after the CNN stem.")
    parser.add_argument("--run-name", type=str, default="default")
    return parser.parse_args()


def compute_directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    signs = np.sign(y_true)
    pred_signs = np.sign(y_pred)
    return float(np.mean(signs == pred_signs))


def walkforward_pipeline(args):
    df = merge_all_series()
    true_cols, pred_cols = align_true_pred_columns(df)
    df = df.sort_values("date").reset_index(drop=True)

    # ensure target present
    if "y_ret" not in df.columns:
        raise ValueError("Target y_ret missing after merge.")
    y = df["y_ret"].values.astype(np.float32)

    split_idx = int(len(df) * 0.6)
    val_size = max(int(len(df) * 0.06), 8)
    train_end = split_idx - val_size
    if train_end <= args.context_len:
        raise ValueError("Not enough data for requested context length.")

    train_true = df.iloc[:train_end][true_cols]
    keep_cols = drop_near_constant_columns(train_true)
    df = df[keep_cols + [c for c in pred_cols if c.replace("_pred", "_true") in keep_cols] + ["date", "sp500_close", "y_ret"]]
    true_cols, pred_cols = align_true_pred_columns(df)

    scaler = StandardScaler()
    scaler.fit(df.iloc[:train_end][true_cols].to_numpy())
    true_scaled = scaler.transform(df[true_cols].to_numpy())
    pred_scaled = scaler.transform(df[pred_cols].to_numpy())

    residual_std = np.maximum(
        (pred_scaled[:train_end] - true_scaled[:train_end]).std(axis=0),
        1e-6,
    )

    train_seq, train_targets = build_sequence_arrays(true_scaled, y, args.context_len, args.context_len, train_end)
    val_seq, val_targets = build_sequence_arrays(true_scaled, y, args.context_len, train_end, split_idx)

    model = ForecastModel(
        feature_dim=true_scaled.shape[1],
        dropout=args.dropout,
        hidden_dims=args.cnn_hidden,
        transformer_layers=args.tfm_layers,
    )
    initial_norm = sum(p.detach().norm().item() for p in model.parameters())
    train_cnn_model(
        model,
        train_seq,
        train_targets,
        val_seq,
        val_targets,
        residual_std,
        args.epochs,
        args.batch_size,
        args.patience,
        args.lr,
    )
    final_norm = sum(p.detach().norm().item() for p in model.parameters())
    print(f"[Weights] Δ||θ|| = {final_norm - initial_norm:.4f}")

    base_feature_mean = df.iloc[:train_end][true_cols].mean()
    base_pred_mean = df.iloc[train_end:][pred_cols].mean()
    drift = (base_pred_mean - base_feature_mean).abs()
    print(f"[Drift] mean absolute drift (train true vs future pred): {drift.mean():.4e}")

    lin_model, rf_model = None, None
    lagged_X, lagged_y = build_lagged_features(true_scaled, y, args.context_len, train_end)
    if len(lagged_X):
        lin_model = LinearRegression().fit(lagged_X, lagged_y)
        rf_model = RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        rf_model.fit(lagged_X, lagged_y)
    else:
        print("[Baselines] Not enough samples for lagged regression.")

    rolling_preds = []
    rolling_dates = []
    rolling_metrics: List[RollingWindowResult] = []
    cursor = split_idx
    window_idx = 1
    fine_tune_lr = args.lr * 0.3
    seq_len = args.context_len
    predicted_indices = set()

    while cursor < len(df):
        future_end = min(cursor + args.horizon_weeks, len(df))
        if future_end - cursor <= 0:
            break

        scenario_seq = []
        scenario_indices = []
        scenario_targets = []
        for idx in range(cursor, future_end):
            if idx < seq_len or idx in predicted_indices:
                continue
            seq = pred_scaled[idx - seq_len : idx]
            scenario_seq.append(seq)
            scenario_indices.append(idx)
            scenario_targets.append(y[idx])
        if not scenario_seq:
            break
        future_seq = np.array(scenario_seq, dtype=np.float32)
        future_features_flat = np.array([seq[-1] for seq in future_seq], dtype=np.float32)
        future_dates = df["date"].iloc[scenario_indices].values
        y_future = np.array(scenario_targets, dtype=np.float32)

        scenario_preds = timesfm_scenarios(future_seq, residual_std, args.scenario_k)
        preds_samples = []
        model.eval()
        with torch.no_grad():
            for scenario in scenario_preds:
                tensor = torch.tensor(scenario, dtype=torch.float32)
                preds = model(tensor).squeeze(-1).numpy()
                preds_samples.append(preds)
        preds_samples = np.stack(preds_samples)
        pred_mean = preds_samples.mean(axis=0)
        pred_p10 = np.percentile(preds_samples, 10, axis=0)
        pred_p50 = np.percentile(preds_samples, 50, axis=0)
        pred_p90 = np.percentile(preds_samples, 90, axis=0)

        lin_pred = None
        rf_pred = None
        if lin_model is not None:
            lin_pred = baseline_predict_sequence(
                lin_model,
                true_scaled[:cursor],
                y[:cursor],
                future_features_flat,
            )
        if rf_model is not None:
            baseline_rows = []
            hist_returns = list(y[:cursor])
            hist_features = list(true_scaled[:cursor])
            for feat in future_features_flat:
                feat_lags = hist_features[-4:]
                ret_lags = hist_returns[-4:]
                if len(feat_lags) < 4 or len(ret_lags) < 4:
                    continue
                baseline_rows.append(np.concatenate(feat_lags[::-1] + [np.array(ret_lags[::-1])]))
                hist_features.append(feat)
                hist_returns.append(pred_mean[len(baseline_rows) - 1])
            if baseline_rows:
                rf_pred = rf_model.predict(np.array(baseline_rows))

        rmse = np.sqrt(mean_squared_error(y_future, pred_mean))
        r2 = r2_score(y_future, pred_mean)
        dir_acc = compute_directional_accuracy(y_future, pred_mean)
        start_dt = pd.to_datetime(future_dates[0])
        end_dt = pd.to_datetime(future_dates[-1])
        rolling_metrics.append(
            RollingWindowResult(
                window=window_idx,
                start_date=start_dt,
                end_date=end_dt,
                rmse=rmse,
                r2=r2,
                direction_acc=dir_acc,
            )
        )
        print(
            f"[Window {window_idx}] {start_dt.date()} → {end_dt.date()} | "
            f"RMSE {rmse:.4f} | R^2 {r2:.4f} | DirAcc {dir_acc:.3f}"
        )

        rolling_preds.append(
            pd.DataFrame(
                {
                    "date": future_dates,
                    "y_true": y_future,
                    "y_pred_mean": pred_mean,
                    "y_pred_p10": pred_p10,
                    "y_pred_p50": pred_p50,
                    "y_pred_p90": pred_p90,
                    "y_pred_lin": lin_pred if lin_pred is not None else np.nan,
                    "y_pred_rf": rf_pred if rf_pred is not None else np.nan,
                }
            )
        )
        rolling_dates.append(future_dates)
        predicted_indices.update(scenario_indices)

        fine_seq, fine_targets = build_sequence_arrays(true_scaled, y, seq_len, cursor, future_end)
        fine_tune_cnn(model, fine_seq, fine_targets, fine_tune_lr, args.fine_tune_epochs, residual_std)

        cursor += args.step_weeks
        window_idx += 1

    if not rolling_preds:
        raise RuntimeError("No rolling predictions produced.")

    results_df = pd.concat(rolling_preds, ignore_index=True).sort_values("date")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    preds_path = RESULTS_DIR / "rolling_predictions.csv"
    results_df.to_csv(preds_path, index=False)

    metrics_df = pd.DataFrame(
        {
            "window": [m.window for m in rolling_metrics],
            "start_date": [m.start_date for m in rolling_metrics],
            "end_date": [m.end_date for m in rolling_metrics],
            "rmse": [m.rmse for m in rolling_metrics],
            "r2": [m.r2 for m in rolling_metrics],
            "directional_accuracy": [m.direction_acc for m in rolling_metrics],
        }
    )
    metrics_path = RESULTS_DIR / "rolling_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved rolling predictions -> {preds_path}")
    print(f"Saved rolling metrics -> {metrics_path}")

    plot_predictions(results_df, args.horizon_weeks)
    plot_residuals(results_df, args.horizon_weeks)
    plot_reconstructed_levels(results_df, df[["date", "sp500_close"]].copy(), args.horizon_weeks)

    overall_rmse = float(np.sqrt(mean_squared_error(results_df["y_true"], results_df["y_pred_mean"])))
    overall_r2 = float(r2_score(results_df["y_true"], results_df["y_pred_mean"]))
    overall_dir = compute_directional_accuracy(results_df["y_true"].values, results_df["y_pred_mean"].values)
    log_row = pd.DataFrame(
        [
            {
                "run_name": args.run_name,
                "context_len": args.context_len,
                "horizon_weeks": args.horizon_weeks,
                "step_weeks": args.step_weeks,
                "epochs": args.epochs,
                "dropout": args.dropout,
                "lr": args.lr,
                "scenario_k": args.scenario_k,
                "cnn_hidden": "-".join(map(str, args.cnn_hidden)),
                "tfm_layers": args.tfm_layers,
                "rmse": overall_rmse,
                "r2": overall_r2,
                "directional_accuracy": overall_dir,
            }
        ]
    )
    log_path = RESULTS_DIR / "hyperparam_runs.csv"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_row.to_csv(log_path, mode="a", header=not log_path.exists(), index=False)
    print(f"[Hyperparam Log] {args.run_name} -> RMSE={overall_rmse:.4f} R^2={overall_r2:.4f} DirAcc={overall_dir:.3f}")

    if results_df["y_pred_mean"].std() < 1e-6:
        print("[Warning] Predictions collapsed; consider --dropout 0.3 and --lr 3e-4 with Huber loss.")


def main():
    args = parse_args()
    walkforward_pipeline(args)


if __name__ == "__main__":
    main()
