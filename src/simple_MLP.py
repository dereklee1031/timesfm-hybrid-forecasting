# -------------------------------------------------------------
# TimesFM Hybrid Forecasting — MLP Version
# Matches partner's Transformer pipeline structure,
# but uses a simpler flattened-sequence MLP.
# -------------------------------------------------------------

import argparse
from dataclasses import dataclass
from functools import reduce
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
BASES = [
    "dgs10", "baa10ym", "cpiaucsl", "dcoilwtico",
    "fedfunds", "indpro", "t10y2y", "unrate"
]

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
RESULTS_DIR = Path("results_mlp_simple")
NEAR_ZERO_STD = 1e-12
SCENARIO_SAMPLES = 5

# -------------------------------------------------------------
# DATA PROCESSING
# -------------------------------------------------------------

def resample_to_weekly_and_ffill(df):
    return df.set_index("date").resample(WEEKLY_FREQ).ffill().reset_index()

def apply_publication_lags(df):
    shifted = df.set_index("date")
    for col, lag_days in PUBLICATION_LAGS_DAYS.items():
        if col not in shifted.columns:
            continue
        weeks = int(np.round(lag_days / 7))
        shifted[col] = shifted[col].shift(max(weeks, 0))
    return shifted.reset_index()

def compute_5day_log_return(sp500):
    daily = sp500.set_index("date").sort_index()
    daily["log_price"] = np.log(daily["sp500_close"])
    daily["y_ret"] = daily["log_price"] - daily["log_price"].shift(5)
    weekly = daily[["sp500_close", "y_ret"]].resample(WEEKLY_FREQ).last()
    return weekly.dropna().reset_index()

def load_base_series(base):
    df = pd.read_csv(
        Path("data/predictions") / f"{base}_backtest.csv",
        parse_dates=["date"]
    ).sort_values("date")
    return apply_publication_lags(resample_to_weekly_and_ffill(df))

def merge_all_series():
    frames = [load_base_series(b) for b in BASES]
    sp = pd.read_csv("data/raw/sp500.csv", parse_dates=["date"]).sort_values("date")
    sp_weekly = compute_5day_log_return(sp)

    dfs = frames + [sp_weekly]
    merged = reduce(lambda left, right:
                    pd.merge(left, right, on="date", how="outer"), dfs)

    merged = merged.sort_values("date").ffill().dropna().reset_index(drop=True)

    # Add log-transformed oil features
    merged["oil_log_true"] = np.log1p(merged["dcoilwtico_true"])
    merged["oil_log_pred"] = np.log1p(merged["dcoilwtico_pred"])

    return merged.dropna().reset_index(drop=True)

def align_true_pred_columns(df):
    true_cols = sorted([c for c in df.columns if c.endswith("_true")])
    pred_cols = sorted([c for c in df.columns if c.endswith("_pred")])

    true_bases = [c.replace("_true", "") for c in true_cols]
    pred_bases = [c.replace("_pred", "") for c in pred_cols]
    if true_bases != pred_bases:
        raise ValueError(f"Mismatch true vs pred columns: {true_cols} vs {pred_cols}")
    return true_cols, pred_cols

def drop_near_constant_columns(train_true_df):
    mask = train_true_df.std(axis=0) >= NEAR_ZERO_STD
    return [col for col, keep in zip(train_true_df.columns, mask) if keep]

# -------------------------------------------------------------
# SEQUENCE & NOISE BUILDING
# -------------------------------------------------------------

def build_sequence_arrays(features, targets, seq_len, start_idx, end_idx):
    seqs, labs = [], []
    begin = max(seq_len, start_idx)
    for idx in range(begin, end_idx):
        seq = features[idx - seq_len: idx]   # shape (seq_len, feature_dim)
        seqs.append(seq)
        labs.append(targets[idx])
    if not seqs:
        return np.empty((0, seq_len, features.shape[1])), np.empty((0,))
    return np.array(seqs, dtype=np.float32), np.array(labs, dtype=np.float32)

def add_scenario_noise(sequences, residual_std):
    """
    sequences:  (batch, seq_len, features)
    residual_std: (features,)
    returns noisy sequences of same shape
    """
    noise = np.random.normal(
        scale=residual_std.reshape(1, 1, -1),
        size=sequences.shape
    )
    return sequences + noise.astype(np.float32)

# -------------------------------------------------------------
# SEQUENCE MLP (flattened)
# -------------------------------------------------------------

class SequenceMLP(nn.Module):
    """
    Accepts input shaped (batch, seq_len, feature_dim),
    flattens to (batch, seq_len * feature_dim),
    then passes through MLP layers.
    """
    def __init__(self, seq_len, feature_dim, hidden_sizes, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.feature_dim = feature_dim

        layers = []
        input_dim = seq_len * feature_dim

        prev_dim = input_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h

        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x: (batch, seq_len, feature_dim)
        x = x.view(x.size(0), -1)  # flatten
        return self.net(x)

# -------------------------------------------------------------
# TRAINING AND FINE-TUNING
# -------------------------------------------------------------

def train_mlp(model, train_seq, train_targets,
              val_seq, val_targets,
              residual_std, epochs, batch_size,
              patience, lr):

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # Validation loader
    val_loader = None
    if len(val_seq) > 0:
        val_ds = TensorDataset(
            torch.tensor(val_seq, dtype=torch.float32),
            torch.tensor(val_targets, dtype=torch.float32).unsqueeze(-1)
        )
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val = float('inf')
    patience_ctr = 0
    best_state = None

    for epoch in range(1, epochs + 1):

        # Add scenario noise to training data
        noisy = add_scenario_noise(train_seq, residual_std)

        train_ds = TensorDataset(
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(train_targets, dtype=torch.float32).unsqueeze(-1)
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        # Train loop
        model.train()
        running = 0.0
        total = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            total += xb.size(0)
        train_loss = running / max(total, 1)

        # Validation
        val_loss = None
        if val_loader is not None:
            model.eval()
            v_running = 0.0
            v_total = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    preds = model(xb)
                    loss = criterion(preds, yb)
                    v_running += loss.item() * xb.size(0)
                    v_total += xb.size(0)
            val_loss = v_running / max(v_total, 1)

        print(f"[MLP] Epoch {epoch} | Train={train_loss:.6f} | Val={val_loss:.6f}")

        # Early stopping
        if val_loss is not None and val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Early stopping MLP.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val

def fine_tune_mlp(model, seq, targets, lr, epochs, residual_std):
    if len(seq) == 0:
        return
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    noisy = add_scenario_noise(seq, residual_std)
    loader = DataLoader(
        TensorDataset(
            torch.tensor(noisy, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)
        ),
        batch_size=32, shuffle=True
    )

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

# -------------------------------------------------------------
# HYPERPARAMETER SWEEP UTILITIES
# -------------------------------------------------------------

def parse_hidden_grid(v):
    """
    Example:
    "512-256-128|384-192-96|256-128"
    → [
         [512,256,128],
         [384,192,96],
         [256,128]
       ]
    """
    return [[int(x) for x in block.split("-")]
            for block in v.split("|") if block.strip()]

def parse_float_grid(v):
    """
    Example: "0.1,0.2,0.3" → [0.1, 0.2, 0.3]
    """
    return [float(x) for x in v.split(",") if x.strip()]

def hyperparameter_sweep(
    seq_len, feature_dim,
    train_seq, train_targets,
    val_seq, val_targets,
    residual_std,
    args
):
    best_cfg = None
    best_val = float('inf')

    for hidden_sizes in args.hidden_grid:
        for dropout in args.dropout_grid:
            for lr in args.lr_grid:

                print(f"[SWEEP] hidden={hidden_sizes} dropout={dropout} lr={lr}")

                model = SequenceMLP(seq_len, feature_dim, hidden_sizes, dropout)

                val_loss = train_mlp(
                    model,
                    train_seq,
                    train_targets,
                    val_seq,
                    val_targets,
                    residual_std,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    patience=args.patience,
                    lr=lr
                )

                print(f"[SWEEP] val_loss={val_loss:.6f}")

                if val_loss < best_val:
                    best_val = val_loss
                    best_cfg = {
                        "hidden_sizes": hidden_sizes,
                        "dropout": dropout,
                        "lr": lr
                    }

    print(f"[SWEEP] Best config: {best_cfg} | Best val={best_val:.6f}")
    return best_cfg

# -------------------------------------------------------------
# METRICS & PLOTS
# -------------------------------------------------------------
@dataclass
class RollingWindowResult:
    window: int
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    rmse: float
    r2: float
    direction_acc: float

def compute_directional_accuracy(y_true, y_pred):
    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))

def plot_predictions(results_df, horizon):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12,4))
    plt.plot(results_df["date"], results_df["y_true"], label="True")
    plt.plot(results_df["date"], results_df["y_pred_mean"], label="MLP Pred")
    plt.title(f"Weekly 5-day Log Returns (MLP h={horizon})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"mlp_simple_predictions_h{horizon}.png", dpi=300)
    plt.close()

def plot_residuals(results_df, horizon):
    residuals = results_df["y_true"] - results_df["y_pred_mean"]
    rolling_rmse = residuals.rolling(20).apply(
        lambda x: np.sqrt(np.mean(x**2)), raw=True)

    plt.figure(figsize=(12,4))
    plt.plot(results_df["date"], residuals, label="Residuals")
    plt.plot(results_df["date"], rolling_rmse, label="20-week RMSE")
    plt.axhline(0, color="black", linewidth=0.8)
    plt.title("Residuals & Rolling RMSE (MLP)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"mlp_simple_residuals_h{horizon}.png", dpi=300)
    plt.close()

def plot_reconstructed_levels(results_df, price_df, horizon):
    merged = results_df.merge(
        price_df[["date", "sp500_close"]],
        on="date",
        how="left"
    ).sort_values("date")

    base_price = merged["sp500_close"].iloc[0]
    merged["level_true"] = base_price * np.exp(merged["y_true"].cumsum())
    merged["level_pred"] = base_price * np.exp(merged["y_pred_mean"].cumsum())

    merged.to_csv(
        RESULTS_DIR / f"mlp_simple_levels_h{horizon}.csv",
        index=False
    )

    plt.figure(figsize=(12,5))
    plt.plot(merged["date"], merged["sp500_close"], label="Actual SP500")
    plt.plot(merged["date"], merged["level_true"], label="Reconstructed True", linestyle="--")
    plt.plot(merged["date"], merged["level_pred"], label="Reconstructed Pred", linestyle="--")
    plt.xlabel("Date")
    plt.ylabel("SP500 Level")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"mlp_simple_levels_h{horizon}.png", dpi=300)
    plt.close()

# -------------------------------------------------------------
# WALK-FORWARD PIPELINE
# -------------------------------------------------------------

def walkforward_pipeline(args):

    df = merge_all_series()
    true_cols, pred_cols = align_true_pred_columns(df)
    df = df.sort_values("date").reset_index(drop=True)

    if "y_ret" not in df.columns:
        raise ValueError("y_ret missing from merged dataframe.")

    y = df["y_ret"].values.astype(np.float32)

    # ---------------------------
    # SPLIT INTO TRAIN / VAL / TEST (walk-forward region)
    # ---------------------------
    split_idx = int(len(df) * 0.6)
    val_size = max(int(len(df) * 0.06), 8)
    train_end = split_idx - val_size

    # Remove near-constant columns
    train_true = df.iloc[:train_end][true_cols]
    keep_cols = drop_near_constant_columns(train_true)

    df = df[
        keep_cols +
        [c for c in pred_cols if c.replace("_pred", "_true") in keep_cols] +
        ["date", "sp500_close", "y_ret"]
    ]

    true_cols, pred_cols = align_true_pred_columns(df)

    # ---------------------------
    # SCALING
    # ---------------------------
    true_df = df[true_cols].copy()
    pred_df = df[pred_cols].copy()

    # Remove suffix for consistent scaler
    true_df.columns = [c.replace("_true", "") for c in true_cols]
    pred_df.columns = [c.replace("_pred", "") for c in pred_cols]

    scaler = StandardScaler().fit(true_df.iloc[:train_end])

    true_scaled = scaler.transform(true_df)
    pred_scaled = scaler.transform(pred_df)

    # Residuals for scenario noise
    residual_std = np.maximum(
        (pred_scaled[:train_end] - true_scaled[:train_end]).std(axis=0),
        1e-6
    )

    # ---------------------------
    # SEQUENCE BUILDING
    # ---------------------------
    seq_len = args.context_len
    feature_dim = true_scaled.shape[1]

    train_seq, train_targets = build_sequence_arrays(true_scaled, y, seq_len, seq_len, train_end)
    val_seq, val_targets = build_sequence_arrays(true_scaled, y, seq_len, train_end, split_idx)

    # ---------------------------
    # HYPERPARAMETER SWEEP
    # ---------------------------
    best_cfg = hyperparameter_sweep(
        seq_len, feature_dim,
        train_seq, train_targets,
        val_seq, val_targets,
        residual_std,
        args
    )

    # ---------------------------
    # TRAIN FINAL MODEL
    # ---------------------------
    model = SequenceMLP(
        seq_len, feature_dim,
        best_cfg["hidden_sizes"],
        best_cfg["dropout"]
    )

    train_mlp(
        model,
        train_seq, train_targets,
        val_seq, val_targets,
        residual_std,
        epochs=args.epochs,
        batch_size=args.batch_size,
        patience=args.patience,
        lr=best_cfg["lr"]
    )

    # ---------------------------
    # WALK-FORWARD ROLLING EVALUATION
    # ---------------------------
    rolling_preds = []
    rolling_metrics = []

    cursor = split_idx
    window_idx = 1
    fine_lr = best_cfg["lr"] * 0.3

    while cursor < len(df):

        future_end = min(cursor + args.horizon_weeks, len(df))
        if future_end <= cursor:
            break

        future_seq = []
        future_dates = []
        future_targets = []

        for idx in range(cursor, future_end):
            if idx < seq_len: 
                continue

            # determine where to pull the context from
            if idx < split_idx:
        # all true history
                seq = true_scaled[idx - seq_len : idx]
            else:
            # use predicted features only for future windows
                seq = pred_scaled[idx - seq_len : idx]

            future_seq.append(seq)
            future_dates.append(df["date"].iloc[idx])
            future_targets.append(y[idx])

        if not future_seq:
            break

        future_seq = np.array(future_seq, dtype=np.float32)
        future_targets = np.array(future_targets, dtype=np.float32)
        future_dates = np.array(future_dates)

        # ---------------------------
        # MONTE CARLO SCENARIOS
        # ---------------------------
        preds_samples = []
        model.eval()
        with torch.no_grad():
            for _ in range(args.scenario_k):
                scen = add_scenario_noise(future_seq, residual_std)
                scen_tensor = torch.tensor(scen, dtype=torch.float32)
                preds = model(scen_tensor).squeeze(-1).numpy()
                preds_samples.append(preds)

        preds_samples = np.stack(preds_samples)
        pred_mean = preds_samples.mean(axis=0)
        pred_p10 = np.percentile(preds_samples, 10, axis=0)
        pred_p50 = np.percentile(preds_samples, 50, axis=0)
        pred_p90 = np.percentile(preds_samples, 90, axis=0)

        # ---------------------------
        # EVALUATE WINDOW
        # ---------------------------
        rmse = np.sqrt(mean_squared_error(future_targets, pred_mean))
        r2 = r2_score(future_targets, pred_mean)
        dir_acc = compute_directional_accuracy(future_targets, pred_mean)

        print(
            f"[Window {window_idx}] {future_dates[0]} → {future_dates[-1]}  "
            f"RMSE={rmse:.4f}  R2={r2:.4f}  DirAcc={dir_acc:.3f}"
        )

        rolling_metrics.append(
            RollingWindowResult(
                window=window_idx,
                start_date=pd.to_datetime(future_dates[0]),
                end_date=pd.to_datetime(future_dates[-1]),
                rmse=float(rmse),
                r2=float(r2),
                direction_acc=float(dir_acc)
            )
        )

        rolling_preds.append(
            pd.DataFrame({
                "date": future_dates,
                "y_true": future_targets,
                "y_pred_mean": pred_mean,
                "y_pred_p10": pred_p10,
                "y_pred_p50": pred_p50,
                "y_pred_p90": pred_p90
            })
        )

        # ---------------------------
        # FINE TUNE
        # ---------------------------
        fine_seq, fine_targets = build_sequence_arrays(true_scaled, y, seq_len, cursor, future_end)

        fine_tune_mlp(
            model,
            fine_seq, fine_targets,
            fine_lr, args.fine_tune_epochs,
            residual_std
        )

        cursor += args.step_weeks
        window_idx += 1

    # ---------------------------
    # SAVE RESULTS
    # ---------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.concat(rolling_preds, ignore_index=True).sort_values("date")
    results_path = RESULTS_DIR / "mlp_simple_rolling_predictions.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved predictions → {results_path}")

    # Metrics CSV
    metrics_df = pd.DataFrame({
        "window": [m.window for m in rolling_metrics],
        "start_date": [m.start_date for m in rolling_metrics],
        "end_date": [m.end_date for m in rolling_metrics],
        "rmse": [m.rmse for m in rolling_metrics],
        "r2": [m.r2 for m in rolling_metrics],
        "directional_accuracy": [m.direction_acc for m in rolling_metrics],
    })
    metrics_path = RESULTS_DIR / "mlp_simple_rolling_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics → {metrics_path}")

    # ---------------------------
    # PLOTS
    # ---------------------------
    plot_predictions(results_df, args.horizon_weeks)
    plot_residuals(results_df, args.horizon_weeks)
    plot_reconstructed_levels(
        results_df,
        df[["date", "sp500_close"]].copy(),
        args.horizon_weeks
    )

    # ---------------------------
    # OVERALL METRICS
    # ---------------------------
    overall_rmse = np.sqrt(mean_squared_error(results_df["y_true"], results_df["y_pred_mean"]))
    overall_r2 = r2_score(results_df["y_true"], results_df["y_pred_mean"])
    overall_dir = compute_directional_accuracy(results_df["y_true"], results_df["y_pred_mean"])
    print(f"[Overall] RMSE={overall_rmse:.4f}  R^2={overall_r2:.4f}  DirAcc={overall_dir:.3f}")

# -------------------------------------------------------------
# CLI
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="TimesFM MLP Forecaster")

    parser.add_argument("--context-len", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)

    parser.add_argument("--horizon-weeks", type=int, default=2)
    parser.add_argument("--step-weeks", type=int, default=1)
    parser.add_argument("--scenario-k", type=int, default=SCENARIO_SAMPLES)

    # Hyperparameter grids
    parser.add_argument(
        "--hidden-grid",
        type=str,
        default="512-256-128|384-192-96|256-128"
    )
    parser.add_argument(
        "--dropout-grid",
        type=str,
        default="0.05,0.1,0.2"
    )
    parser.add_argument(
        "--lr-grid",
        type=str,
        default="5e-5,1e-4,3e-4"
    )

    args = parser.parse_args()

    args.hidden_grid = parse_hidden_grid(args.hidden_grid)
    args.dropout_grid = parse_float_grid(args.dropout_grid)
    args.lr_grid = parse_float_grid(args.lr_grid)

    return args

# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    args = parse_args()
    walkforward_pipeline(args)

if __name__ == "__main__":
    main()


