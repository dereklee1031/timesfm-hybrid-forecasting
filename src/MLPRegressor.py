import argparse
from functools import reduce
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# Named regime-shift windows (inclusive bounds) based on major market stress events.
STRESS_WINDOWS = {
    "dotcom": ("2000-03-10", "2002-10-09"),
    "gfc": ("2007-07-01", "2009-06-30"),
    "flash_crash": ("2010-04-01", "2010-07-31"),
    "euro_debt": ("2011-07-01", "2012-06-30"),
    "covid": ("2020-02-01", "2020-09-30"),
    "inflation": ("2021-11-01", "2022-12-31"),
}


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.model(x)


def make_tensor_dataset(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler_X: StandardScaler,
    scaler_y: StandardScaler,
    fit_scalers: bool = False,
) -> TensorDataset:
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


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion, optimizer) -> float:
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


def evaluate_loss(model: nn.Module, loader: Optional[DataLoader], criterion) -> Optional[float]:
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


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion,
    optimizer,
    num_epochs: int,
    patience: int,
):
    best_state = None
    best_val = np.inf
    patience_counter = 0
    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss = evaluate_loss(model, val_loader, criterion)
        print(
            f"[Initial Training] Epoch {epoch:03d} | Train Loss {train_loss:.6f} | Val Loss {val_loss:.6f}"
        )
        if val_loss is not None and val_loss < best_val - 1e-6:
            best_val = val_loss
            patience_counter = 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered during initial training.")
                break
    return best_state


def fine_tune_model(model: nn.Module, optimizer, loader: DataLoader, criterion, epochs: int):
    for epoch in range(epochs):
        fine_tune_loss = train_one_epoch(model, loader, criterion, optimizer)
    print(f"[Fine-tune] Trained for {epochs} epochs | Last loss {fine_tune_loss:.6f}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train hybrid model with optional stress evaluation.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--fine-tune-epochs", type=int, default=50)
    parser.add_argument("--horizon-days", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-csv", type=str, default="results/rolling_predictions.csv")
    parser.add_argument("--plot-path", type=str, help="Path for the baseline plot. Defaults to results/hybrid_sp500_h{horizon}.png")
    parser.add_argument("--stress-output-dir", type=str, default="results")
    parser.add_argument("--stress-start", type=str, help="Inclusive start date for manual stress window (YYYY-MM-DD).")
    parser.add_argument("--stress-end", type=str, help="Inclusive end date for manual stress window (YYYY-MM-DD).")
    parser.add_argument("--stress-label", type=str, default="stress")
    parser.add_argument("--stress-only", action="store_true", help="Skip saving the baseline outputs and only emit stress-window artifacts.")
    if STRESS_WINDOWS:
        parser.add_argument(
            "--stress-preset",
            action="append",
            choices=sorted(STRESS_WINDOWS.keys()),
            help="Run additional evaluations for one or more named stress regimes (can be repeated).",
        )
    return parser.parse_args()


def load_and_prepare_data():
    def _load(path: str) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.sort_values("date").dropna()

    InterestR_df = _load("data/predictions/dgs10_backtest.csv")
    credit_spread_df = _load("data/predictions/baa10ym_backtest.csv")
    inflation_df = _load("data/predictions/cpiaucsl_backtest.csv")
    oil_price_df = _load("data/predictions/dcoilwtico_backtest.csv")
    fed_funds_df = _load("data/predictions/fedfunds_backtest.csv")
    industrial_prod_df = _load("data/predictions/indpro_backtest.csv")
    yield_spread_df = _load("data/predictions/t10y2y_backtest.csv")
    unemployment_rate_df = _load("data/predictions/unrate_backtest.csv")
    sp500_df = _load("data/raw/sp500.csv")

    macro_dfs = [
        InterestR_df,
        credit_spread_df,
        inflation_df,
        oil_price_df,
        fed_funds_df,
        industrial_prod_df,
        yield_spread_df,
        unemployment_rate_df,
    ]

    def _resample_daily(df: pd.DataFrame) -> pd.DataFrame:
        return df.set_index("date").resample("D").ffill().reset_index()

    macro_dfs = [_resample_daily(df.copy()) for df in macro_dfs]

    (
        InterestR_df,
        credit_spread_df,
        inflation_df,
        oil_price_df,
        fed_funds_df,
        industrial_prod_df,
        yield_spread_df,
        unemployment_rate_df,
    ) = macro_dfs

    dfs = [
        InterestR_df,
        credit_spread_df,
        inflation_df,
        oil_price_df,
        fed_funds_df,
        industrial_prod_df,
        yield_spread_df,
        unemployment_rate_df,
        sp500_df,
    ]

    df_merged = reduce(lambda left, right: pd.merge(left, right, on="date", how="outer"), dfs)
    df_merged = df_merged.sort_values("date").ffill().dropna().reset_index(drop=True)

    print("Merged shape:", df_merged.shape)
    print(df_merged.head())
    print("After merge:", df_merged.shape)
    print("Date range:", df_merged["date"].min(), "→", df_merged["date"].max())
    print(df_merged.tail())

    return df_merged


def plot_series(df: pd.DataFrame, title: str, output_path: Path, show: bool):
    plt.figure(figsize=(12, 5))
    plt.plot(df["date"], df["sp500_true"], label="True")
    plt.plot(df["date"], df["sp500_pred"], label="Predicted")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    if show:
        plt.show()
    else:
        plt.close()


def evaluate_stress_window(
    results_df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp, label: str, output_dir: Path
):
    start_ts = pd.to_datetime(start)
    end_ts = pd.to_datetime(end)
    window_df = results_df[
        (results_df["date"] >= start_ts) & (results_df["date"] <= end_ts)
    ].copy()
    if window_df.empty:
        raise ValueError(
            f"Stress window {label} produced no rows in [{start_ts.date()}, {end_ts.date()}]."
        )
    rmse = np.sqrt(mean_squared_error(window_df["sp500_true"], window_df["sp500_pred"]))
    r2 = r2_score(window_df["sp500_true"], window_df["sp500_pred"])
    safe_label = label.replace(" ", "_")
    csv_path = output_dir / f"rolling_predictions_{safe_label}.csv"
    window_df.to_csv(csv_path, index=False)
    plot_path = output_dir / f"rolling_predictions_{safe_label}.png"
    plot_series(window_df, f"Hybrid Forecast ({label})", plot_path, show=False)
    print(
        f"[Stress {label}] {start_ts.date()} → {end_ts.date()} | RMSE {rmse:.4f} | R^2 {r2:.4f}"
    )
    print(f"[Stress {label}] Saved -> {csv_path} and {plot_path}")


def run_stress_evaluations(args, results_df: pd.DataFrame):
    data_start = pd.to_datetime(results_df["date"].min())
    data_end = pd.to_datetime(results_df["date"].max())
    windows = []
    if args.stress_start or args.stress_end:
        manual_start = pd.to_datetime(args.stress_start) if args.stress_start else data_start
        manual_end = pd.to_datetime(args.stress_end) if args.stress_end else data_end
        windows.append((manual_start, manual_end, args.stress_label))
    if args.stress_preset:
        for preset in args.stress_preset:
            start, end = STRESS_WINDOWS[preset]
            windows.append((pd.to_datetime(start), pd.to_datetime(end), preset))
    if not windows:
        return
    output_dir = Path(args.stress_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for start, end, label in windows:
        clipped_start = max(start, data_start)
        clipped_end = min(end, data_end)
        if clipped_start > clipped_end:
            print(
                f"[Stress {label}] skipped: requested {start.date()} → {end.date()} "
                f"is outside available predictions ({data_start.date()} → {data_end.date()})."
            )
            continue
        if clipped_start > start or clipped_end < end:
            print(
                f"[Stress {label}] clipped to {clipped_start.date()} → {clipped_end.date()} "
                f"(requested {start.date()} → {end.date()})."
            )
        evaluate_stress_window(results_df, clipped_start, clipped_end, label, output_dir)


def main():
    args = parse_args()
    if args.stress_only and not (args.stress_start or args.stress_end or args.stress_preset):
        raise ValueError("--stress-only requires --stress-start/--stress-end or --stress-preset.")

    df_merged = load_and_prepare_data()
    true_features = [col for col in df_merged.columns if col.endswith("_true")]
    pred_features = [col for col in df_merged.columns if col.endswith("_pred")]
    input_dim = len(true_features)

    num_epochs = args.epochs
    patience = args.patience
    fine_tune_epochs = args.fine_tune_epochs
    horizon_days = args.horizon_days
    batch_size = args.batch_size

    total_rows = len(df_merged)
    if total_rows <= horizon_days:
        raise ValueError("Not enough rows to support the requested rolling horizon.")

    initial_train_size = max(horizon_days, int(total_rows * 0.6))
    if initial_train_size + horizon_days > total_rows:
        initial_train_size = total_rows - horizon_days
    if initial_train_size <= 0:
        raise ValueError("Insufficient data to leave a full forecasting buffer.")

    print(
        f"Config -> initial training rows: {initial_train_size}, horizon: {horizon_days} days, fine-tune epochs per block: {fine_tune_epochs}"
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

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
            f"[Window {window_idx}] {start_date} → {end_date} | RMSE {window_rmse:.4f} | R^2 {window_r2:.4f}"
        )

        rolling_predictions.append(future_pred)
        rolling_truth.append(y_future)
        rolling_dates.append(future_slice["date"].values)

        revealed_dataset = make_tensor_dataset(future_slice, true_features, scaler_X, scaler_y)
        revealed_loader = DataLoader(revealed_dataset, batch_size=batch_size, shuffle=True)
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
    print(
        f"Rolling horizon {horizon_days} days | Overall RMSE: {overall_rmse:.4f}, R^2: {overall_r2:.4f}"
    )

    results_df = pd.DataFrame(
        {
            "date": all_dates,
            "sp500_true": all_true,
            "sp500_pred": all_pred,
        }
    )

    if not args.stress_only:
        csv_path = Path(args.output_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(csv_path, index=False)
        plot_path = Path(args.plot_path) if args.plot_path else Path("results") / f"hybrid_sp500_h{horizon_days}.png"
        plot_series(
            results_df,
            f"Rolling S&P500 Forecast (horizon={horizon_days} days)",
            plot_path,
            show=True,
        )
        print(f"Saved hybrid forecast plot -> {plot_path}")
    else:
        print("[Stress-only] Skipping baseline CSV/plot outputs.")

    run_stress_evaluations(args, results_df)


if __name__ == "__main__":
    main()
