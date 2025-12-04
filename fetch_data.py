import os
import argparse
from pathlib import Path
from typing import Optional
import pandas as pd
import yfinance as yf
from fredapi import Fred

# Load FRED_API_KEY from config.txt 
cfg = Path("config.txt")
if cfg.exists():
    for line in cfg.read_text().splitlines():
        if line.strip().startswith("FRED_API_KEY="):
            os.environ["FRED_API_KEY"] = line.strip().split("=", 1)[1]
            break

FRED_SERIES = [
    {"series_id": "DGS10", "column": "dgs10", "filename": "fred_10y.csv"},
    {"series_id": "T10Y2Y", "column": "t10y2y", "filename": "t10y2y.csv"},
    {"series_id": "FEDFUNDS", "column": "fedfunds", "filename": "fedfunds.csv"},
    {"series_id": "CPIAUCSL", "column": "cpiaucsl", "filename": "cpiaucsl.csv"},
    {"series_id": "DCOILWTICO", "column": "dcoilwtico", "filename": "dcoilwtico.csv"},
    {"series_id": "BAA10YM", "column": "baa10ym", "filename": "baa10ym.csv"},
    {"series_id": "INDPRO", "column": "indpro", "filename": "indpro.csv"},
    {"series_id": "UNRATE", "column": "unrate", "filename": "unrate.csv"},
]

def ensure_dirs():
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure dataframe has single-level string columns without ticker suffixes."""
    if isinstance(df.columns, pd.MultiIndex):
        flat = []
        for col in df.columns:
            parts = [str(p) for p in col if p is not None and str(p).strip()]
            # drop leading ticker symbols like ^GSPC
            parts = [p for p in parts if p.upper() not in {"^GSPC", "GSPC"}]
            if not parts:
                parts = [str(col[-1])]
            flat.append("_".join(parts))
        df.columns = flat
    else:
        df.columns = [str(c) for c in df.columns]
    return df


def fetch_sp500() -> pd.DataFrame:
    print("[S&P 500] yfinance period=max ...")
    df = yf.download("^GSPC", period="max", progress=False, auto_adjust=False, group_by="column")
    if df.empty:
        raise RuntimeError("Empty ^GSPC from yfinance.")
    df = _flatten_columns(df)
    col_map = {c.lower().replace(" ", "").replace("_", ""): c for c in df.columns}
    if "adjclose" in col_map:
        price = df[col_map["adjclose"]]
    elif "close" in col_map:
        price = df[col_map["close"]]
    else:
        raise RuntimeError(f"Could not find close price column in {df.columns}.")
    out = price.rename("sp500_close").to_frame()
    out.columns.name = None
    out.index = pd.to_datetime(out.index)
    sp_path = Path("data/raw") / "sp500.csv"
    out.to_csv(sp_path, index_label="date")
    print(f"[S&P 500] -> {sp_path} ({len(out)} rows)")
    return out


def _init_fred() -> Fred:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY missing. Put it in config.txt or export it.")
    return Fred(api_key=key)


def fetch_fred_series(
    fred: Fred,
    series_id: str,
    column: str,
    start: Optional[str],
    end: Optional[str],
    filename: str,
) -> pd.DataFrame:
    print(f"[FRED] {series_id} -> {filename} ...")
    kwargs = {}
    if start:
        kwargs["observation_start"] = start
    if end:
        kwargs["observation_end"] = end
    s = fred.get_series(series_id, **kwargs)
    df = s.to_frame(column.lower())
    df.index = pd.to_datetime(df.index)
    out_path = Path("data/raw") / filename
    df.to_csv(out_path, index_label="date")
    print(f"[FRED] -> {out_path} ({len(df)} rows)")
    return df

def merge_and_save(sp500: pd.DataFrame, dgs10: pd.DataFrame) -> pd.DataFrame:
    print("[Merge] business-day align + returns ...")
    start = max(sp500.index.min(), dgs10.index.min())
    end = min(sp500.index.max(), dgs10.index.max())
    idx = pd.date_range(start=start, end=end, freq="B")

    # Reindex
    s = sp500.reindex(idx).rename_axis("date")
    f = dgs10.reindex(idx).rename_axis("date").ffill(limit=5)

    # Force canonical S&P column name
    if "sp500_close" not in s.columns:
        for c in ["sp500_close", "Adj Close", "Adj_Close", "Close", "close"]:
            if c in s.columns:
                s = s.rename(columns={c: "sp500_close"})
                break
    if "sp500_close" not in s.columns:
        raise KeyError(f"Couldn't find S&P close column in {list(s.columns)}")

    m = pd.concat([s, f], axis=1)

    # Returns from the guaranteed column
    m["sp500_ret_1d"] = m["sp500_close"].pct_change()
    m["sp500_ret_5d"] = m["sp500_close"].pct_change(5)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    m.to_csv("data/processed/merged.csv", index_label="date")
    print(f"[Merge] -> data/processed/merged.csv ({len(m)} rows)")
    return m

def main():
    argparse.ArgumentParser().parse_args()

    ensure_dirs()
    sp = fetch_sp500()
    # Use the S&P span for FRED request bounds
    start_iso = sp.index.min().date().isoformat()
    end_iso = sp.index.max().date().isoformat()
    fred = _init_fred()
    fred_results = {}
    for meta in FRED_SERIES:
        fred_results[meta["column"]] = fetch_fred_series(
            fred=fred,
            series_id=meta["series_id"],
            column=meta["column"],
            start=start_iso,
            end=end_iso,
            filename=meta["filename"],
        )
    fr = fred_results["dgs10"]
    merge_and_save(sp, fr)

if __name__ == "__main__":
    main()
