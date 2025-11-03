import os
import argparse
from pathlib import Path
import pandas as pd
import yfinance as yf
from fredapi import Fred

# Load FRED_API_KEY from config.txt if present (format: FRED_API_KEY=xxxxxxxx...)
cfg = Path("config.txt")
for line in cfg.read_text().splitlines():
    if line.strip().startswith("FRED_API_KEY="):
        os.environ["FRED_API_KEY"] = line.strip().split("=", 1)[1]
        break

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
    out.to_csv("data/raw/sp500.csv", index_label="date")
    print(f"[S&P 500] -> data/raw/sp500.csv ({len(out)} rows)")
    return out

def fetch_fred_10y(start: str, end: str) -> pd.DataFrame:
    key = os.getenv("FRED_API_KEY")
    if not key:
        raise RuntimeError("FRED_API_KEY missing. Put it in config.txt or export it.")
    fred = Fred(api_key=key)
    print("[FRED] DGS10 ...")
    s = fred.get_series("DGS10", observation_start=start, observation_end=end)
    df = s.to_frame("dgs10")
    df.index = pd.to_datetime(df.index)
    df.to_csv("data/raw/fred_10y.csv", index_label="date")
    print(f"[FRED] -> data/raw/fred_10y.csv ({len(df)} rows)")
    return df

def load_sentiment_kaggle(csv_path: str) -> pd.DataFrame:
    """Expect CSV with columns: date, sentiment_compound (already computed)."""
    print(f"[Sentiment] Kaggle simple: {csv_path}")
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "sentiment_compound" not in df.columns:
        raise ValueError("Sentiment CSV must have 'date' and 'sentiment_compound' columns.")
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    df.to_csv("data/raw/sentiment.csv", index_label="date")
    print(f"[Sentiment] -> data/raw/sentiment.csv ({len(df)} rows)")
    return df

def merge_and_save(sp500: pd.DataFrame, dgs10: pd.DataFrame, sentiment: pd.DataFrame | None) -> pd.DataFrame:
    print("[Merge] business-day align + returns ...")
    start = max(sp500.index.min(), dgs10.index.min())
    end = min(sp500.index.max(), dgs10.index.max())
    idx = pd.date_range(start=start, end=end, freq="B")

    # Reindex
    s = sp500.reindex(idx).rename_axis("date")
    f = dgs10.reindex(idx).rename_axis("date").ffill(limit=5)

    # 🔒 Force canonical S&P column name
    if "sp500_close" not in s.columns:
        for c in ["sp500_close", "Adj Close", "Adj_Close", "Close", "close"]:
            if c in s.columns:
                s = s.rename(columns={c: "sp500_close"})
                break
    if "sp500_close" not in s.columns:
        raise KeyError(f"Couldn't find S&P close column in {list(s.columns)}")

    frames = [s, f]
    if sentiment is not None:
        sent = sentiment.reindex(idx).rename_axis("date").ffill(limit=2)
        if "sentiment_compound" not in sent.columns and len(sent.columns) == 1:
            sent = sent.rename(columns={sent.columns[0]: "sentiment_compound"})
        frames.append(sent)

    m = pd.concat(frames, axis=1)

    # Returns from the guaranteed column
    m["sp500_ret_1d"] = m["sp500_close"].pct_change()
    m["sp500_ret_5d"] = m["sp500_close"].pct_change(5)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    m.to_csv("data/processed/merged.csv", index_label="date")
    print(f"[Merge] -> data/processed/merged.csv ({len(m)} rows)")
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sentiment", choices=["none", "kaggle"], default="none")
    ap.add_argument("--kaggle-path", type=str, help="CSV with date,sentiment_compound")
    args = ap.parse_args()

    ensure_dirs()
    sp = fetch_sp500()
    # Use the S&P span for FRED request bounds
    start_iso = sp.index.min().date().isoformat()
    end_iso = sp.index.max().date().isoformat()
    fr = fetch_fred_10y(start_iso, end_iso)

    sent = None
    if args.sentiment == "kaggle":
        if not args.kaggle_path:
            raise ValueError("--kaggle-path required when --sentiment kaggle")
        sent = load_sentiment_kaggle(args.kaggle_path)

    merge_and_save(sp, fr, sent)

if __name__ == "__main__":
    main()
