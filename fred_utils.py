import os
from pathlib import Path
from typing import Optional

import pandas as pd
from fredapi import Fred


def _resolve_api_key() -> str:
    key = os.getenv("FRED_API_KEY")
    if key:
        return key
    cfg = Path("config.txt")
    if cfg.exists():
        for line in cfg.read_text().splitlines():
            if line.strip().startswith("FRED_API_KEY="):
                key = line.strip().split("=", 1)[1]
                os.environ["FRED_API_KEY"] = key
                return key
    raise RuntimeError("FRED_API_KEY missing. Set it in env or config.txt.")


def ensure_fred_series(series_id: str, output_path: Path, column_name: Optional[str] = None) -> None:
    """
    Download the specified FRED series if the CSV file does not exist yet.
    """
    if output_path.exists():
        return
    key = _resolve_api_key()
    fred = Fred(api_key=key)
    print(f"[FRED] downloading {series_id} -> {output_path}")
    series = fred.get_series(series_id)
    if series.empty:
        raise RuntimeError(f"FRED returned empty series for {series_id}")
    column = column_name or series_id.lower()
    df = series.to_frame(column)
    df.index = pd.to_datetime(df.index)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index_label="date")
    print(f"[FRED] saved {len(df)} rows to {output_path}")


def load_series(path: Path, series_id: str, column_name: Optional[str] = None) -> pd.Series:
    """
    Ensure the series exists locally, then load it as a float Series indexed by date.
    """
    ensure_fred_series(series_id, path, column_name)
    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"{path} must contain a 'date' column.")
    col = (column_name or series_id.lower()).lower()
    if col not in df.columns:
        candidates = [c for c in df.columns if c != "date"]
        if len(candidates) != 1:
            raise ValueError(f"Unable to identify value column in {path}")
        col = candidates[0]
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    return df.set_index("date")[col].astype(float).dropna()
