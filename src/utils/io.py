"""
I/O helpers.

The original factor notebook loads a txt file like 'IJR_20080114.txt' with a Date column.
This helper makes that loading reproducible.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd


def load_price_table(path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """
    Load a wide price table from a .txt/.csv/.tsv file.

    The file is expected to contain a date column plus one column per ticker/asset.
    Dates will be converted to int if possible, then set as the index.
    """
    path = Path(path)
    # try to infer delimiter
    if path.suffix.lower() in {".csv"}:
        df = pd.read_csv(path)
    else:
        # default to tab-separated as in the original notebook
        df = pd.read_table(path)

    if date_col not in df.columns:
        raise ValueError(f"Expected a '{date_col}' column in {path.name}")

    df[date_col] = df[date_col].astype("int", errors="ignore")
    df = df.set_index(date_col).sort_index()
    df = df.ffill()
    return df
