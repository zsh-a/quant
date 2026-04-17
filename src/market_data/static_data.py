"""Unified loader for static reference data (ETF list, industry index, etc).

All static CSV data lives under config/static_data/ and is loaded lazily
with caching. This replaces scattered pd.read_csv("xxx.csv") calls that
depended on the working directory.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

# Project root: three levels up from this file (src/market_data/static_data.py -> project root)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_STATIC_DIR = _PROJECT_ROOT / "config" / "static_data"


def _resolve(filename: str) -> Path:
    path = _STATIC_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Static data file not found: {path}. Expected under config/static_data/")
    return path


@lru_cache(maxsize=1)
def load_etf_list() -> pd.DataFrame:
    """Load ETF reference list.

    Returns DataFrame with columns: code, type, name
    """
    return pd.read_csv(
        _resolve("all_etf.csv"),
        names=["code", "type", "name"],
        dtype=str,
    )


def get_etf_codes() -> list[str]:
    """Get list of ETF codes (string)."""
    return load_etf_list()["code"].tolist()


@lru_cache(maxsize=1)
def load_sw_industry(index_col: str = "index") -> pd.DataFrame:
    """Load Shenwan industry index reference.

    Args:
        index_col: Column to use as index ('index' for code col, '代码' for Chinese header)

    Returns DataFrame with columns: index/代码, name, count
    """
    return pd.read_csv(_resolve("sw_industry.csv"), index_col=index_col)
