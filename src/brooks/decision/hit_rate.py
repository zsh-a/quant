"""Historical hit-rate lookup table.

The decision layer scores signals via the Trader's Equation, which needs
``p`` (probability of reaching the target) for each (pattern, regime,
htf_aligned, side) bucket. This module loads a pre-computed parquet table
of those statistics and provides a typed lookup with prior fallbacks for
buckets without enough samples.

The table is built offline by ``scripts/brooks_build_hit_rate.py`` from
session-DB backtest history.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd

DEFAULT_TABLE_PATH = Path("data/brooks/hit_rate_table.parquet")
DEFAULT_PRIOR_HIT_RATE = 0.55
DEFAULT_PRIOR_AVG_R = 0.0
MIN_SUFFICIENT_SAMPLES = 30

REQUIRED_COLUMNS = (
    "pattern",
    "regime",
    "htf_aligned",
    "side",
    "samples",
    "hit_rate_1r",
    "hit_rate_2r",
    "avg_realized_r",
)


@dataclass(frozen=True)
class HitRateKey:
    pattern: str
    regime: str
    htf_aligned: bool
    side: str

    def as_tuple(self) -> Tuple[str, str, bool, str]:
        return (self.pattern, self.regime, bool(self.htf_aligned), self.side)


class HitRateTable:
    """In-memory wrapper around a parquet-backed hit-rate index.

    Lookups return either the bucket statistics (when present) or a prior
    distribution. Use :meth:`is_sufficient` to decide whether to trust the
    bucket — buckets with fewer than ``min_samples`` rows are treated as
    insufficient and the prior should be used instead.
    """

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            self._index: Dict[Tuple[str, str, bool, str], Dict[str, Any]] = {}
            return
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"hit_rate dataframe missing columns: {missing}")
        normalized = df.copy()
        normalized["htf_aligned"] = normalized["htf_aligned"].astype(bool)
        indexed = normalized.set_index(["pattern", "regime", "htf_aligned", "side"])
        self._index = {tuple(k): _row_to_stats(row) for k, row in indexed.iterrows()}

    @classmethod
    def load(cls, path: Path = DEFAULT_TABLE_PATH) -> "HitRateTable":
        path = Path(path)
        if not path.exists():
            return cls(pd.DataFrame(columns=list(REQUIRED_COLUMNS)))
        df = pd.read_parquet(path)
        return cls(df)

    @classmethod
    def empty(cls) -> "HitRateTable":
        return cls(pd.DataFrame(columns=list(REQUIRED_COLUMNS)))

    def lookup(
        self,
        key: HitRateKey,
        default: float = DEFAULT_PRIOR_HIT_RATE,
    ) -> Dict[str, Any]:
        """Return ``{hit_rate_1r, hit_rate_2r, avg_r, samples}`` for the bucket.

        Falls back to a prior with ``samples=0`` when the key is missing.
        """
        stats = self._index.get(key.as_tuple())
        if stats is None:
            return {
                "hit_rate_1r": float(default),
                "hit_rate_2r": float(default) * 0.6,
                "avg_r": DEFAULT_PRIOR_AVG_R,
                "samples": 0,
            }
        return dict(stats)

    def is_sufficient(self, key: HitRateKey, min_samples: int = MIN_SUFFICIENT_SAMPLES) -> bool:
        stats = self._index.get(key.as_tuple())
        if stats is None:
            return False
        return int(stats.get("samples", 0)) >= int(min_samples)

    def __len__(self) -> int:
        return len(self._index)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, HitRateKey):
            return False
        return key.as_tuple() in self._index

    def to_dataframe(self) -> pd.DataFrame:
        if not self._index:
            return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
        rows = []
        for (pattern, regime, htf_aligned, side), stats in self._index.items():
            rows.append(
                {
                    "pattern": pattern,
                    "regime": regime,
                    "htf_aligned": bool(htf_aligned),
                    "side": side,
                    "samples": int(stats["samples"]),
                    "hit_rate_1r": float(stats["hit_rate_1r"]),
                    "hit_rate_2r": float(stats["hit_rate_2r"]),
                    "avg_realized_r": float(stats["avg_r"]),
                }
            )
        return pd.DataFrame(rows, columns=list(REQUIRED_COLUMNS))

    def save(self, path: Path = DEFAULT_TABLE_PATH) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.to_dataframe().to_parquet(path, index=False)
        return path


def _row_to_stats(row: pd.Series) -> Dict[str, Any]:
    return {
        "hit_rate_1r": float(row["hit_rate_1r"]),
        "hit_rate_2r": float(row["hit_rate_2r"]),
        "avg_r": float(row["avg_realized_r"]),
        "samples": int(row["samples"]),
    }


__all__ = [
    "DEFAULT_PRIOR_HIT_RATE",
    "DEFAULT_TABLE_PATH",
    "MIN_SUFFICIENT_SAMPLES",
    "HitRateKey",
    "HitRateTable",
]
