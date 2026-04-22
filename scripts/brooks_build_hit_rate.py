"""Build the Brooks hit-rate parquet table from session-DB backtest history.

Reads ``simulation_run_steps`` rows from ``session_db.SessionDB``, looking
for paired ``signal`` / ``fill_close`` events tagged with the brooks
strategy name. For each emitted signal we compute:

* whether the trade reached 1R within ``--horizon-bars`` bars,
* whether it reached 2R,
* the realized R at the eventual close.

Results are bucketed by ``(pattern, regime, htf_aligned, side)``. Buckets
with fewer than ``--min-samples`` rows are dropped (the runtime falls
back to the prior). The output is written as parquet to
``data/brooks/hit_rate_table.parquet`` (override with ``--output``).

Dry-run / no-data behaviour
---------------------------
* ``--dry-run`` emits an empty parquet (with the canonical schema) without
  touching the DB. This is the smoke-test mode and the path used by CI
  before any backtest data exists.
* If the DB is reachable but no brooks events are found, the script also
  emits an empty parquet — the runtime simply falls back to the prior for
  every bucket.

Usage
-----
    python scripts/brooks_build_hit_rate.py --dry-run
    python scripts/brooks_build_hit_rate.py --strategy brooks_v2 \
        --horizon-bars 24 --min-samples 30
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brooks.decision.hit_rate import (  # noqa: E402
    DEFAULT_TABLE_PATH,
    MIN_SUFFICIENT_SAMPLES,
    REQUIRED_COLUMNS,
    HitRateTable,
)

DEFAULT_HORIZON_BARS = 24
DEFAULT_STRATEGY = "brooks_v2"


@dataclass(frozen=True)
class TradeOutcome:
    pattern: str
    regime: str
    htf_aligned: bool
    side: str
    hit_1r: bool
    hit_2r: bool
    realized_r: float


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    output = Path(args.output)

    if args.dry_run:
        print(f"[dry-run] writing empty hit-rate table to {output}")
        HitRateTable.empty().save(output)
        return 0

    db_path = args.db_path
    if db_path is None:
        try:
            from src.config.paths import SESSIONS_DB

            db_path = str(SESSIONS_DB)
        except Exception:  # pragma: no cover - settings import failure shouldn't crash dry-runs
            db_path = "sessions.db"

    if not Path(db_path).exists():
        print(f"[no-data] session DB not found at {db_path}; writing empty table to {output}")
        HitRateTable.empty().save(output)
        return 0

    outcomes = list(_collect_outcomes(db_path, strategy=args.strategy, horizon_bars=args.horizon_bars))
    if not outcomes:
        print(f"[no-data] no brooks signal/fill pairs found for strategy={args.strategy!r}; writing empty table")
        HitRateTable.empty().save(output)
        return 0

    df = _bucket(outcomes, min_samples=args.min_samples)
    HitRateTable(df).save(output)
    print(f"wrote {len(df)} buckets ({len(outcomes)} samples) to {output}")
    return 0


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--db-path", default=None, help="Override session_db path (defaults to settings.SESSIONS_DB).")
    p.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help=f"Strategy name to filter on (default: {DEFAULT_STRATEGY}).",
    )
    p.add_argument(
        "--horizon-bars",
        type=int,
        default=DEFAULT_HORIZON_BARS,
        help=f"Bars after entry to look for 1R/2R hits (default: {DEFAULT_HORIZON_BARS}).",
    )
    p.add_argument(
        "--min-samples",
        type=int,
        default=MIN_SUFFICIENT_SAMPLES,
        help=f"Drop buckets with fewer rows (default: {MIN_SUFFICIENT_SAMPLES}).",
    )
    p.add_argument(
        "--output",
        default=str(DEFAULT_TABLE_PATH),
        help=f"Parquet output path (default: {DEFAULT_TABLE_PATH}).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip DB read, write an empty parquet with the canonical schema.",
    )
    return p.parse_args(argv)


def _collect_outcomes(db_path: str, *, strategy: str, horizon_bars: int) -> Iterable[TradeOutcome]:
    """Stream completed trade outcomes from the simulation_run_steps event log.

    The brooks strategy emits ``signal`` events when a setup fires and
    ``fill_close`` events when the position closes (stop / target / time
    stop). We pair them by ``run_id + signal_id``; if no pairing can be
    found within ``horizon_bars`` we treat the signal as ungratified and
    skip it (the prior takes over for that bucket).
    """
    if horizon_bars <= 0:
        return
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        try:
            rows = conn.execute(
                """
                SELECT s.run_id, s.step_index, s.event_type, s.payload
                FROM simulation_run_steps s
                JOIN simulation_runs r ON r.run_id = s.run_id
                JOIN simulation_jobs j ON j.job_id = r.job_id
                WHERE j.strategy_name = ?
                  AND s.event_type IN ('signal', 'fill_close')
                ORDER BY s.run_id, s.step_index
                """,
                (strategy,),
            ).fetchall()
        except sqlite3.OperationalError:
            return

    if not rows:
        return

    open_signals: Dict[Tuple[str, str], Dict] = {}
    for row in rows:
        try:
            payload = json.loads(row["payload"]) if row["payload"] else {}
        except (TypeError, ValueError):
            continue
        sig_id = payload.get("signal_id") or payload.get("decision_id")
        if not sig_id:
            continue
        key = (row["run_id"], sig_id)
        if row["event_type"] == "signal":
            open_signals[key] = {"step_index": row["step_index"], "payload": payload}
            continue
        opened = open_signals.pop(key, None)
        if opened is None:
            continue
        if row["step_index"] - opened["step_index"] > horizon_bars:
            continue
        outcome = _build_outcome(opened["payload"], payload)
        if outcome is not None:
            yield outcome


def _build_outcome(signal_payload: dict, fill_payload: dict) -> Optional[TradeOutcome]:
    pattern = signal_payload.get("pattern")
    side = signal_payload.get("side")
    regime = signal_payload.get("regime") or fill_payload.get("regime") or "unknown"
    if not pattern or side not in {"long", "short"}:
        return None
    htf_aligned = bool(signal_payload.get("htf_aligned", False))
    realized_r = fill_payload.get("realized_r")
    if realized_r is None:
        return None
    realized_r = float(realized_r)
    hit_1r = bool(fill_payload.get("hit_1r", realized_r >= 1.0))
    hit_2r = bool(fill_payload.get("hit_2r", realized_r >= 2.0))
    return TradeOutcome(
        pattern=str(pattern),
        regime=str(regime),
        htf_aligned=htf_aligned,
        side=str(side),
        hit_1r=hit_1r,
        hit_2r=hit_2r,
        realized_r=realized_r,
    )


def _bucket(outcomes: Iterable[TradeOutcome], *, min_samples: int) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "pattern": o.pattern,
                "regime": o.regime,
                "htf_aligned": bool(o.htf_aligned),
                "side": o.side,
                "hit_1r": int(o.hit_1r),
                "hit_2r": int(o.hit_2r),
                "realized_r": float(o.realized_r),
            }
            for o in outcomes
        ]
    )
    if df.empty:
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS))
    grouped = (
        df.groupby(["pattern", "regime", "htf_aligned", "side"], sort=False)
        .agg(
            samples=("hit_1r", "count"),
            hit_rate_1r=("hit_1r", "mean"),
            hit_rate_2r=("hit_2r", "mean"),
            avg_realized_r=("realized_r", "mean"),
        )
        .reset_index()
    )
    return grouped[grouped["samples"] >= min_samples][list(REQUIRED_COLUMNS)].reset_index(drop=True)


if __name__ == "__main__":
    raise SystemExit(main())
