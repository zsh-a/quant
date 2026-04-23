"""Auto-label historical bars into a silver Brooks dataset.

Reads OHLCV bars from a parquet/jsonl input file, runs the Brooks rule
analyst plus N LLM analysts (loaded by name from
``AnalystRegistry``), and writes a :class:`GoldenDataset` of silver
samples wherever ``--min-agreement`` analysts agree.

The script is intentionally agnostic about *where* the bars come from:
the input file just needs the columns ``timestamp_ns, open, high, low,
close, volume`` (extra columns are ignored). For testing without an LLM
backend, omit ``--llm`` — the rule analyst alone will be enough to
produce silver samples wherever it fires (with ``--min-agreement 1``).

Usage
-----
    python scripts/brooks_label_silver.py \
        --input data/brooks/raw/btcusdt_5m.parquet \
        --output data/brooks/silver/btcusdt_5m.parquet \
        --symbol BTCUSDT --interval 5m \
        --llm openai:gpt-4o-mini --llm openai:gpt-4o \
        --min-agreement 2

    # Dry-run: scan only, do not write
    python scripts/brooks_label_silver.py \
        --input data/brooks/raw/sample.jsonl \
        --symbol BTC --interval 5m \
        --min-agreement 1 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.brooks.analyst.base import AnalystRegistry  # noqa: E402
from src.brooks.context import Bar  # noqa: E402
from src.brooks.eval.auto_label import AutoLabeler  # noqa: E402
from src.brooks.eval.golden import GoldenDataset  # noqa: E402

# Importing the analyst package triggers registration of "rule" + "llm".
import src.brooks.analyst  # noqa: E402,F401


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    bars = _load_bars(Path(args.input))
    if not bars:
        print(f"[no-data] {args.input!s} is empty; nothing to label")
        return 0

    rule_analyst = AnalystRegistry.build("rule")
    llm_analysts = []
    for spec in args.llm or []:
        params = _parse_llm_spec(spec)
        llm_analysts.append(AnalystRegistry.build("llm", **params))

    labeler = AutoLabeler(
        rule_analyst=rule_analyst,
        llm_analysts=llm_analysts,
        min_agreement=args.min_agreement,
        max_concurrent=args.max_concurrent,
        entry_tolerance=args.entry_tolerance,
        min_bars_for_label=args.min_bars,
    )

    samples = asyncio.run(
        labeler.label(bars=bars, symbol=args.symbol, interval=args.interval)
    )

    print(
        f"[done] scanned {len(bars)} bars; emitted {len(samples)} silver samples "
        f"(analysts={labeler.analyst_count}, min_agreement={labeler.min_agreement})"
    )

    if not samples or args.dry_run:
        if args.dry_run:
            print("[dry-run] not writing output file")
        return 0

    output = Path(args.output)
    GoldenDataset.from_samples(samples).save(output)
    print(f"[wrote] {len(samples)} samples → {output!s}")
    return 0


# ---------------------------------------------------------------------------
# CLI plumbing
# ---------------------------------------------------------------------------


def _parse_args(argv: Optional[List[str]]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--input", required=True, help="parquet or jsonl file with OHLCV bars")
    p.add_argument("--output", help="silver dataset output (parquet or jsonl)")
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", required=True, help='bar interval label, e.g. "5m"')
    p.add_argument(
        "--llm",
        action="append",
        default=[],
        help='Add an LLM analyst, e.g. "openai:gpt-4o" or '
        "JSON params via 'llm:{json}'",
    )
    p.add_argument("--min-agreement", type=int, default=2, dest="min_agreement")
    p.add_argument("--max-concurrent", type=int, default=4, dest="max_concurrent")
    p.add_argument("--entry-tolerance", type=float, default=0.005, dest="entry_tolerance")
    p.add_argument("--min-bars", type=int, default=20, dest="min_bars")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    if not args.dry_run and not args.output:
        p.error("--output is required unless --dry-run is set")
    return args


def _parse_llm_spec(spec: str) -> dict:
    """Parse ``provider:model`` or ``provider:{json-params}`` into kwargs."""
    if not spec:
        raise ValueError("empty --llm spec")
    if spec.startswith("{"):
        return json.loads(spec)
    if ":" not in spec:
        raise ValueError(
            f"invalid --llm spec {spec!r}; expected 'provider:model' or JSON object"
        )
    provider, rest = spec.split(":", 1)
    rest = rest.strip()
    if rest.startswith("{"):
        params = json.loads(rest)
        params.setdefault("provider", provider)
        return params
    return {"provider": provider, "model": rest}


# ---------------------------------------------------------------------------
# Bar loading
# ---------------------------------------------------------------------------


def _load_bars(path: Path) -> List[Bar]:
    if not path.exists():
        raise FileNotFoundError(f"input not found: {path!s}")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        import pandas as pd

        df = pd.read_parquet(path)
        return [_row_to_bar(row.to_dict()) for _, row in df.iterrows()]
    if suffix == ".jsonl":
        out: List[Bar] = []
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                out.append(_row_to_bar(json.loads(line)))
        return out
    raise ValueError(f"unsupported input suffix: {suffix!r}")


def _row_to_bar(row: dict) -> Bar:
    return Bar(
        timestamp_ns=int(row["timestamp_ns"]),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        volume=float(row.get("volume", 0.0)),
    )


if __name__ == "__main__":
    raise SystemExit(main())
