"""Golden dataset loader for Brooks evaluation.

A :class:`GoldenSample` is a single labeled bar window: the bars provide
the analyst's input context, ``target_bar_idx`` points at the bar where
the expected signal fires, and ``expected_*`` fields encode the ground
truth (pattern, side, entry/stop/target, regime, HTF alignment).

Two storage formats are supported transparently:

* **parquet** — one row per sample; ``bars`` is stored as a JSON-encoded
  string in the column so each parquet row stays scalar.
* **jsonl**   — one JSON object per line, mirroring the parquet schema.

Loaders pick the format from the file suffix; directories are walked
recursively and every matching file is concatenated into a single
:class:`GoldenDataset`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, List, Literal, Optional

from src.brooks.context import Bar

__all__ = ["GoldenSample", "GoldenDataset"]


SourceTag = Literal["human", "silver"]


@dataclass
class GoldenSample:
    """One labeled (bars → expected signal) data point."""

    id: str
    symbol: str
    interval: str
    bars: List[Bar]
    target_bar_idx: int
    expected_pattern: str
    expected_side: Literal["long", "short"]
    expected_entry: float
    expected_stop: float
    expected_target: Optional[float] = None
    regime: str = "unknown"
    htf_aligned: bool = False
    source: SourceTag = "human"
    reasoning: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.bars:
            raise ValueError(f"GoldenSample {self.id!r}: bars must be non-empty")
        if not 0 <= self.target_bar_idx < len(self.bars):
            raise ValueError(
                f"GoldenSample {self.id!r}: target_bar_idx={self.target_bar_idx} out of range [0, {len(self.bars)})"
            )
        if self.expected_side not in ("long", "short"):
            raise ValueError(f"GoldenSample {self.id!r}: invalid side {self.expected_side!r}")
        if self.expected_entry == self.expected_stop:
            raise ValueError(f"GoldenSample {self.id!r}: entry must differ from stop")

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def context_bars(self) -> List[Bar]:
        """Bars up to and including ``target_bar_idx`` (analyst input)."""
        return self.bars[: self.target_bar_idx + 1]

    @property
    def future_bars(self) -> List[Bar]:
        """Bars strictly after ``target_bar_idx`` (used for hit-rate scoring)."""
        return self.bars[self.target_bar_idx + 1 :]

    @property
    def one_r(self) -> float:
        """Absolute per-share risk (|entry − stop|)."""
        return abs(self.expected_entry - self.expected_stop)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise to a JSON-friendly dict (bars become a list of dicts)."""
        d = asdict(self)
        d["bars"] = [_bar_to_dict(b) for b in self.bars]
        return d

    @classmethod
    def from_dict(cls, raw: dict) -> "GoldenSample":
        """Inverse of :meth:`to_dict` — parquet rows pass through here too."""
        data = dict(raw)
        bars_raw = data.pop("bars", None)
        if bars_raw is None:
            raise ValueError("GoldenSample.from_dict: missing 'bars' field")
        if isinstance(bars_raw, str):
            bars_raw = json.loads(bars_raw)
        bars = [_bar_from_dict(b) for b in bars_raw]
        meta = data.pop("meta", None)
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (TypeError, ValueError):
                meta = {}
        if meta is None:
            meta = {}
        target_val = data.get("expected_target")
        if target_val is not None:
            try:
                if float(target_val) != float(target_val):  # NaN check
                    data["expected_target"] = None
                else:
                    data["expected_target"] = float(target_val)
            except (TypeError, ValueError):
                data["expected_target"] = None
        return cls(bars=bars, meta=meta, **data)


@dataclass
class GoldenDataset:
    """A collection of :class:`GoldenSample`'s with filter/iter helpers."""

    samples: List[GoldenSample] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path | str) -> "GoldenDataset":
        """Load samples from a file or directory.

        Supported suffixes: ``.parquet``, ``.jsonl`` (and ``.json`` as a
        single-object-or-array convenience). Directories are walked
        recursively and every matching file contributes its samples.
        """
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"GoldenDataset.load: {p!s} does not exist")
        files: List[Path] = []
        if p.is_dir():
            for suffix in ("*.parquet", "*.jsonl", "*.json"):
                files.extend(sorted(p.rglob(suffix)))
            if not files:
                return cls(samples=[])
        else:
            files = [p]

        samples: List[GoldenSample] = []
        for fp in files:
            samples.extend(_load_one(fp))
        return cls(samples=samples)

    @classmethod
    def from_samples(cls, samples: Iterable[GoldenSample]) -> "GoldenDataset":
        return cls(samples=list(samples))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path | str) -> Path:
        """Write the dataset to ``path``; format is chosen by the suffix."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        suffix = out.suffix.lower()
        if suffix == ".parquet":
            _write_parquet(self.samples, out)
        elif suffix in {".jsonl", ".json"}:
            _write_jsonl(self.samples, out)
        else:
            raise ValueError(f"GoldenDataset.save: unsupported suffix {suffix!r}")
        return out

    # ------------------------------------------------------------------
    # Filtering
    # ------------------------------------------------------------------

    def filter(self, **criteria: Any) -> "GoldenDataset":
        """Return a new dataset whose samples match every keyword filter.

        Filter values may be a single value (equality) or a collection
        (membership). Unknown keys raise :class:`AttributeError`.
        """
        if not criteria:
            return GoldenDataset(samples=list(self.samples))

        def _matches(sample: GoldenSample) -> bool:
            for key, want in criteria.items():
                if not hasattr(sample, key):
                    raise AttributeError(f"GoldenSample has no attribute {key!r}")
                got = getattr(sample, key)
                if isinstance(want, (list, tuple, set, frozenset)):
                    if got not in want:
                        return False
                elif got != want:
                    return False
            return True

        return GoldenDataset(samples=[s for s in self.samples if _matches(s)])

    # ------------------------------------------------------------------
    # Container protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[GoldenSample]:
        return iter(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> GoldenSample:
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


_BAR_FIELDS = ("timestamp_ns", "open", "high", "low", "close", "volume")


def _bar_to_dict(bar: Bar) -> dict:
    return {f: getattr(bar, f) for f in _BAR_FIELDS}


def _bar_from_dict(raw: dict) -> Bar:
    return Bar(
        timestamp_ns=int(raw["timestamp_ns"]),
        open=float(raw["open"]),
        high=float(raw["high"]),
        low=float(raw["low"]),
        close=float(raw["close"]),
        volume=float(raw.get("volume", 0.0)),
    )


def _load_one(path: Path) -> List[GoldenSample]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return _read_parquet(path)
    if suffix == ".jsonl":
        return _read_jsonl(path)
    if suffix == ".json":
        return _read_json(path)
    return []


def _read_parquet(path: Path) -> List[GoldenSample]:
    import pandas as pd  # local import to keep cold-import light

    df = pd.read_parquet(path)
    out: List[GoldenSample] = []
    for _, row in df.iterrows():
        out.append(GoldenSample.from_dict(row.to_dict()))
    return out


def _read_jsonl(path: Path) -> List[GoldenSample]:
    out: List[GoldenSample] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            out.append(GoldenSample.from_dict(json.loads(line)))
    return out


def _read_json(path: Path) -> List[GoldenSample]:
    with path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict):
        data = [data]
    return [GoldenSample.from_dict(d) for d in data]


def _write_parquet(samples: Iterable[GoldenSample], path: Path) -> None:
    import pandas as pd

    rows = []
    for s in samples:
        d = s.to_dict()
        d["bars"] = json.dumps(d["bars"])
        d["meta"] = json.dumps(d.get("meta") or {})
        rows.append(d)
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)


def _write_jsonl(samples: Iterable[GoldenSample], path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for s in samples:
            fh.write(json.dumps(s.to_dict()) + "\n")
