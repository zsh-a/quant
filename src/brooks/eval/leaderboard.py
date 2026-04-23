"""Cross-analyst leaderboard over a Brooks :class:`GoldenDataset`.

The leaderboard reuses :class:`EvalRunner` / :class:`EvalReport` to score
each configured analyst (rule / LLM / VLM / ensemble.*) against the same
golden dataset, then renders a single HTML page that lets us compare
quality, cost, and per-bucket behaviour side by side.

Config is a YAML file (see ``config/brooks/leaderboard.yaml``) so adding
a new model is a one-line change — the :class:`AnalystFactory` resolves
``{type: llm, model: ...}`` or ``{type: ensemble.critic, ...}`` specs
into ready-to-run :class:`Analyst` instances.

Results are persisted to a parquet log on every run so weekly history
can be queried for trend charts.
"""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from src.brooks.analyst.base import Analyst, AnalystRegistry
from src.brooks.analyst.ensemble import CriticAnalyst, RouterAnalyst, VoteAnalyst
from src.brooks.eval.golden import GoldenDataset
from src.brooks.eval.report import EvalReport
from src.brooks.eval.runner import EvalRunner

__all__ = [
    "AnalystSpec",
    "LeaderboardConfig",
    "LeaderboardEntry",
    "Leaderboard",
    "AnalystFactory",
    "DEFAULT_COST_PER_MTOKEN",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------


DEFAULT_COST_PER_MTOKEN: Dict[str, Dict[str, float]] = {
    "claude-opus-4-7": {"input": 15.0, "output": 75.0},
    "claude-opus-4": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-sonnet-4": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    "gpt-4.1": {"input": 2.0, "output": 8.0},
    "gpt-4o": {"input": 2.5, "output": 10.0},
    "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "gemini-2.5-flash": {"input": 0.3, "output": 2.5},
}


def _cost_per_run(model: str, avg_in: float, avg_out: float, table: Dict[str, Dict[str, float]]) -> float:
    """USD per call for the given model and average token usage."""
    rates = table.get(model)
    if not rates:
        return 0.0
    return (avg_in * float(rates.get("input", 0.0)) + avg_out * float(rates.get("output", 0.0))) / 1_000_000.0


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class AnalystSpec:
    """One analyst entry in ``leaderboard.yaml``."""

    type: str
    model: Optional[str] = None
    label: Optional[str] = None
    # Ensemble / nested analyst specs — values are strings ("llm:claude-opus-4-7")
    # or nested spec dicts.
    params: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "AnalystSpec":
        if "type" not in raw:
            raise ValueError(f"AnalystSpec missing 'type': {raw!r}")
        known = {"type", "model", "label"}
        params = {k: v for k, v in raw.items() if k not in known}
        return cls(
            type=str(raw["type"]),
            model=raw.get("model"),
            label=raw.get("label"),
            params=params,
        )

    def display_name(self) -> str:
        if self.label:
            return self.label
        if self.model:
            return f"{self.type}:{self.model}"
        return self.type


@dataclass
class LeaderboardConfig:
    """Loaded ``leaderboard.yaml``."""

    dataset: Path
    analysts: List[AnalystSpec]
    output_dir: Path = Path("data/brooks/leaderboards")
    history_path: Path = Path("data/brooks/leaderboard.parquet")
    max_concurrent: int = 4
    schedule: Optional[str] = None
    # Default cost table — YAML can override per model via ``cost_per_mtoken``.
    cost_per_mtoken: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {k: dict(v) for k, v in DEFAULT_COST_PER_MTOKEN.items()}
    )

    @classmethod
    def load(cls, path: Path | str) -> "LeaderboardConfig":
        import yaml

        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"leaderboard config must be a YAML mapping, got {type(raw).__name__}")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "LeaderboardConfig":
        if "dataset" not in raw:
            raise ValueError("leaderboard config missing 'dataset'")
        analyst_specs = [AnalystSpec.from_dict(a) for a in raw.get("analysts", [])]
        if not analyst_specs:
            raise ValueError("leaderboard config must list at least one analyst")
        cost_table = dict(DEFAULT_COST_PER_MTOKEN)
        for model, rates in (raw.get("cost_per_mtoken") or {}).items():
            cost_table[model] = dict(rates)
        return cls(
            dataset=Path(raw["dataset"]),
            analysts=analyst_specs,
            output_dir=Path(raw.get("output_dir", "data/brooks/leaderboards")),
            history_path=Path(raw.get("history_path", "data/brooks/leaderboard.parquet")),
            max_concurrent=int(raw.get("max_concurrent", 4)),
            schedule=raw.get("schedule"),
            cost_per_mtoken=cost_table,
        )


# ---------------------------------------------------------------------------
# Analyst factory
# ---------------------------------------------------------------------------


ProviderFactory = Callable[[str], Any]


class AnalystFactory:
    """Resolve :class:`AnalystSpec` → ready :class:`Analyst` instance.

    LLM/VLM specs require a ``provider_factory`` — a callable that
    returns a :class:`Provider` for a given model id. Leaving it as
    ``None`` means LLM/VLM specs will raise; this keeps pure-rule
    leaderboards runnable without any provider wiring.
    """

    def __init__(self, provider_factory: Optional[ProviderFactory] = None) -> None:
        self._provider_factory = provider_factory

    def build(self, spec: AnalystSpec) -> Analyst:
        t = spec.type
        if t == "rule":
            return AnalystRegistry.build("rule", **spec.params)
        if t == "llm":
            return self._build_llm(spec, analyst_name="llm")
        if t == "vlm":
            # VLM analysts are optional — only instantiate if registered.
            if "vlm" in AnalystRegistry.all():
                return self._build_llm(spec, analyst_name="vlm")
            raise KeyError("VLM analyst is not registered in this build")
        if t == "ensemble.critic":
            producer = self._build_nested(spec.params.get("producer"))
            critic = self._build_nested(spec.params.get("critic"))
            overlay = spec.params.get("critic_prompt_overlay")
            return CriticAnalyst(producer=producer, critic=critic, critic_prompt_overlay=overlay)
        if t == "ensemble.vote":
            analysts = [self._build_nested(a) for a in spec.params.get("analysts", [])]
            weights = spec.params.get("weights")
            return VoteAnalyst(
                analysts=analysts,
                weights=weights,
                min_agree_count=int(spec.params.get("min_agree_count", 2)),
            )
        if t == "ensemble.router":
            from src.brooks.regime import BrooksRegime

            routes_raw = spec.params.get("routes", {}) or {}
            routes = {BrooksRegime(k): self._build_nested(v) for k, v in routes_raw.items()}
            default = self._build_nested(spec.params.get("default"))
            return RouterAnalyst(routes=routes, default=default)
        raise ValueError(f"Unsupported analyst type: {t!r}")

    def _build_llm(self, spec: AnalystSpec, *, analyst_name: str) -> Analyst:
        if self._provider_factory is None:
            raise RuntimeError(
                f"{analyst_name}:{spec.model} requires a provider_factory; "
                "pass provider_factory=... to AnalystFactory or Leaderboard"
            )
        if not spec.model:
            raise ValueError(f"{analyst_name} spec must include 'model'")
        provider = self._provider_factory(spec.model)
        kwargs = dict(spec.params)
        kwargs.pop("provider", None)
        return AnalystRegistry.build(analyst_name, provider=provider, model=spec.model, **kwargs)

    def _build_nested(self, raw: Any) -> Analyst:
        """Accept either a string ``"llm:claude-opus-4-7"`` or a full spec dict."""
        if raw is None:
            raise ValueError("ensemble analyst missing nested analyst spec")
        if isinstance(raw, str):
            if ":" in raw:
                head, _, tail = raw.partition(":")
                return self.build(AnalystSpec(type=head, model=tail))
            return self.build(AnalystSpec(type=raw))
        if isinstance(raw, dict):
            return self.build(AnalystSpec.from_dict(raw))
        raise TypeError(f"unsupported nested analyst spec: {type(raw).__name__}")


# ---------------------------------------------------------------------------
# Entry / Leaderboard
# ---------------------------------------------------------------------------


@dataclass
class LeaderboardEntry:
    """One analyst's score on the leaderboard."""

    model_id: str
    analyst_name: str
    run_at: datetime
    # Overall quality
    f1_pattern: float
    hit_rate_1r: float
    expected_r_mean: float
    side_accuracy: float = 0.0
    pattern_precision: float = 0.0
    pattern_recall: float = 0.0
    samples: int = 0
    # Cost
    avg_latency_ms: float = 0.0
    avg_input_tokens: float = 0.0
    avg_output_tokens: float = 0.0
    cache_hit_rate: float = 0.0
    cost_per_run_usd: float = 0.0
    # Bucket breakdowns
    by_regime: Dict[str, Dict[str, float]] = field(default_factory=dict)
    by_pattern: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Optional error string if this analyst failed to run
    error: Optional[str] = None

    def to_row(self) -> dict:
        d = asdict(self)
        d["run_at"] = self.run_at.isoformat()
        d["by_regime"] = json.dumps(self.by_regime)
        d["by_pattern"] = json.dumps(self.by_pattern)
        return d


class Leaderboard:
    """Run every configured analyst and produce a comparison report."""

    def __init__(
        self,
        config: LeaderboardConfig,
        factory: Optional[AnalystFactory] = None,
    ) -> None:
        self._config = config
        self._factory = factory or AnalystFactory()
        self._entries: List[LeaderboardEntry] = []
        self._reports: Dict[str, EvalReport] = {}

    @property
    def config(self) -> LeaderboardConfig:
        return self._config

    @property
    def entries(self) -> List[LeaderboardEntry]:
        return list(self._entries)

    async def run_all(self, dataset: GoldenDataset) -> List[LeaderboardEntry]:
        """Score every configured analyst; skip ones that fail to build/run."""
        if len(dataset) == 0:
            raise ValueError("Leaderboard.run_all: dataset is empty")

        self._entries.clear()
        self._reports.clear()

        for spec in self._config.analysts:
            entry = await self._run_one(spec, dataset)
            self._entries.append(entry)
        return list(self._entries)

    async def _run_one(self, spec: AnalystSpec, dataset: GoldenDataset) -> LeaderboardEntry:
        display = spec.display_name()
        now = datetime.now(timezone.utc)
        try:
            analyst = self._factory.build(spec)
        except Exception as exc:  # build failure — record but do not abort
            logger.warning("Leaderboard: failed to build %s: %s", display, exc)
            return LeaderboardEntry(
                model_id=display,
                analyst_name=display,
                run_at=now,
                f1_pattern=0.0,
                hit_rate_1r=0.0,
                expected_r_mean=0.0,
                error=f"build_failed: {type(exc).__name__}: {exc}",
            )
        try:
            runner = EvalRunner(
                analyst=analyst,
                dataset=dataset,
                max_concurrent=self._config.max_concurrent,
            )
            report = await runner.run()
        except Exception as exc:  # run failure — record and continue
            logger.warning("Leaderboard: %s failed during run: %s", display, exc)
            return LeaderboardEntry(
                model_id=spec.model or display,
                analyst_name=getattr(analyst, "name", display),
                run_at=now,
                f1_pattern=0.0,
                hit_rate_1r=0.0,
                expected_r_mean=0.0,
                error=f"run_failed: {type(exc).__name__}: {exc}",
            )

        self._reports[display] = report
        return self._entry_from_report(spec, analyst, report, now)

    def _entry_from_report(
        self,
        spec: AnalystSpec,
        analyst: Analyst,
        report: EvalReport,
        run_at: datetime,
    ) -> LeaderboardEntry:
        overall = report.overall()
        cost = report.cost_summary()
        samples = max(1, cost.get("samples", 0))
        avg_in = cost.get("total_input_tokens", 0) / samples
        avg_out = cost.get("total_output_tokens", 0) / samples
        model_key = spec.model or ""
        cost_per_run = _cost_per_run(model_key, avg_in, avg_out, self._config.cost_per_mtoken)

        def _bucket_dict(dim: str) -> Dict[str, Dict[str, float]]:
            return {
                b.value: {
                    "samples": b.samples,
                    "pattern_f1": b.pattern_f1,
                    "hit_rate_1r": b.hit_rate_1r,
                    "avg_realized_r": b.avg_realized_r,
                }
                for b in report.bucket_metrics(dim)
            }

        return LeaderboardEntry(
            model_id=spec.model or spec.display_name(),
            analyst_name=getattr(analyst, "name", spec.display_name()),
            run_at=run_at,
            f1_pattern=overall.pattern_f1,
            hit_rate_1r=overall.hit_rate_1r,
            expected_r_mean=overall.avg_realized_r,
            side_accuracy=overall.side_accuracy,
            pattern_precision=overall.pattern_precision,
            pattern_recall=overall.pattern_recall,
            samples=overall.samples,
            avg_latency_ms=cost.get("avg_latency_ms", 0.0),
            avg_input_tokens=avg_in,
            avg_output_tokens=avg_out,
            cache_hit_rate=cost.get("cache_hit_rate", 0.0),
            cost_per_run_usd=cost_per_run,
            by_regime=_bucket_dict("regime"),
            by_pattern=_bucket_dict("pattern"),
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def persist(self, db_path: Optional[Path] = None) -> Path:
        """Append current entries to the history parquet log.

        The log is append-only: each weekly run adds one row per analyst
        so :meth:`load_history` can produce trend charts over time.
        """
        if not self._entries:
            raise ValueError("Leaderboard.persist: no entries to persist; call run_all first")
        import pandas as pd

        path = Path(db_path) if db_path is not None else self._config.history_path
        path.parent.mkdir(parents=True, exist_ok=True)
        new_rows = pd.DataFrame([e.to_row() for e in self._entries])
        if path.exists():
            prior = pd.read_parquet(path)
            combined = pd.concat([prior, new_rows], ignore_index=True)
        else:
            combined = new_rows
        combined.to_parquet(path, index=False)
        return path

    @classmethod
    def load_history(cls, db_path: Path):
        """Return the full history as a ``pandas.DataFrame`` (empty if absent)."""
        import pandas as pd

        p = Path(db_path)
        if not p.exists():
            return pd.DataFrame()
        return pd.read_parquet(p)

    # ------------------------------------------------------------------
    # HTML rendering
    # ------------------------------------------------------------------

    def to_html(self, path: Optional[Path] = None) -> str:
        if not self._entries:
            raise ValueError("Leaderboard.to_html: no entries; call run_all first")

        sections: List[str] = []
        run_at = datetime.now(timezone.utc).isoformat()
        sections.append(_render_header(len(self._entries), run_at, self._config))
        sections.append(_render_overall_table(self._entries))
        pareto = _render_pareto(self._entries)
        if pareto:
            sections.append(pareto)
        sections.append(_render_bucket_section("By regime", "regime", self._entries, lambda e: e.by_regime))
        sections.append(_render_bucket_section("By pattern", "pattern", self._entries, lambda e: e.by_pattern))
        sections.append(_render_cost_table(self._entries))

        doc = _HTML_TEMPLATE.format(
            title=html.escape("Brooks Leaderboard"),
            body="\n".join(sections),
        )
        if path is not None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(doc, encoding="utf-8")
        return doc


# ---------------------------------------------------------------------------
# HTML rendering helpers
# ---------------------------------------------------------------------------


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 24px; color: #1f2933; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }}
h2 {{ font-size: 1.1rem; margin-top: 1.8rem; border-bottom: 1px solid #d9e2ec; padding-bottom: 4px; }}
table {{ border-collapse: collapse; margin: 0.5rem 0 1.5rem; font-size: 0.85rem; }}
th, td {{ border: 1px solid #cbd5e0; padding: 4px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; font-weight: 500; }}
tr:nth-child(even) {{ background: #f7fafc; }}
tr.err td {{ color: #b91c1c; }}
.muted {{ color: #627d98; font-size: 0.8rem; }}
.pareto img {{ max-width: 720px; border: 1px solid #d9e2ec; border-radius: 4px; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""


def _render_header(n: int, run_at: str, config: LeaderboardConfig) -> str:
    return (
        "<h1>Brooks Leaderboard</h1>"
        f"<p class='muted'>run at {html.escape(run_at)} · analysts: {n} · "
        f"dataset: {html.escape(str(config.dataset))}</p>"
    )


def _render_overall_table(entries: List[LeaderboardEntry]) -> str:
    headers = [
        "analyst",
        "samples",
        "F1",
        "precision",
        "recall",
        "side",
        "hit-1R",
        "avg R",
        "latency",
        "cost/run",
    ]
    rows = []
    ranked = sorted(entries, key=lambda e: (-(e.f1_pattern or 0.0), -(e.hit_rate_1r or 0.0)))
    for e in ranked:
        err_class = " class='err'" if e.error else ""
        rows.append(
            f"<tr{err_class}>"
            f"<td>{html.escape(e.analyst_name)}</td>"
            f"<td>{e.samples}</td>"
            f"<td>{_pct(e.f1_pattern)}</td>"
            f"<td>{_pct(e.pattern_precision)}</td>"
            f"<td>{_pct(e.pattern_recall)}</td>"
            f"<td>{_pct(e.side_accuracy)}</td>"
            f"<td>{_pct(e.hit_rate_1r)}</td>"
            f"<td>{e.expected_r_mean:.2f}</td>"
            f"<td>{e.avg_latency_ms:.1f} ms</td>"
            f"<td>${e.cost_per_run_usd:.5f}</td>"
            "</tr>"
        )
    body = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body += "<tbody>" + "\n".join(rows) + "</tbody>"
    errs = [e for e in entries if e.error]
    extra = ""
    if errs:
        extra = (
            "<p class='muted'>errors: "
            + ", ".join(f"{html.escape(e.analyst_name)} — {html.escape(e.error or '')}" for e in errs)
            + "</p>"
        )
    return f"<h2>Overall</h2><table>{body}</table>{extra}"


def _render_cost_table(entries: List[LeaderboardEntry]) -> str:
    headers = ["analyst", "avg in tokens", "avg out tokens", "cache hit", "latency", "cost/run"]
    rows = []
    for e in sorted(entries, key=lambda e: e.cost_per_run_usd):
        rows.append(
            "<tr>"
            f"<td>{html.escape(e.analyst_name)}</td>"
            f"<td>{e.avg_input_tokens:.0f}</td>"
            f"<td>{e.avg_output_tokens:.0f}</td>"
            f"<td>{_pct(e.cache_hit_rate)}</td>"
            f"<td>{e.avg_latency_ms:.1f} ms</td>"
            f"<td>${e.cost_per_run_usd:.5f}</td>"
            "</tr>"
        )
    body = "<thead><tr>" + "".join(f"<th>{h}</th>" for h in headers) + "</tr></thead>"
    body += "<tbody>" + "\n".join(rows) + "</tbody>"
    return f"<h2>Cost</h2><table>{body}</table>"


def _render_bucket_section(
    title: str,
    dim: str,
    entries: List[LeaderboardEntry],
    getter: Callable[[LeaderboardEntry], Dict[str, Dict[str, float]]],
) -> str:
    values = sorted({v for e in entries for v in getter(e).keys()})
    if not values:
        return f"<h2>{html.escape(title)}</h2><p class='muted'>no data</p>"
    header = "<tr><th>analyst</th>" + "".join(f"<th>{html.escape(v)}</th>" for v in values) + "</tr>"
    rows = []
    for e in entries:
        cells = []
        bd = getter(e)
        for v in values:
            m = bd.get(v)
            if m is None:
                cells.append("<td>—</td>")
            else:
                cells.append(f"<td>{_pct(m.get('pattern_f1', 0.0))}</td>")
        rows.append(f"<tr><td>{html.escape(e.analyst_name)}</td>{''.join(cells)}</tr>")
    body = f"<thead>{header}</thead><tbody>{''.join(rows)}</tbody>"
    return f"<h2>{html.escape(title)} — F1 by bucket</h2><table>{body}</table>"


def _render_pareto(entries: List[LeaderboardEntry]) -> str:
    """Scatter plot of F1 vs cost-per-run; base64 PNG inline."""
    points = [(e.cost_per_run_usd, e.f1_pattern, e.analyst_name) for e in entries if e.error is None]
    if not points:
        return ""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib is a hard dep, but don't break HTML
        logger.warning("leaderboard: matplotlib unavailable (%s), skipping Pareto chart", exc)
        return ""

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.scatter(xs, ys, s=60, c="#2b6cb0", edgecolors="#1a365d")
    for x, y, name in points:
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Cost per run (USD)")
    ax.set_ylabel("Pattern F1")
    ax.set_title("Quality / Cost Pareto")
    ax.grid(True, linestyle="--", alpha=0.4)
    if any(x > 0 for x in xs):
        ax.set_xscale("symlog", linthresh=max(1e-6, min(x for x in xs if x > 0) / 2))

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=140)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return (
        f"<h2>Pareto (quality vs cost)</h2><p class='pareto'><img src='data:image/png;base64,{b64}' alt='pareto'></p>"
    )


def _pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    return f"{x * 100:.1f}%"
