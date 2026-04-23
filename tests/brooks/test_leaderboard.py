"""Tests for the cross-model leaderboard (Phase 4.5)."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import pandas as pd
import pytest

from src.alpha.llm.cache import CacheSpec
from src.alpha.llm.provider import Message, Response, Usage
from src.brooks.analyst.base import Analyst, AnalystRegistry
from src.brooks.context import Bar, BrooksContext
from src.brooks.eval.golden import GoldenDataset, GoldenSample
from src.brooks.eval.leaderboard import (
    AnalystFactory,
    AnalystSpec,
    Leaderboard,
    LeaderboardConfig,
    _cost_per_run,
)
from src.brooks.prompts import PromptBundle
from src.brooks.schema import Signal

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _bar(i: int, close: float = 100.0) -> Bar:
    return Bar(
        timestamp_ns=1_700_000_000_000_000_000 + i * 60_000_000_000,
        open=close,
        high=close + 0.5,
        low=close - 0.5,
        close=close,
        volume=1.0,
    )


def _sample(
    sample_id: str,
    *,
    regime: str = "weak_bull_trend",
    expected_pattern: str = "h2",
    entry: float = 100.0,
    stop: float = 99.0,
    future_hit_2r: bool = False,
) -> GoldenSample:
    ctx = [_bar(i) for i in range(5)]
    if future_hit_2r:
        future = [_bar(5, close=100.2), _bar(6, close=102.5)]
    else:
        future = [_bar(5, close=100.2), _bar(6, close=100.4)]
    future = [
        Bar(
            timestamp_ns=b.timestamp_ns,
            open=b.open,
            high=(102.5 if i == 1 and future_hit_2r else b.close + 0.3),
            low=b.low,
            close=b.close,
            volume=b.volume,
        )
        for i, b in enumerate(future)
    ]
    return GoldenSample(
        id=sample_id,
        symbol=sample_id,
        interval="5m",
        bars=ctx + future,
        target_bar_idx=4,
        expected_pattern=expected_pattern,
        expected_side="long",
        expected_entry=entry,
        expected_stop=stop,
        expected_target=102.0,
        regime=regime,
        htf_aligned=True,
        source="human",
    )


@pytest.fixture
def small_dataset() -> GoldenDataset:
    return GoldenDataset.from_samples(
        [
            _sample("one", regime="weak_bull_trend", future_hit_2r=True),
            _sample("two", regime="strong_bull_trend", future_hit_2r=True),
            _sample("three", regime="weak_bull_trend", expected_pattern="wedge"),
        ]
    )


@pytest.fixture
def stub_prompts() -> PromptBundle:
    return PromptBundle(
        system_text="SYSTEM",
        concept_manual="MANUAL",
        fewshot=[],
        schema_description="SCHEMA",
    )


@dataclass
class _RecordingMockProvider:
    """In-process Provider that returns a deterministic ``LLMSignalBatch``."""

    model: str = "mock-model"
    signals: List[Signal] = field(default_factory=list)
    usage: Usage = field(default_factory=lambda: Usage(input_tokens=80, output_tokens=30))
    latency_ms: float = 9.0
    name: str = "mock"
    cache_hit: bool = False
    call_count: int = 0

    async def complete(
        self,
        messages: list[Message],
        schema: type,
        cache: Optional[CacheSpec] = None,
        seed: Optional[int] = None,
        max_tokens: int = 4096,
        temperature: float = 0.0,
    ) -> Response[Any]:
        self.call_count += 1
        parsed = schema(reasoning="r", signals=list(self.signals))
        return Response(
            parsed=parsed,
            raw={"mock": True},
            usage=self.usage,
            latency_ms=self.latency_ms,
            model=self.model,
            cache_hit=self.cache_hit,
        )


def _mock_signal() -> Signal:
    return Signal(
        pattern="h2",
        side="long",
        signal_bar_idx=4,
        entry_px=100.0,
        stop_px=99.0,
        target_px=102.0,
        probability=0.55,
        quality=0.6,
        source="llm:placeholder",
    )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# LeaderboardConfig loading
# ---------------------------------------------------------------------------


def test_config_from_dict_parses_analyst_list(tmp_path):
    raw = {
        "dataset": str(tmp_path / "ds.parquet"),
        "analysts": [
            {"type": "rule"},
            {"type": "llm", "model": "claude-opus-4-7"},
            {"type": "ensemble.critic", "producer": "rule", "critic": "llm:claude-sonnet-4-6"},
        ],
        "output_dir": str(tmp_path / "out"),
        "history_path": str(tmp_path / "hist.parquet"),
        "max_concurrent": 2,
    }
    cfg = LeaderboardConfig.from_dict(raw)
    assert cfg.dataset == Path(str(tmp_path / "ds.parquet"))
    assert cfg.output_dir == Path(str(tmp_path / "out"))
    assert cfg.max_concurrent == 2
    assert [a.type for a in cfg.analysts] == ["rule", "llm", "ensemble.critic"]
    critic_spec = cfg.analysts[2]
    assert critic_spec.params["producer"] == "rule"
    assert critic_spec.params["critic"] == "llm:claude-sonnet-4-6"


def test_config_raises_when_dataset_missing():
    with pytest.raises(ValueError):
        LeaderboardConfig.from_dict({"analysts": [{"type": "rule"}]})


def test_config_raises_when_no_analysts():
    with pytest.raises(ValueError):
        LeaderboardConfig.from_dict({"dataset": "x", "analysts": []})


def test_config_load_reads_yaml_file(tmp_path):
    yaml_text = "dataset: data/ds.parquet\nanalysts:\n  - {type: rule}\n  - {type: llm, model: claude-opus-4-7}\n"
    p = tmp_path / "leaderboard.yaml"
    p.write_text(yaml_text)
    cfg = LeaderboardConfig.load(p)
    assert len(cfg.analysts) == 2
    assert cfg.analysts[1].model == "claude-opus-4-7"


# ---------------------------------------------------------------------------
# AnalystFactory
# ---------------------------------------------------------------------------


def test_factory_builds_rule_analyst():
    factory = AnalystFactory()
    analyst = factory.build(AnalystSpec(type="rule"))
    assert analyst.name == "rule"


def test_factory_builds_llm_analyst_via_provider_factory(stub_prompts):
    created: List[str] = []

    def provider_factory(model: str):
        created.append(model)
        return _RecordingMockProvider(model=model)

    factory = AnalystFactory(provider_factory=provider_factory)
    spec = AnalystSpec(type="llm", model="claude-opus-4-7", params={"prompts": stub_prompts})
    analyst = factory.build(spec)
    assert analyst.name == "llm:claude-opus-4-7"
    assert created == ["claude-opus-4-7"]


def test_factory_llm_without_provider_factory_raises():
    factory = AnalystFactory()
    with pytest.raises(RuntimeError):
        factory.build(AnalystSpec(type="llm", model="claude-opus-4-7"))


def test_factory_builds_ensemble_critic(stub_prompts):
    def provider_factory(model: str):
        return _RecordingMockProvider(model=model)

    factory = AnalystFactory(provider_factory=provider_factory)
    spec = AnalystSpec(
        type="ensemble.critic",
        params={
            "producer": "rule",
            "critic": {"type": "llm", "model": "claude-sonnet-4-6", "prompts": stub_prompts},
        },
    )
    analyst = factory.build(spec)
    assert analyst.name == "ensemble.critic"


def test_factory_vlm_without_registration_raises():
    # VLM analyst isn't registered in this codebase yet (Phase 4.3 dep).
    # The factory should surface that clearly rather than silently fall back.
    assert "vlm" not in AnalystRegistry.all()

    def pf(_m):
        return _RecordingMockProvider()

    factory = AnalystFactory(provider_factory=pf)
    with pytest.raises(KeyError):
        factory.build(AnalystSpec(type="vlm", model="gemini-2.5-pro"))


def test_factory_unknown_type_raises():
    factory = AnalystFactory()
    with pytest.raises(ValueError):
        factory.build(AnalystSpec(type="quantum-magic"))


# ---------------------------------------------------------------------------
# Fixed analyst for deterministic Leaderboard runs
# ---------------------------------------------------------------------------


@dataclass
class _FixedAnalyst:
    name: str
    output: List[Signal]

    async def analyze(self, ctx: BrooksContext) -> List[Signal]:
        return list(self.output)


class _FixedFactory(AnalystFactory):
    """Bypass registry — map AnalystSpec by ``display_name`` to a fixed analyst."""

    def __init__(self, mapping: dict[str, Analyst]) -> None:
        super().__init__()
        self._mapping = mapping

    def build(self, spec: AnalystSpec) -> Analyst:
        key = spec.display_name()
        if key not in self._mapping:
            raise KeyError(f"no fixed analyst for {key}")
        return self._mapping[key]


# ---------------------------------------------------------------------------
# Leaderboard.run_all / metrics
# ---------------------------------------------------------------------------


def test_run_all_produces_entries_per_analyst(small_dataset):
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[
            AnalystSpec(type="rule", label="A"),
            AnalystSpec(type="llm", model="mock-m", label="B"),
        ],
    )
    fixed = _FixedFactory(
        {
            "A": _FixedAnalyst(name="A", output=[_mock_signal()]),
            "B": _FixedAnalyst(name="B", output=[]),
        }
    )
    board = Leaderboard(cfg, factory=fixed)

    entries = _run(board.run_all(small_dataset))
    assert len(entries) == 2
    by_name = {e.analyst_name: e for e in entries}
    assert by_name["A"].samples == len(small_dataset)
    assert by_name["A"].f1_pattern > 0  # at least one pattern match
    assert by_name["B"].f1_pattern == 0  # analyst emits nothing
    # regime bucket key must be populated from the dataset
    assert "weak_bull_trend" in by_name["A"].by_regime
    assert "strong_bull_trend" in by_name["A"].by_regime


def test_run_all_captures_build_errors_without_aborting(small_dataset):
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[
            AnalystSpec(type="rule", label="A"),
            AnalystSpec(type="llm", model="needs-provider", label="missing"),
        ],
    )
    fixed = _FixedFactory({"A": _FixedAnalyst(name="A", output=[])})
    # "missing" is not in the fixed map → raises KeyError → recorded as error.
    board = Leaderboard(cfg, factory=fixed)

    entries = _run(board.run_all(small_dataset))
    assert len(entries) == 2
    err_entry = next(e for e in entries if e.analyst_name == "missing")
    assert err_entry.error is not None
    assert err_entry.f1_pattern == 0.0


def test_run_all_empty_dataset_raises():
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[AnalystSpec(type="rule", label="only")],
    )
    board = Leaderboard(cfg, factory=_FixedFactory({"only": _FixedAnalyst(name="only", output=[])}))
    with pytest.raises(ValueError):
        _run(board.run_all(GoldenDataset.from_samples([])))


# ---------------------------------------------------------------------------
# Cost calculation
# ---------------------------------------------------------------------------


def test_cost_per_run_uses_model_rates():
    table = {"claude-opus-4-7": {"input": 15.0, "output": 75.0}}
    # 1000 input + 500 output → 1000/1e6 * 15 + 500/1e6 * 75 = 0.015 + 0.0375 = 0.0525
    cost = _cost_per_run("claude-opus-4-7", 1000, 500, table)
    assert cost == pytest.approx(0.0525)


def test_cost_per_run_unknown_model_returns_zero():
    assert _cost_per_run("unknown-model", 1000, 500, {}) == 0.0


def test_entry_includes_cost_for_known_model(small_dataset, stub_prompts):
    from src.brooks.analyst.llm import LLMAnalyst

    provider = _RecordingMockProvider(model="claude-opus-4-7")
    analyst = LLMAnalyst(provider=provider, model="claude-opus-4-7", prompts=stub_prompts)
    # Wire the LLMAnalyst to emit a matching signal so pattern_match is True.
    # We override via a thin wrapper so the MockProvider's token usage drives
    # the cost calculation.
    provider.signals = [_mock_signal()]

    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[AnalystSpec(type="llm", model="claude-opus-4-7")],
    )
    fixed = _FixedFactory({"llm:claude-opus-4-7": analyst})
    board = Leaderboard(cfg, factory=fixed)
    entries = _run(board.run_all(small_dataset))
    entry = entries[0]
    # avg_input_tokens should be set; cost > 0 because the default table knows the model.
    assert entry.avg_input_tokens > 0
    assert entry.cost_per_run_usd > 0


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


def test_to_html_writes_self_contained_document(small_dataset, tmp_path):
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[AnalystSpec(type="rule", label="A")],
    )
    board = Leaderboard(cfg, factory=_FixedFactory({"A": _FixedAnalyst(name="A", output=[_mock_signal()])}))
    _run(board.run_all(small_dataset))

    out = tmp_path / "lb.html"
    doc = board.to_html(out)
    assert out.exists()
    assert "<html" in doc
    assert "Brooks Leaderboard" in doc
    assert "By regime" in doc or "regime" in doc.lower()


def test_to_html_before_run_raises():
    cfg = LeaderboardConfig(dataset=Path("x"), analysts=[AnalystSpec(type="rule")])
    board = Leaderboard(cfg)
    with pytest.raises(ValueError):
        board.to_html()


# ---------------------------------------------------------------------------
# Persistence / history
# ---------------------------------------------------------------------------


def test_persist_appends_and_load_history_reads_rows(small_dataset, tmp_path):
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[AnalystSpec(type="rule", label="A"), AnalystSpec(type="rule", label="B")],
        history_path=tmp_path / "hist.parquet",
    )
    factory = _FixedFactory(
        {
            "A": _FixedAnalyst(name="A", output=[_mock_signal()]),
            "B": _FixedAnalyst(name="B", output=[]),
        }
    )

    # First run → creates the file with two rows
    board1 = Leaderboard(cfg, factory=factory)
    _run(board1.run_all(small_dataset))
    path = board1.persist()
    assert path.exists()

    df = Leaderboard.load_history(path)
    assert len(df) == 2
    assert set(df["analyst_name"]) == {"A", "B"}

    # Second run → appends another two rows
    board2 = Leaderboard(cfg, factory=factory)
    _run(board2.run_all(small_dataset))
    board2.persist()
    df2 = Leaderboard.load_history(path)
    assert len(df2) == 4


def test_persist_before_run_raises(tmp_path):
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[AnalystSpec(type="rule")],
        history_path=tmp_path / "hist.parquet",
    )
    board = Leaderboard(cfg)
    with pytest.raises(ValueError):
        board.persist()


def test_load_history_missing_file_returns_empty(tmp_path):
    df = Leaderboard.load_history(tmp_path / "does-not-exist.parquet")
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_entry_to_row_encodes_buckets_as_json_strings(small_dataset):
    cfg = LeaderboardConfig(
        dataset=Path("unused"),
        analysts=[AnalystSpec(type="rule", label="A")],
    )
    factory = _FixedFactory({"A": _FixedAnalyst(name="A", output=[_mock_signal()])})
    board = Leaderboard(cfg, factory=factory)
    entry = _run(board.run_all(small_dataset))[0]
    row = entry.to_row()
    # Buckets must be JSON strings so parquet can store them as scalars.
    assert isinstance(row["by_regime"], str)
    parsed = json.loads(row["by_regime"])
    assert "weak_bull_trend" in parsed
    assert isinstance(row["run_at"], str)  # datetime.isoformat()


# ---------------------------------------------------------------------------
# CLI smoke test (mock-llm path)
# ---------------------------------------------------------------------------


def test_cli_mock_llm_runs_end_to_end(tmp_path, small_dataset):
    """Smoke test for ``scripts/brooks_leaderboard.py --mock-llm``.

    Uses a rule analyst (no provider needed) to keep the test self-contained
    while still exercising the config loader + HTML writer + persistence.
    """
    dataset_path = tmp_path / "ds.parquet"
    small_dataset.save(dataset_path)

    cfg_path = tmp_path / "leaderboard.yaml"
    cfg_path.write_text(
        "dataset: {ds}\nanalysts:\n  - {{type: rule}}\noutput_dir: {out}\nhistory_path: {hist}\n".format(
            ds=dataset_path,
            out=tmp_path / "out",
            hist=tmp_path / "hist.parquet",
        )
    )

    from scripts.brooks_leaderboard import main  # type: ignore

    out_html = tmp_path / "run.html"
    rc = main(["--config", str(cfg_path), "--output", str(out_html), "--mock-llm"])
    assert rc == 0
    assert out_html.exists()
    assert (tmp_path / "hist.parquet").exists()
