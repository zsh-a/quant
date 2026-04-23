"""Schema + generator tests for the Brooks taxonomy and manual renderer."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import scripts.brooks_render_manual as render_mod
from scripts.brooks_render_manual import (
    REQUIRED_SECTIONS,
    TOKEN_BUDGET,
    Taxonomy,
    count_tokens,
    load_sections,
    load_taxonomy,
    render,
)


@pytest.fixture(scope="module")
def taxonomy() -> Taxonomy:
    return load_taxonomy()


@pytest.fixture(scope="module")
def sections() -> dict[str, str]:
    return load_sections()


@pytest.fixture(scope="module")
def raw_taxonomy() -> dict:
    return yaml.safe_load(render_mod.TAXONOMY_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_taxonomy_loads_and_validates(taxonomy: Taxonomy) -> None:
    assert taxonomy.version == 1
    assert len(taxonomy.bar_types) >= 11
    assert len(taxonomy.patterns) >= 21
    assert len(taxonomy.regimes) >= 7


def test_required_bar_types_present(taxonomy: Taxonomy) -> None:
    names = {b.name for b in taxonomy.bar_types}
    expected = {
        "trend_bull",
        "trend_bear",
        "doji_bull",
        "doji_bear",
        "signal_bar",
        "climactic_bar",
        "reversal_bar",
        "ii",
        "iii",
        "outside_bar",
        "shaved_bar",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


def test_required_patterns_present(taxonomy: Taxonomy) -> None:
    names = {p.name for p in taxonomy.patterns}
    expected = {
        "h1",
        "h2",
        "h3",
        "h4",
        "l1",
        "l2",
        "l3",
        "l4",
        "ii_breakout",
        "iii_breakout",
        "two_bar_reversal",
        "wedge",
        "double_top",
        "double_bottom",
        "final_flag",
        "micro_channel",
        "breakout",
        "breakout_pullback",
        "failed_breakout",
        "mtr",
        "measured_move",
    }
    assert expected.issubset(names), f"missing: {expected - names}"


def test_seven_regimes(taxonomy: Taxonomy) -> None:
    assert len(taxonomy.regimes) >= 7


def test_pattern_probabilities_in_range(taxonomy: Taxonomy) -> None:
    for p in taxonomy.patterns:
        assert 0.0 <= p.typical_probability <= 1.0


def test_trader_equation_buckets_sorted(taxonomy: Taxonomy) -> None:
    buckets = [b.bucket for b in taxonomy.trader_equation.probability_buckets]
    assert buckets == sorted(buckets), "probability buckets must be ascending"
    assert buckets[0] >= 0.0 and buckets[-1] <= 1.0
    assert {0.40, 0.55, 0.65, 0.75, 0.85}.issubset(set(buckets))


def test_always_in_rules_present(taxonomy: Taxonomy) -> None:
    rules = taxonomy.always_in_rules
    assert rules.flip_to_long_when
    assert rules.flip_to_short_when
    assert rules.stay_neutral_when
    assert rules.hysteresis


def test_no_duplicate_names(taxonomy: Taxonomy) -> None:
    for label, items in (
        ("bar_types", [b.name for b in taxonomy.bar_types]),
        ("patterns", [p.name for p in taxonomy.patterns]),
        ("regimes", [r.name for r in taxonomy.regimes]),
    ):
        assert len(items) == len(set(items)), f"duplicate name in {label}"


def test_extra_fields_rejected(raw_taxonomy: dict) -> None:
    bad = copy.deepcopy(raw_taxonomy)
    bad["bar_types"][0]["unexpected_key"] = "x"
    with pytest.raises(ValidationError):
        Taxonomy.model_validate(bad)


def test_out_of_range_probability_rejected(raw_taxonomy: dict) -> None:
    bad = copy.deepcopy(raw_taxonomy)
    bad["patterns"][0]["typical_probability"] = 1.7
    with pytest.raises(ValidationError):
        Taxonomy.model_validate(bad)


def test_missing_required_pattern_rejected(raw_taxonomy: dict) -> None:
    bad = copy.deepcopy(raw_taxonomy)
    bad["patterns"] = [p for p in bad["patterns"] if p["name"] != "h2"]
    with pytest.raises(ValidationError):
        Taxonomy.model_validate(bad)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def test_required_sections_present(sections: dict[str, str]) -> None:
    for name in REQUIRED_SECTIONS:
        assert name in sections
        assert sections[name].strip(), f"section {name} is empty"


# ---------------------------------------------------------------------------
# Renderer — determinism, idempotence, token budget
# ---------------------------------------------------------------------------


def test_render_is_deterministic(taxonomy: Taxonomy, sections: dict[str, str]) -> None:
    a = render(taxonomy, sections)
    b = render(taxonomy, sections)
    assert a == b


def test_render_is_idempotent_on_disk_via_check(tmp_path: Path) -> None:
    # The committed manual must match what the renderer produces *now*.
    rc = render_mod.main(["--check"])
    assert rc == 0, "concept_manual.md is stale; re-run brooks_render_manual.py"


def test_token_count_under_budget(taxonomy: Taxonomy, sections: dict[str, str]) -> None:
    text = render(taxonomy, sections)
    n_tokens, encoder = count_tokens(text)
    assert n_tokens <= TOKEN_BUDGET, f"manual is {n_tokens} tokens (encoder={encoder}); budget={TOKEN_BUDGET}"


def test_render_includes_all_pattern_names(taxonomy: Taxonomy, sections: dict[str, str]) -> None:
    text = render(taxonomy, sections)
    for p in taxonomy.patterns:
        assert f"`{p.name}`" in text, f"pattern {p.name} missing from manual"


def test_render_includes_all_bar_type_names(taxonomy: Taxonomy, sections: dict[str, str]) -> None:
    text = render(taxonomy, sections)
    for b in taxonomy.bar_types:
        assert f"`{b.name}`" in text


def test_render_includes_all_regime_names(taxonomy: Taxonomy, sections: dict[str, str]) -> None:
    text = render(taxonomy, sections)
    for r in taxonomy.regimes:
        assert f"`{r.name}`" in text


def test_render_unaffected_by_pattern_dict_key_order(raw_taxonomy: dict, sections: dict[str, str]) -> None:
    # Build two taxonomies whose dict keys are inserted in different orders
    # (json roundtrip then re-validate). Output must be identical.
    import json

    forward = Taxonomy.model_validate(raw_taxonomy)
    reordered = json.loads(json.dumps(raw_taxonomy, sort_keys=True))
    backward = Taxonomy.model_validate(reordered)
    assert render(forward, sections) == render(backward, sections)


def test_check_mode_detects_drift(tmp_path: Path) -> None:
    out = tmp_path / "manual.md"
    out.write_text("stale content\n", encoding="utf-8")
    rc = render_mod.main(["--check", "--output", str(out)])
    assert rc == 1
