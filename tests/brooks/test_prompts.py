"""Tests for src/brooks/prompts.py — PromptBundle asset loader."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.alpha.llm.provider import Message, TextPart
from src.brooks.prompts import (
    DEFAULT_PROMPT_DIR,
    PromptBundle,
    build_schema_description,
)
from src.brooks.schema import Decision


# ---------------------------------------------------------------------------
# Schema description — generated from Pydantic, never hand-written
# ---------------------------------------------------------------------------


def test_schema_description_contains_decision_fields() -> None:
    text = build_schema_description(Decision)
    for field_name in [
        "symbol",
        "side",
        "entry_px",
        "stop_px",
        "target_px",
        "quantity",
        "probability",
        "expected_r",
        "regime",
        "htf_aligned",
        "signals",
        "source",
        "reasoning",
    ]:
        assert field_name in text, f"missing field in schema: {field_name}"
    assert "```json" in text
    assert "Output schema" in text


def test_schema_description_is_valid_json_block() -> None:
    text = build_schema_description(Decision)
    start = text.index("```json") + len("```json")
    end = text.rindex("```")
    payload = text[start:end].strip()
    parsed = json.loads(payload)
    assert parsed.get("title") == "Decision"
    assert "properties" in parsed


# ---------------------------------------------------------------------------
# Real-asset loading from prompts/brooks
# ---------------------------------------------------------------------------


def test_load_from_default_dir_succeeds() -> None:
    bundle = PromptBundle.load(DEFAULT_PROMPT_DIR)
    assert bundle.system_text.strip().startswith("#")
    assert "Brooks" in bundle.system_text
    assert "Brooks" in bundle.concept_manual
    assert bundle.schema_description.startswith("## Output schema")


def test_system_prompt_has_no_embedded_json_schema() -> None:
    """Acceptance criterion: system_analyst.md must not bake in a JSON schema."""
    bundle = PromptBundle.load(DEFAULT_PROMPT_DIR)
    text = bundle.system_text
    assert "```json" not in text
    assert '"$defs"' not in text
    assert '"properties"' not in text


def test_fewshot_examples_are_loaded() -> None:
    bundle = PromptBundle.load(DEFAULT_PROMPT_DIR)
    assert len(bundle.fewshot) >= 5
    for ex in bundle.fewshot:
        assert isinstance(ex, dict)
        assert "user" in ex and isinstance(ex["user"], str)
        assert "assistant" in ex and isinstance(ex["assistant"], dict)
        a = ex["assistant"]
        for k in ("symbol", "side", "entry_px", "stop_px", "regime"):
            assert k in a, f"few-shot example missing key: {k}"


# ---------------------------------------------------------------------------
# build_messages — message shape, cache markers
# ---------------------------------------------------------------------------


def test_build_messages_default_returns_three_messages() -> None:
    """Acceptance criterion: build_messages returns 3 messages, the
    first one carries the cache marker on concept_manual, the second
    carries the few-shot cache marker."""
    bundle = PromptBundle.load(DEFAULT_PROMPT_DIR)
    msgs = bundle.build_messages(
        user_context="== LTF 5m ==\n#0 bull body=62% close=hi"
    )
    assert len(msgs) == 3
    system_msg, fewshot_msg, live_msg = msgs

    # Message 1: system prompt — contains system_text, schema, and
    # concept_manual with the cache="concept_manual" marker.
    assert isinstance(system_msg, Message)
    assert system_msg.role == "system"
    assert all(isinstance(p, TextPart) for p in system_msg.content)
    assert any("Brooks" in p.text for p in system_msg.content)
    cache_markers = [p.cache for p in system_msg.content]
    assert "concept_manual" in cache_markers

    # Message 2: few-shot user message with cache="fewshot".
    assert fewshot_msg.role == "user"
    assert len(fewshot_msg.content) == 1
    assert fewshot_msg.content[0].cache == "fewshot"
    assert "Example 1" in fewshot_msg.content[0].text

    # Message 3: live context (no cache marker).
    assert live_msg.role == "user"
    assert len(live_msg.content) == 1
    assert isinstance(live_msg.content[0], TextPart)
    assert live_msg.content[0].cache is None
    assert "LTF" in live_msg.content[0].text


def test_build_messages_first_message_carries_concept_manual_cache() -> None:
    """Acceptance criterion: first message carries the cache marker on
    the concept manual TextPart — providers can cache everything up to
    and including that block."""
    bundle = PromptBundle.load(DEFAULT_PROMPT_DIR)
    msgs = bundle.build_messages(user_context="ctx")
    system_parts = msgs[0].content
    # concept_manual is the cached part and must appear after system_text
    # so the cached prefix covers system + schema + manual.
    last_cached = next(
        (p for p in reversed(system_parts) if p.cache == "concept_manual"), None
    )
    assert last_cached is not None
    assert "Brooks" in last_cached.text
    # Schema description lives in the system message (not hand-baked).
    assert any("Output schema" in p.text for p in system_parts)


def test_build_messages_without_fewshot_returns_two_messages() -> None:
    bundle = PromptBundle.load(DEFAULT_PROMPT_DIR)
    msgs = bundle.build_messages(user_context="ctx", include_fewshot=False)
    assert len(msgs) == 2
    system_msg, live_msg = msgs
    assert system_msg.role == "system"
    # concept_manual cache marker is still present — the stable prefix
    # should be cacheable regardless of whether few-shot is included.
    assert any(p.cache == "concept_manual" for p in system_msg.content)
    assert live_msg.role == "user"


def test_build_messages_with_empty_fewshot_skips_fewshot_message() -> None:
    """A bundle with no few-shot examples never produces the
    fewshot-tagged user message, even when include_fewshot=True."""
    bundle = PromptBundle(
        system_text="SYSTEM",
        concept_manual="MANUAL",
        fewshot=[],
        schema_description="SCHEMA",
    )
    msgs = bundle.build_messages(user_context="ctx", include_fewshot=True)
    assert len(msgs) == 2
    assert msgs[0].role == "system"
    assert not any(
        part.cache == "fewshot" for msg in msgs for part in msg.content
    )


# ---------------------------------------------------------------------------
# Isolated loader paths (synthetic directory)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_prompt_dir(tmp_path: Path) -> Path:
    root = tmp_path / "prompts" / "brooks"
    (root / "fewshot").mkdir(parents=True)
    (root / "system_analyst.md").write_text("# TEST SYSTEM\nBody.\n", encoding="utf-8")
    (root / "concept_manual.md").write_text("# MANUAL\n", encoding="utf-8")
    lines = [
        json.dumps({"user": "CTX1", "assistant": {"side": "long"}}),
        "",  # blank line tolerated
        json.dumps({"user": "CTX2", "assistant": {"side": "short"}}),
    ]
    (root / "fewshot" / "analyst_examples.jsonl").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return root


def test_load_synthetic_dir_roundtrips(synthetic_prompt_dir: Path) -> None:
    bundle = PromptBundle.load(synthetic_prompt_dir)
    assert "TEST SYSTEM" in bundle.system_text
    assert "MANUAL" in bundle.concept_manual
    assert len(bundle.fewshot) == 2
    assert bundle.fewshot[0]["user"] == "CTX1"


def test_load_rejects_malformed_jsonl(tmp_path: Path) -> None:
    root = tmp_path / "prompts" / "brooks"
    (root / "fewshot").mkdir(parents=True)
    (root / "system_analyst.md").write_text("x", encoding="utf-8")
    (root / "concept_manual.md").write_text("y", encoding="utf-8")
    (root / "fewshot" / "analyst_examples.jsonl").write_text(
        "{not valid json\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="invalid JSON"):
        PromptBundle.load(root)


def test_load_with_missing_fewshot_file_is_fine(tmp_path: Path) -> None:
    root = tmp_path / "prompts" / "brooks"
    root.mkdir(parents=True)
    (root / "system_analyst.md").write_text("# sys", encoding="utf-8")
    (root / "concept_manual.md").write_text("# man", encoding="utf-8")
    bundle = PromptBundle.load(root)
    assert bundle.fewshot == []


def test_build_messages_fewshot_transcript_includes_json(
    synthetic_prompt_dir: Path,
) -> None:
    bundle = PromptBundle.load(synthetic_prompt_dir)
    msgs = bundle.build_messages(user_context="live")
    fewshot_text = msgs[1].content[0].text
    assert "USER:" in fewshot_text
    assert "ASSISTANT:" in fewshot_text
    assert "CTX1" in fewshot_text
    assert '"side": "long"' in fewshot_text or '"side":"long"' in fewshot_text
