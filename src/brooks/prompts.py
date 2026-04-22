"""Versioned prompt assets for the Brooks analyst.

Loads the Brooks system prompt, concept manual, and few-shot examples from
``prompts/brooks/`` and assembles them into a provider-neutral ``list[Message]``.

The output schema is derived from :class:`src.brooks.schema.Decision` via
Pydantic — it is never hand-written into the prompt files, so schema drift
can't silently desync from the runtime model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

from pydantic import BaseModel

from src.alpha.llm.provider import Message, TextPart
from src.brooks.schema import Decision

__all__ = ["PromptBundle", "build_schema_description", "DEFAULT_PROMPT_DIR"]


DEFAULT_PROMPT_DIR = Path("prompts/brooks")

_SYSTEM_FILE = "system_analyst.md"
_CONCEPT_FILE = "concept_manual.md"
_FEWSHOT_FILE = Path("fewshot") / "analyst_examples.jsonl"

_SCHEMA_HEADING = "## Output schema (authoritative)"


def build_schema_description(model: Type[BaseModel] = Decision) -> str:
    """Render a Pydantic model's JSON schema as a prompt-ready string.

    The output is a single markdown block. It's a *static* byproduct of
    the Pydantic model so schema updates flow from code, not from hand
    edits to the prompt markdown.
    """
    schema_dict = model.model_json_schema()
    schema_json = json.dumps(schema_dict, indent=2, ensure_ascii=False)
    return f"{_SCHEMA_HEADING}\n\n```json\n{schema_json}\n```\n"


@dataclass
class PromptBundle:
    """Bundle of versioned Brooks prompt assets."""

    system_text: str
    concept_manual: str
    fewshot: list[dict] = field(default_factory=list)
    schema_description: str = ""

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def load(
        cls,
        dir: Path | str = DEFAULT_PROMPT_DIR,
        *,
        schema_model: Type[BaseModel] = Decision,
    ) -> "PromptBundle":
        """Load the bundle from ``dir``.

        ``dir`` must contain ``system_analyst.md`` and
        ``concept_manual.md``; ``fewshot/analyst_examples.jsonl`` is
        optional (absent or empty → empty few-shot list, tolerated for
        unit tests).
        """
        root = Path(dir)
        system_text = (root / _SYSTEM_FILE).read_text(encoding="utf-8")
        concept_manual = (root / _CONCEPT_FILE).read_text(encoding="utf-8")

        fewshot: list[dict] = []
        fewshot_path = root / _FEWSHOT_FILE
        if fewshot_path.exists():
            fewshot = _load_jsonl(fewshot_path)

        return cls(
            system_text=system_text,
            concept_manual=concept_manual,
            fewshot=fewshot,
            schema_description=build_schema_description(schema_model),
        )

    # ------------------------------------------------------------------
    # Message assembly
    # ------------------------------------------------------------------

    def build_messages(
        self,
        user_context: str,
        include_fewshot: bool = True,
    ) -> list[Message]:
        """Assemble the provider-facing message list.

        Shape (matches the task-spec signature):

        * ``messages[0]`` — ``system`` message with three TextParts:
          ``system_text``, the Pydantic-derived ``schema_description``,
          and the ``concept_manual`` (last part carries
          ``cache="concept_manual"`` so providers can cache the whole
          stable prefix).
        * ``messages[1]`` — ``user`` message containing the few-shot
          transcript (``cache="fewshot"``). Emitted only when
          ``include_fewshot`` is ``True`` *and* examples are loaded.
        * ``messages[-1]`` — ``user`` message with the live
          ``user_context`` (no cache marker; this changes every call).
        """
        system_parts = [
            TextPart(text=self.system_text),
            TextPart(text="\n" + self.schema_description),
            TextPart(text="\n" + self.concept_manual, cache="concept_manual"),
        ]

        messages: list[Message] = [Message(role="system", content=system_parts)]

        if include_fewshot and self.fewshot:
            messages.append(
                Message(
                    role="user",
                    content=[
                        TextPart(text=self._fewshot_text(), cache="fewshot"),
                    ],
                )
            )

        messages.append(
            Message(role="user", content=[TextPart(text=user_context)])
        )
        return messages

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _fewshot_text(self) -> str:
        """Render few-shot records as a single transcript block.

        Each example is serialized as ``USER:`` then ``ASSISTANT:`` JSON
        so the whole block collapses cleanly into one cacheable user
        message.
        """
        lines: list[str] = [
            "The following are canonical USER → ASSISTANT exchanges. "
            "Match this exact output style for the live context below."
        ]
        for i, ex in enumerate(self.fewshot, 1):
            user = ex.get("user", "").rstrip()
            assistant = ex.get("assistant", {})
            assistant_json = json.dumps(assistant, ensure_ascii=False)
            lines.append(f"\n--- Example {i} ---")
            lines.append("USER:")
            lines.append(user)
            lines.append("ASSISTANT:")
            lines.append(assistant_json)
        return "\n".join(lines)


def _load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"{path}: invalid JSON on line {lineno}: {e}"
                ) from e
    return rows
