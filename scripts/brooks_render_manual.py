"""Render `prompts/brooks/concept_manual.md` from the Brooks taxonomy.

Inputs:
    docs/brooks/taxonomy.yaml          — single source of truth (this script's
                                         schema is enforced via pydantic).
    prompts/brooks/sections/*.md       — manually-authored prose chapters
                                         (intro, trader_equation, common_misreads
                                         are required; others are appended in
                                         lexicographic order if present).

Output:
    prompts/brooks/concept_manual.md   — deterministic, idempotent.

Determinism contract:
    * Two consecutive runs produce byte-identical output.
    * The output never embeds timestamps, machine-specific paths, or random ids.
    * Token count uses tiktoken `cl100k_base` when available; otherwise falls
      back to a coarse char/4 estimator (warning printed). The 10K limit is
      enforced against whichever count was used.

Usage:
    python scripts/brooks_render_manual.py            # write file
    python scripts/brooks_render_manual.py --check    # fail if file is stale
    python scripts/brooks_render_manual.py --dry-run  # print to stdout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parent.parent
TAXONOMY_PATH = REPO_ROOT / "docs" / "brooks" / "taxonomy.yaml"
SECTIONS_DIR = REPO_ROOT / "prompts" / "brooks" / "sections"
OUTPUT_PATH = REPO_ROOT / "prompts" / "brooks" / "concept_manual.md"

REQUIRED_SECTIONS = ("intro", "trader_equation", "common_misreads")
TOKEN_BUDGET = 10_000


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BarType(_Strict):
    name: str
    aliases: List[str] = Field(default_factory=list)
    definition: str
    detection: Dict[str, Any]
    examples: Optional[Dict[str, List[str]]] = None


class Pattern(_Strict):
    name: str
    full_name: str
    setup: str
    detection: Dict[str, Any]
    entry: str
    stop: str
    target: str
    typical_probability: float
    common_misreads: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _prob_in_range(self) -> "Pattern":
        if not 0.0 <= self.typical_probability <= 1.0:
            raise ValueError(f"pattern {self.name}: typical_probability out of [0,1]")
        return self


class Regime(_Strict):
    name: str
    detection: Dict[str, Any]
    entry_bias: str
    typical_probability: float
    common_misreads: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _prob_in_range(self) -> "Regime":
        if not 0.0 <= self.typical_probability <= 1.0:
            raise ValueError(f"regime {self.name}: typical_probability out of [0,1]")
        return self


class ProbBucket(_Strict):
    bucket: float
    description: str


class WorkedExample(_Strict):
    name: str
    probability: float
    risk_ticks: float
    reward_ticks: float
    expectancy_R: float
    verdict: str


class TraderEquation(_Strict):
    formula: str
    notes: str
    probability_buckets: List[ProbBucket]
    worked_examples: List[WorkedExample]


class AlwaysInRules(_Strict):
    description: str
    flip_to_long_when: List[str]
    flip_to_short_when: List[str]
    stay_neutral_when: List[str]
    hysteresis: str


class Taxonomy(_Strict):
    version: int
    bar_types: List[BarType]
    patterns: List[Pattern]
    regimes: List[Regime]
    trader_equation: TraderEquation
    always_in_rules: AlwaysInRules

    @model_validator(mode="after")
    def _coverage_check(self) -> "Taxonomy":
        required_bar_types = {
            "trend_bull", "trend_bear", "doji_bull", "doji_bear",
            "signal_bar", "climactic_bar", "reversal_bar",
            "ii", "iii", "outside_bar", "shaved_bar",
        }
        required_patterns = {
            "h1", "h2", "h3", "h4", "l1", "l2", "l3", "l4",
            "ii_breakout", "iii_breakout", "two_bar_reversal", "wedge",
            "double_top", "double_bottom", "final_flag", "micro_channel",
            "breakout", "breakout_pullback", "failed_breakout", "mtr",
            "measured_move",
        }
        required_regimes_min = 7
        seen_bar_types = {b.name for b in self.bar_types}
        seen_patterns = {p.name for p in self.patterns}
        missing_bar = required_bar_types - seen_bar_types
        missing_pat = required_patterns - seen_patterns
        if missing_bar:
            raise ValueError(f"missing required bar_types: {sorted(missing_bar)}")
        if missing_pat:
            raise ValueError(f"missing required patterns: {sorted(missing_pat)}")
        if len(self.regimes) < required_regimes_min:
            raise ValueError(
                f"need at least {required_regimes_min} regimes, got {len(self.regimes)}"
            )
        names_dup = [
            seq for seq in (
                [b.name for b in self.bar_types],
                [p.name for p in self.patterns],
                [r.name for r in self.regimes],
            )
            if len(seq) != len(set(seq))
        ]
        if names_dup:
            raise ValueError("duplicate names within bar_types / patterns / regimes")
        return self


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_taxonomy(path: Path = TAXONOMY_PATH) -> Taxonomy:
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Taxonomy.model_validate(raw)


def load_sections(sections_dir: Path = SECTIONS_DIR) -> Dict[str, str]:
    sections: Dict[str, str] = {}
    if not sections_dir.exists():
        raise FileNotFoundError(f"sections dir missing: {sections_dir}")
    for md in sorted(sections_dir.glob("*.md")):
        sections[md.stem] = md.read_text(encoding="utf-8").rstrip() + "\n"
    for required in REQUIRED_SECTIONS:
        if required not in sections:
            raise FileNotFoundError(f"required section missing: {required}.md")
    return sections


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _yaml_block(value: Any) -> str:
    """Dump a value as a yaml code-block fragment (sorted keys, deterministic)."""
    text = yaml.safe_dump(
        value,
        sort_keys=True,
        default_flow_style=False,
        allow_unicode=True,
        width=88,
    ).rstrip()
    return f"```yaml\n{text}\n```"


def _render_bar_type(b: BarType) -> str:
    parts = [f"### `{b.name}`"]
    if b.aliases:
        parts.append(f"*aliases:* {', '.join(f'`{a}`' for a in b.aliases)}")
    parts.append(b.definition.strip())
    parts.append("**detection**")
    parts.append(_yaml_block(b.detection))
    if b.examples:
        if b.examples.get("positive"):
            parts.append("**positive examples**")
            parts.append("\n".join(f"- {ex}" for ex in b.examples["positive"]))
        if b.examples.get("negative"):
            parts.append("**negative examples**")
            parts.append("\n".join(f"- {ex}" for ex in b.examples["negative"]))
    return "\n\n".join(parts)


def _render_pattern(p: Pattern) -> str:
    parts = [
        f"### `{p.name}` — {p.full_name}",
        f"*typical_probability:* `{p.typical_probability:.2f}`",
        "**setup**",
        p.setup.strip(),
        "**detection**",
        _yaml_block(p.detection),
        f"**entry** — {p.entry}",
        f"**stop** — {p.stop}",
        f"**target** — {p.target}",
    ]
    if p.common_misreads:
        parts.append("**common_misreads**")
        parts.append("\n".join(f"- {m}" for m in p.common_misreads))
    return "\n\n".join(parts)


def _render_regime(r: Regime) -> str:
    parts = [
        f"### `{r.name}`",
        f"*typical_probability:* `{r.typical_probability:.2f}`",
        "**detection**",
        _yaml_block(r.detection),
        f"**entry_bias** — {r.entry_bias}",
    ]
    if r.common_misreads:
        parts.append("**common_misreads**")
        parts.append("\n".join(f"- {m}" for m in r.common_misreads))
    return "\n\n".join(parts)


def _render_trader_equation(te: TraderEquation) -> str:
    parts = [
        "**formula**",
        f"```\n{te.formula}\n```",
        te.notes.strip(),
        "**probability_buckets**",
        "\n".join(
            f"- `{b.bucket:.2f}` — {b.description}" for b in te.probability_buckets
        ),
        "**worked_examples**",
        _yaml_block([w.model_dump() for w in te.worked_examples]),
    ]
    return "\n\n".join(parts)


def _render_always_in(rules: AlwaysInRules) -> str:
    parts = [
        rules.description.strip(),
        "**flip_to_long_when**",
        "\n".join(f"- {x}" for x in rules.flip_to_long_when),
        "**flip_to_short_when**",
        "\n".join(f"- {x}" for x in rules.flip_to_short_when),
        "**stay_neutral_when**",
        "\n".join(f"- {x}" for x in rules.stay_neutral_when),
        f"**hysteresis** — {rules.hysteresis}",
    ]
    return "\n\n".join(parts)


def render(taxonomy: Taxonomy, sections: Dict[str, str]) -> str:
    """Render the concept manual deterministically."""
    blocks: List[str] = []

    blocks.append(f"# Brooks Concept Manual (taxonomy v{taxonomy.version})")
    blocks.append(
        "> Auto-generated by `scripts/brooks_render_manual.py` from "
        "`docs/brooks/taxonomy.yaml` and `prompts/brooks/sections/*.md`. "
        "Do not edit by hand."
    )

    blocks.append(sections["intro"].strip())

    blocks.append("## Bar types")
    for bt in taxonomy.bar_types:
        blocks.append(_render_bar_type(bt))

    blocks.append("## Patterns")
    for pat in taxonomy.patterns:
        blocks.append(_render_pattern(pat))

    blocks.append("## Regimes")
    for reg in taxonomy.regimes:
        blocks.append(_render_regime(reg))

    blocks.append("## Trader's equation (canonical block)")
    blocks.append(_render_trader_equation(taxonomy.trader_equation))
    blocks.append(sections["trader_equation"].strip())

    blocks.append("## Always-in rules")
    blocks.append(_render_always_in(taxonomy.always_in_rules))

    blocks.append(sections["common_misreads"].strip())

    extras = [
        name for name in sorted(sections)
        if name not in REQUIRED_SECTIONS
    ]
    for name in extras:
        blocks.append(sections[name].strip())

    body = "\n\n".join(b.strip() for b in blocks if b.strip())
    return body.rstrip() + "\n"


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------


def count_tokens(text: str) -> tuple[int, str]:
    """Return (n_tokens, encoder_name). Falls back to char/4 if tiktoken absent."""
    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text)), "cl100k_base"
    except ImportError:
        return max(1, len(text) // 4), "char-div-4-fallback"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="exit 1 if the rendered output differs from the on-disk file")
    parser.add_argument("--dry-run", action="store_true",
                        help="print to stdout instead of writing the output file")
    parser.add_argument("--taxonomy", type=Path, default=TAXONOMY_PATH)
    parser.add_argument("--sections", type=Path, default=SECTIONS_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    args = parser.parse_args(argv)

    taxonomy = load_taxonomy(args.taxonomy)
    sections = load_sections(args.sections)
    rendered = render(taxonomy, sections)

    n_tokens, enc_name = count_tokens(rendered)
    sys.stderr.write(
        f"[brooks_render_manual] tokens={n_tokens} encoder={enc_name} "
        f"budget={args.token_budget}\n"
    )
    if n_tokens > args.token_budget:
        sys.stderr.write(
            f"ERROR: rendered manual exceeds token budget "
            f"({n_tokens} > {args.token_budget}).\n"
        )
        return 2

    if args.dry_run:
        sys.stdout.write(rendered)
        return 0

    if args.check:
        existing = args.output.read_text(encoding="utf-8") if args.output.exists() else ""
        if existing != rendered:
            sys.stderr.write(
                f"ERROR: {args.output} is stale — re-run "
                f"scripts/brooks_render_manual.py to refresh.\n"
            )
            return 1
        sys.stderr.write("[brooks_render_manual] up to date.\n")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    sys.stderr.write(f"[brooks_render_manual] wrote {args.output}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
