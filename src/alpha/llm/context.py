"""
LLM context builder for alpha pipeline analysis.

Converts pipeline execution state into concise, structured text that can
be sent to an LLM for analysis, bottleneck identification, and strategy
suggestions. Does NOT call any LLM — only builds text.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ..search.pipeline import ArchiveEntry, PipelineRecord, StageKind

if TYPE_CHECKING:
    from ..knowledge.memory import StrategyMemory


def build_pipeline_summary(
    pipeline: PipelineRecord,
    archive: list[ArchiveEntry],
    strategy_memory: "StrategyMemory | None" = None,
    max_formulas: int = 8,
) -> str:
    """Build a concise (~2K token) text summary of current search state.

    Suitable for injection into an LLM prompt for search analysis.
    """
    lines: list[str] = []

    # Header
    n_rounds = len(pipeline.rounds)
    lines.append(f"# Alpha Search Summary (job={pipeline.job_id})")
    lines.append(f"Rounds: {n_rounds} | Evaluations: {pipeline.total_evaluations} | Rejected: {pipeline.total_rejected}")
    if pipeline.total_evaluations > 0:
        reject_rate = pipeline.total_rejected / max(pipeline.total_evaluations + pipeline.total_rejected, 1)
        lines.append(f"Rejection rate: {reject_rate:.1%}")
    lines.append("")

    # Per-round stage summary
    lines.append("## Round-by-round")
    for rr in pipeline.rounds:
        strats = ", ".join(rr.strategies_activated) or "none"
        lines.append(f"R{rr.round_idx}: strategies=[{strats}] archive={rr.archive_size} best={rr.best_fitness:.4f} ({rr.duration_ms:.0f}ms)")
        for stage in rr.stages:
            arrow = f"{stage.input_count}→{stage.output_count}"
            lines.append(f"  {stage.kind.value:15s} [{stage.strategy}] {arrow} ({stage.duration_ms:.0f}ms)")
    lines.append("")

    # Stage-level aggregates (bottleneck detection)
    stage_stats: dict[str, dict[str, float]] = {}
    for rr in pipeline.rounds:
        for s in rr.stages:
            key = s.kind.value
            if key not in stage_stats:
                stage_stats[key] = {"total_in": 0, "total_out": 0, "total_ms": 0}
            stage_stats[key]["total_in"] += s.input_count
            stage_stats[key]["total_out"] += s.output_count
            stage_stats[key]["total_ms"] += s.duration_ms

    if stage_stats:
        lines.append("## Stage Aggregates")
        for kind, st in stage_stats.items():
            passthrough = st["total_out"] / max(st["total_in"], 1) if kind != "generate" else 1.0
            lines.append(f"  {kind:15s}: in={st['total_in']:.0f} out={st['total_out']:.0f} pass={passthrough:.0%} time={st['total_ms']:.0f}ms")

        # Identify bottleneck
        worst_stage = None
        worst_drop = 1.0
        for kind, st in stage_stats.items():
            if kind == "generate":
                continue
            passthrough = st["total_out"] / max(st["total_in"], 1)
            if passthrough < worst_drop:
                worst_drop = passthrough
                worst_stage = kind
        if worst_stage and worst_drop < 0.5:
            lines.append(f"  ** Bottleneck: {worst_stage} passes only {worst_drop:.0%} of candidates **")
        lines.append("")

    # Archive top-N
    if archive:
        lines.append(f"## Archive Top-{min(max_formulas, len(archive))}")
        for i, entry in enumerate(archive[:max_formulas], 1):
            lines.append(
                f"{i}. [{entry.origin}] IC={entry.rank_ic:.4f} Sharpe={entry.sharpe:.3f} "
                f"Turnover={entry.turnover:.4f} Fitness={entry.fitness:.4f}"
            )
            lines.append(f"   {entry.formula}")
        lines.append("")

    # Strategy memory insights
    if strategy_memory is not None:
        lines.append("## Strategy Memory Insights")

        # Theme performance
        theme_stats = strategy_memory.get_theme_summary()
        if theme_stats:
            lines.append("Themes (by avg fitness):")
            sorted_themes = sorted(theme_stats.items(), key=lambda x: x[1].get("avg_fitness", 0), reverse=True)
            for theme_id, stats in sorted_themes[:8]:
                lines.append(
                    f"  {theme_id}: n={stats.get('count', 0)} "
                    f"avg_fitness={stats.get('avg_fitness', 0):.4f} "
                    f"success={stats.get('success_rate', 0):.0%}"
                )

        # Top operators
        op_stats = strategy_memory.get_operator_summary()
        if op_stats:
            lines.append("Top operators (by avg fitness):")
            sorted_ops = sorted(op_stats.items(), key=lambda x: x[1].get("avg_fitness", 0), reverse=True)
            for op, stats in sorted_ops[:6]:
                lines.append(f"  {op}: n={stats.get('count', 0)} avg_fitness={stats.get('avg_fitness', 0):.4f}")
        lines.append("")

    return "\n".join(lines)


def build_analysis_prompt(
    summary: str,
    user_instruction: str = "",
) -> str:
    """Wrap pipeline summary into an LLM analysis prompt.

    The caller decides which LLM to send this to.
    """
    base = (
        "You are an expert quantitative researcher analyzing an automated "
        "alpha factor search pipeline for cryptocurrency futures.\n\n"
        "Below is a summary of the current search state. Analyze it and provide:\n"
        "1. **Diagnosis**: What is working well and what isn't?\n"
        "2. **Bottlenecks**: Which pipeline stages are rejecting the most candidates and why?\n"
        "3. **Suggestions**: Concrete, actionable suggestions for the next search iteration.\n"
        "   - Which financial themes or operator patterns to explore\n"
        "   - Whether to adjust search parameters (population size, generations, etc.)\n"
        "   - Specific formula templates that might work based on the archive patterns\n\n"
        "Be concise and specific. Reference actual formulas and metrics from the summary.\n\n"
    )

    if user_instruction:
        base += f"User's specific question: {user_instruction}\n\n"

    return base + "---\n\n" + summary
