"""Lightweight IC-only screening for mass formula filtering.

Evaluates rank IC on a single dataset without backtest, signal transform,
or CPCV.  Processes formulas in chunks to cap peak memory.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

from .compiler import FormulaCompiler
from .dataset import AlphaDataset
from .dsl import TensorSchema
from .evaluation import compute_forward_returns, compute_rank_ic
from .vm import StackVM, TensorStore

_CHUNK_SIZE = 64  # formulas per VM batch — keeps peak memory bounded


def fast_screen_ic(
    formulas: list[str],
    dataset: AlphaDataset,
    compiler: FormulaCompiler,
    vm: StackVM,
    schema: TensorSchema,
    *,
    min_abs_ic: float = 0.015,
    min_coverage: float = 0.3,
    fwd_period: int = 5,
    chunk_size: int = _CHUNK_SIZE,
) -> list[tuple[str, float]]:
    """Screen formulas by rank IC.  No backtest, no signal transform.

    Processes in chunks of *chunk_size* to avoid holding all 500+ alpha
    tensors in memory simultaneously.

    Returns ``(formula, rank_ic)`` pairs sorted by ``|rank_ic|`` descending.
    """
    if not formulas:
        return []

    store = TensorStore(dataset.fields)
    close = np.asarray(dataset.fields["close"], dtype=np.float32)
    fwd_returns = compute_forward_returns(close, periods=fwd_period)

    # Compile all formulas first (cheap, ~5ms total)
    compiled: list[tuple[str, object]] = []
    for f in formulas:
        try:
            compiled.append((f, compiler.compile(f, schema)))
        except Exception:
            continue

    if not compiled:
        return []

    # Process in chunks to bound memory
    results: list[tuple[str, float]] = []
    for start in range(0, len(compiled), chunk_size):
        chunk = compiled[start : start + chunk_size]
        programs = [prog for _, prog in chunk]
        chunk_formulas = [f for f, _ in chunk]

        try:
            alphas = vm.run_batch(programs, store)
        except Exception:
            continue

        for formula, alpha_raw in zip(chunk_formulas, alphas):
            if hasattr(alpha_raw, "cpu"):
                alpha = alpha_raw.cpu().numpy()
            else:
                alpha = np.asarray(alpha_raw, dtype=np.float32)

            coverage = float(np.isfinite(alpha).mean()) if alpha.size else 0.0
            if coverage < min_coverage:
                continue

            ic = compute_rank_ic(alpha, fwd_returns)
            if abs(ic) >= min_abs_ic:
                results.append((formula, ic))

        # Explicitly release chunk outputs
        del alphas

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    logger.info(
        "fast_screen: input={} compiled={} passed={} top_ic={:.4f}",
        len(formulas), len(compiled), len(results),
        abs(results[0][1]) if results else 0.0,
    )
    return results
