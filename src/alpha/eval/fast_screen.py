"""Lightweight IC-only screening for mass formula filtering.

Evaluates rank IC on a single dataset without backtest, signal transform,
or CPCV.  Processes formulas in GPU-batched chunks for speed.
"""

from __future__ import annotations

from time import perf_counter

import numpy as np
from loguru import logger

from ..core.compiler import FormulaCompiler
from ..core.dataset import AlphaDataset
from ..core.dsl import TensorSchema
from .metrics import compute_forward_returns
from ..core.vm import StackVM, TensorStore

_CHUNK_SIZE = 128  # formulas per GPU batch

try:
    import torch as _torch
except Exception:
    _torch = None


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

    Uses batched GPU computation when available for ~50x speedup.
    Returns ``(formula, rank_ic)`` pairs sorted by ``|rank_ic|`` descending.
    """
    if not formulas:
        return []

    t0 = perf_counter()

    # Prepare store — convert to torch for GPU path
    store = TensorStore(dataset.fields)
    if vm.backend == "torch" and vm.device is not None:
        store = vm._prepare_store(store)

    close_np = np.asarray(dataset.fields["close"], dtype=np.float32)
    fwd_returns_np = compute_forward_returns(close_np, periods=fwd_period)

    # Compile all formulas first (cheap, ~5ms total)
    compiled: list[tuple[str, object]] = []
    for f in formulas:
        try:
            compiled.append((f, compiler.compile(f, schema)))
        except Exception:
            continue

    if not compiled:
        return []

    compile_ms = (perf_counter() - t0) * 1000
    logger.info("fast_screen: compiled={} in {:.0f}ms", len(compiled), compile_ms)

    # Check if GPU batch path is available
    use_gpu = (
        _torch is not None
        and vm.backend == "torch"
        and vm.device is not None
        and store.uses_torch()
    )

    if use_gpu:
        # Pre-transfer forward returns to GPU once
        fwd_returns_gpu = _torch.as_tensor(fwd_returns_np, device=vm.device, dtype=_torch.float32)

    results: list[tuple[str, float]] = []
    for start in range(0, len(compiled), chunk_size):
        chunk = compiled[start : start + chunk_size]
        programs = [prog for _, prog in chunk]
        chunk_formulas = [f for f, _ in chunk]

        try:
            alphas = vm.run_batch(programs, store)
        except Exception:
            continue

        if use_gpu and alphas and isinstance(alphas[0], _torch.Tensor):
            # --- GPU batched rank IC ---
            stacked = _torch.stack(alphas)  # (N, T, S)
            N = stacked.shape[0]

            # Coverage check on GPU
            coverage = (~stacked.isnan()).float().mean(dim=(1, 2))  # (N,)

            # Batch rank IC on GPU
            alpha_ic = stacked[:, :-fwd_period, :]  # (N, T', S)
            ret_ic = fwd_returns_gpu[:-fwd_period, :].unsqueeze(0).expand_as(alpha_ic)
            mask = ~_torch.isnan(alpha_ic) & ~_torch.isnan(ret_ic)
            counts = mask.sum(dim=2)  # (N, T')

            safe_a = _torch.where(mask, alpha_ic, _torch.zeros_like(alpha_ic))
            safe_r = _torch.where(mask, ret_ic, _torch.zeros_like(ret_ic))
            denom = counts.clamp(min=1).float()
            a_mean = safe_a.sum(dim=2) / denom
            r_mean = safe_r.sum(dim=2) / denom
            ca = _torch.where(mask, alpha_ic - a_mean.unsqueeze(2), _torch.zeros_like(alpha_ic))
            cr = _torch.where(mask, ret_ic - r_mean.unsqueeze(2), _torch.zeros_like(ret_ic))
            cov = (ca * cr).sum(dim=2)
            va = (ca * ca).sum(dim=2)
            vr = (cr * cr).sum(dim=2)
            valid = (counts >= 2) & (va > 1e-24) & (vr > 1e-24)
            corr = cov / (_torch.sqrt(va * vr) + 1e-24)
            corr = _torch.clamp(corr, -1.0, 1.0)
            corr_safe = _torch.where(valid, corr, _torch.zeros_like(corr))
            valid_cnt = valid.float().sum(dim=1).clamp(min=1)
            rank_ic_batch = (corr_safe.sum(dim=1) / valid_cnt).cpu().numpy()  # (N,)
            coverage_cpu = coverage.cpu().numpy()

            for j, formula in enumerate(chunk_formulas):
                if coverage_cpu[j] < min_coverage:
                    continue
                ic = float(rank_ic_batch[j])
                if abs(ic) >= min_abs_ic:
                    results.append((formula, ic))

            del stacked, alphas
        else:
            # --- CPU fallback (per-formula) ---
            from .metrics import compute_rank_ic

            for formula, alpha_raw in zip(chunk_formulas, alphas):
                if hasattr(alpha_raw, "cpu"):
                    alpha = alpha_raw.cpu().numpy()
                else:
                    alpha = np.asarray(alpha_raw, dtype=np.float32)

                coverage = float(np.isfinite(alpha).mean()) if alpha.size else 0.0
                if coverage < min_coverage:
                    continue

                ic = compute_rank_ic(alpha, fwd_returns_np)
                if abs(ic) >= min_abs_ic:
                    results.append((formula, ic))

            del alphas

        if (start + chunk_size) % (chunk_size * 4) == 0 or start + chunk_size >= len(compiled):
            logger.info(
                "fast_screen: progress {}/{} passed={}",
                min(start + chunk_size, len(compiled)), len(compiled), len(results),
            )

    results.sort(key=lambda x: abs(x[1]), reverse=True)

    elapsed_ms = (perf_counter() - t0) * 1000
    logger.info(
        "fast_screen: input={} compiled={} passed={} top_ic={:.4f} gpu={} {:.0f}ms",
        len(formulas), len(compiled), len(results),
        abs(results[0][1]) if results else 0.0,
        use_gpu, elapsed_ms,
    )
    return results
