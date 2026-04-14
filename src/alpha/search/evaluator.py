"""Shared fast evaluation primitive for strategies.

Wraps compile → VM execute → IC metrics into a cached, reusable utility.
Strategies use this instead of reimplementing evaluation loops.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger

from ..core.compiler import FormulaCompiler, BytecodeProgram
from ..core.dataset import AlphaDataset
from ..core.dsl import TensorSchema
from ..core.vm import StackVM, TensorStore, to_numpy
from ..eval.metrics import compute_forward_returns, compute_rank_ic, compute_ic_metrics

try:
    import torch as _torch
except Exception:
    _torch = None

_MAX_SCREEN_ROWS = 30_000


class FormulaEvaluator:
    """Fast internal evaluation: compile → VM execute → IC metrics.

    Caches the TensorStore and forward returns so strategies don't rebuild them.
    NOT a replacement for the full CPCV evaluate_fn — this is for strategies
    that need cheap internal scoring during generation (MCTS rollout,
    REINFORCE reward, AlphaForge predictor training, etc.).

    Usage::

        evaluator = FormulaEvaluator(compiler, vm, schema, dataset)
        ic = evaluator.eval_ic("ts_mean(close, 20)")
        results = evaluator.eval_ic_batch(["ts_mean(close, 5)", "ts_std(volume, 10)"])
    """

    def __init__(
        self,
        compiler: FormulaCompiler,
        vm: StackVM,
        schema: TensorSchema,
        dataset: AlphaDataset,
        *,
        fwd_period: int = 5,
        max_rows: int = _MAX_SCREEN_ROWS,
    ) -> None:
        self.compiler = compiler
        self.vm = vm
        self.schema = schema

        # Subsample for speed — IC ranking is stable at ~30K rows
        T = dataset.fields["close"].shape[0]
        if T > max_rows:
            offset = T - max_rows
            fields = {k: v[offset:] for k, v in dataset.fields.items()}
        else:
            fields = dataset.fields

        # Prepare store (GPU if available)
        self._store = TensorStore(fields)
        if vm.backend == "torch" and vm.device is not None:
            self._store = vm._prepare_store(self._store)

        # Cache forward returns (numpy, for IC computation)
        close_np = np.asarray(fields["close"], dtype=np.float32)
        self._close_np = close_np
        self._fwd_returns = compute_forward_returns(close_np, periods=fwd_period)
        self._fwd_period = fwd_period

        # GPU forward returns for batched IC
        self._fwd_returns_gpu = None
        if _torch is not None and vm.backend == "torch" and vm.device is not None:
            self._fwd_returns_gpu = _torch.as_tensor(
                self._fwd_returns, device=vm.device, dtype=_torch.float32,
            )

        # Compile cache
        self._program_cache: dict[str, BytecodeProgram | None] = {}
        # Metrics cache — avoids re-evaluating identical formulas (keyed by expr_hash)
        self._metrics_cache: dict[str, float] = {}

    def _compile(self, formula: str) -> BytecodeProgram | None:
        """Compile with caching. Returns None on failure."""
        if formula not in self._program_cache:
            try:
                self._program_cache[formula] = self.compiler.compile(formula, self.schema)
            except Exception:
                self._program_cache[formula] = None
        return self._program_cache[formula]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eval_ic(self, formula: str) -> float:
        """Single formula rank IC. Returns 0.0 on failure."""
        program = self._compile(formula)
        if program is None:
            return 0.0
        # Check metrics cache by expr_hash
        if program.expr_hash in self._metrics_cache:
            return self._metrics_cache[program.expr_hash]
        try:
            alpha = self.vm.run(program, self._store)
            alpha_np = to_numpy(alpha)
            ic = compute_rank_ic(alpha_np, self._fwd_returns)
            self._metrics_cache[program.expr_hash] = ic
            return ic
        except Exception:
            self._metrics_cache[program.expr_hash] = 0.0
            return 0.0

    def eval_ic_batch(
        self,
        formulas: list[str],
        *,
        min_coverage: float = 0.3,
        chunk_size: int = 128,
    ) -> list[tuple[str, float]]:
        """Batch GPU rank IC. Returns [(formula, ic)] for valid formulas."""
        compiled: list[tuple[str, BytecodeProgram]] = []
        for f in formulas:
            prog = self._compile(f)
            if prog is not None:
                compiled.append((f, prog))
        if not compiled:
            return []

        use_gpu = (
            self._fwd_returns_gpu is not None
            and self._store.uses_torch()
        )

        results: list[tuple[str, float]] = []
        for start in range(0, len(compiled), chunk_size):
            chunk = compiled[start:start + chunk_size]
            programs = [prog for _, prog in chunk]
            chunk_formulas = [f for f, _ in chunk]
            try:
                alphas = self.vm.run_batch(programs, self._store)
            except Exception:
                continue

            if use_gpu and alphas and isinstance(alphas[0], _torch.Tensor):
                results.extend(self._batch_ic_gpu(chunk_formulas, alphas, min_coverage))
            else:
                results.extend(self._batch_ic_cpu(chunk_formulas, alphas, min_coverage))
            del alphas

        return results

    def eval_metrics(self, formula: str, fwd_windows: list[int] | None = None) -> dict[str, float]:
        """Full IC metrics (rank_ic, ic_ir, ic_std, per-window IC, etc.).

        Drop-in replacement for MCTSEngine._evaluate_formula.
        """
        program = self._compile(formula)
        if program is None:
            return {"rank_ic": 0.0, "ic_ir": 0.0, "error": "compile_failed"}
        try:
            alpha = self.vm.run(program, self._store)
            alpha_np = to_numpy(alpha)
            return compute_ic_metrics(alpha_np, self._close_np, fwd_windows=fwd_windows)
        except Exception as e:
            return {"rank_ic": 0.0, "ic_ir": 0.0, "error": str(e)}

    def execute(self, formula: str) -> np.ndarray | None:
        """Compile + execute, return raw alpha signal array.

        For strategies that need signal values (correlation checks, diversity).
        Returns None on failure.
        """
        program = self._compile(formula)
        if program is None:
            return None
        try:
            alpha = self.vm.run(program, self._store)
            return to_numpy(alpha)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _batch_ic_gpu(
        self,
        formulas: list[str],
        alphas: list[Any],
        min_coverage: float,
    ) -> list[tuple[str, float]]:
        """GPU-batched rank IC computation."""
        stacked = _torch.stack(alphas)
        coverage = (~stacked.isnan()).float().mean(dim=(1, 2))

        fp = self._fwd_period
        alpha_ic = stacked[:, :-fp, :]
        ret_ic = self._fwd_returns_gpu[:-fp, :].unsqueeze(0).expand_as(alpha_ic)
        mask = ~_torch.isnan(alpha_ic) & ~_torch.isnan(ret_ic)
        counts = mask.sum(dim=2)

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
        rank_ic_batch = (corr_safe.sum(dim=1) / valid_cnt).cpu().numpy()
        coverage_cpu = coverage.cpu().numpy()

        results = []
        for j, formula in enumerate(formulas):
            if coverage_cpu[j] >= min_coverage:
                results.append((formula, float(rank_ic_batch[j])))
        return results

    def _batch_ic_cpu(
        self,
        formulas: list[str],
        alphas: list[Any],
        min_coverage: float,
    ) -> list[tuple[str, float]]:
        """CPU fallback for rank IC."""
        results = []
        for formula, alpha_raw in zip(formulas, alphas):
            alpha = to_numpy(alpha_raw)
            coverage = float(np.isfinite(alpha).mean()) if alpha.size else 0.0
            if coverage < min_coverage:
                continue
            ic = compute_rank_ic(alpha, self._fwd_returns)
            results.append((formula, ic))
        return results
