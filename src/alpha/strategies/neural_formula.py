"""
Neural formula generation strategy (AlphaGPT-style).

Transformer + REINFORCE autoregressive generation of RPN token sequences,
with internal fast-IC evaluation for dense reward signal.

Key design (aligned with AlphaGPT):
  1. Sample large batch (4096) of RPN sequences autoregressively
  2. Decode ALL to DSL, compile, execute on dataset via StackVM
  3. Compute rank-IC as reward; invalid/constant formulas get negative reward
  4. REINFORCE update with normalized advantage on the FULL batch
  5. Return only top-K unique candidates to the orchestrator for CPCV
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger
from torch.distributions import Categorical

from ..evolution import Individual
from ..operators import OperatorRegistry, OperatorSpec
from ..dsl import TensorSchema
from ..pipeline import Lineage
from ..search_strategy import SearchContext, build_individual


# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

_CORE_FIELDS = (
    "close", "open", "high", "low", "volume",
    "open_interest", "funding_rate", "mark_close",
    "premium_close", "vwap", "long_short_ratio", "taker_buy_volume",
)

_CORE_OPS = (
    "abs", "log", "sign", "neg", "sigmoid",
    "add", "sub", "mul", "div",
    "ts_mean", "ts_std", "ts_sum", "ts_rank", "ts_zscore",
    "decay_linear", "delta", "returns_n", "ts_ema",
    "cs_rank", "cs_zscore", "cs_demean",
)

_WINDOW_CONSTANTS = (3, 5, 10, 20, 40, 60)

# Reward constants (aligned with AlphaGPT engine.py)
_REWARD_INVALID = -5.0
_REWARD_CONSTANT = -2.0
_IC_REWARD_SCALE = 20.0  # scale rank_ic (~0.02-0.10) to meaningful reward range


@dataclass(frozen=True)
class VocabToken:
    idx: int
    name: str
    kind: str        # "field", "op", "const", "bos"
    arity: int = 0
    const_value: int | float | None = None


class FormulaVocab:
    """Token vocabulary built from OperatorRegistry + TensorSchema."""

    def __init__(self, tokens: list[VocabToken]) -> None:
        self.tokens = tokens
        self.size = len(tokens)
        self._by_name: dict[str, VocabToken] = {t.name: t for t in tokens}
        self.operand_ids = frozenset(t.idx for t in tokens if t.kind in ("field", "const"))
        self.operator_ids = frozenset(t.idx for t in tokens if t.kind == "op")
        self.bos_id = next(t.idx for t in tokens if t.kind == "bos")

        # Precompute arity tensor for vectorized masking
        self.arity_tensor = torch.tensor([t.arity for t in tokens], dtype=torch.long)
        self.is_operator_tensor = torch.tensor(
            [1 if t.kind == "op" else 0 for t in tokens], dtype=torch.bool,
        )
        self.is_operand_tensor = torch.tensor(
            [1 if t.kind in ("field", "const") else 0 for t in tokens], dtype=torch.bool,
        )
        self.is_bos_tensor = torch.tensor(
            [1 if t.kind == "bos" else 0 for t in tokens], dtype=torch.bool,
        )

    @classmethod
    def from_registry(
        cls,
        registry: OperatorRegistry,
        schema: TensorSchema,
        extra_fields: tuple[str, ...] = (),
        extra_ops: tuple[str, ...] = (),
        windows: tuple[int, ...] = _WINDOW_CONSTANTS,
    ) -> "FormulaVocab":
        tokens: list[VocabToken] = []
        idx = 0

        tokens.append(VocabToken(idx=idx, name="<bos>", kind="bos"))
        idx += 1

        for f in list(_CORE_FIELDS) + list(extra_fields):
            if f in schema.fields:
                tokens.append(VocabToken(idx=idx, name=f, kind="field"))
                idx += 1

        for w in windows:
            tokens.append(VocabToken(idx=idx, name=f"W{w}", kind="const", const_value=w))
            idx += 1

        seen_ops: set[str] = set()
        for op_name in list(_CORE_OPS) + list(extra_ops):
            if op_name in seen_ops:
                continue
            norm = registry.normalize_name(op_name)
            if not registry.has(norm):
                continue
            spec = registry.get(norm)
            tokens.append(VocabToken(idx=idx, name=norm, kind="op", arity=spec.min_args))
            seen_ops.add(op_name)
            idx += 1

        return cls(tokens)

    def get_token(self, token_id: int) -> VocabToken:
        return self.tokens[token_id]


# ---------------------------------------------------------------------------
# RPN -> DSL conversion
# ---------------------------------------------------------------------------


def rpn_to_dsl(token_ids: list[int], vocab: FormulaVocab) -> str | None:
    """Convert RPN token sequence to infix DSL string."""
    stack: list[str] = []
    for tid in token_ids:
        tok = vocab.get_token(tid)
        if tok.kind == "bos":
            continue
        if tok.kind == "field":
            stack.append(tok.name)
        elif tok.kind == "const":
            stack.append(str(tok.const_value))
        elif tok.kind == "op":
            if len(stack) < tok.arity:
                return None
            args = stack[-tok.arity:]
            stack = stack[:-tok.arity]
            stack.append(f"{tok.name}({', '.join(args)})")
        else:
            return None
    return stack[0] if len(stack) == 1 else None


# ---------------------------------------------------------------------------
# Vectorized action masking (GPU-friendly)
# ---------------------------------------------------------------------------


def compute_action_mask_batch(
    stack_depths: torch.Tensor,
    step: int,
    max_len: int,
    vocab: FormulaVocab,
    device: torch.device,
) -> torch.Tensor:
    """Vectorized action mask for a batch of sequences.

    Args:
        stack_depths: [B] current stack depth per sequence
        step: current generation step
        max_len: maximum formula length
        device: target device

    Returns:
        mask: [B, V] with 0 for valid, -inf for invalid
    """
    B = stack_depths.shape[0]
    V = vocab.size
    remaining = max_len - step - 1

    arity = vocab.arity_tensor.to(device)          # [V]
    is_op = vocab.is_operator_tensor.to(device)     # [V]
    is_operand = vocab.is_operand_tensor.to(device) # [V]
    is_bos = vocab.is_bos_tensor.to(device)         # [V]

    # new_depth after applying each token: [B, V]
    # operators: depth - arity + 1;  operands: depth + 1
    depth = stack_depths.unsqueeze(1)  # [B, 1]
    new_depth = torch.where(
        is_op.unsqueeze(0),
        depth - arity.unsqueeze(0) + 1,
        depth + 1,
    )  # [B, V]

    mask = torch.zeros(B, V, device=device)

    # BOS never valid
    mask[:, is_bos] = float("-inf")

    # Operators: need stack_depth >= arity
    op_invalid = is_op.unsqueeze(0) & (depth < arity.unsqueeze(0))  # [B, V]
    mask = mask.masked_fill(op_invalid, float("-inf"))

    # Final step: must reach depth == 1
    if remaining == 0:
        not_one = new_depth != 1
        mask = mask.masked_fill(not_one, float("-inf"))
    else:
        # Operands: don't push if stack too deep to reduce in remaining steps
        too_deep = is_operand.unsqueeze(0) & (new_depth - 1 > remaining)
        mask = mask.masked_fill(too_deep, float("-inf"))

        # Operators: check that resulting depth can still reach 1
        op_unreachable = is_op.unsqueeze(0) & (new_depth - 1 > remaining)
        mask = mask.masked_fill(op_unreachable, float("-inf"))

    # Fallback: if all masked for any row, allow operands
    all_masked = mask.max(dim=1).values == float("-inf")  # [B]
    if all_masked.any():
        fallback = is_operand.unsqueeze(0).expand(B, V)
        fallback_mask = all_masked.unsqueeze(1) & fallback
        mask = mask.masked_fill(fallback_mask, 0.0)

    return mask


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class FormulaTransformer(nn.Module):
    """Causal Transformer for autoregressive RPN generation."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        max_len: int = 16,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len + 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = RMSNorm(d_model)
        self.head_policy = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.size()
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.encoder(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        return self.head_policy(x[:, -1, :])  # [B, vocab_size]


# ---------------------------------------------------------------------------
# LoRD regularization
# ---------------------------------------------------------------------------


@torch.no_grad()
def lord_step(
    model: nn.Module,
    decay_rate: float = 1e-3,
    iterations: int = 5,
    keywords: tuple[str, ...] = ("attention", "self_attn"),
) -> None:
    """Newton-Schulz low-rank decay on 2D attention parameters."""
    for name, W in model.named_parameters():
        if not W.requires_grad or W.ndim != 2:
            continue
        if not any(k in name for k in keywords):
            continue
        X = W.float()
        transposed = X.shape[0] > X.shape[1]
        if transposed:
            X = X.T
        norm = X.norm() + 1e-8
        Y = X / norm
        I = torch.eye(Y.shape[-1], device=Y.device, dtype=Y.dtype)
        for _ in range(iterations):
            Y = 0.5 * Y @ (3.0 * I - Y.T @ Y)
        if transposed:
            Y = Y.T
        W.sub_(decay_rate * Y.to(W.dtype))


# ---------------------------------------------------------------------------
# Training history
# ---------------------------------------------------------------------------


@dataclass
class TrainingSnapshot:
    round_idx: int
    step: int
    loss: float
    avg_reward: float
    best_reward: float
    valid_ratio: float
    unique: int
    best_formula: str


class TrainingHistory:
    def __init__(self) -> None:
        self.snapshots: list[TrainingSnapshot] = []

    def record(self, snap: TrainingSnapshot) -> None:
        self.snapshots.append(snap)

    def to_dict(self) -> list[dict[str, Any]]:
        return [
            {
                "round": s.round_idx, "step": s.step,
                "loss": round(s.loss, 5),
                "avg_reward": round(s.avg_reward, 5),
                "best_reward": round(s.best_reward, 5),
                "valid_ratio": round(s.valid_ratio, 4),
                "unique": s.unique,
                "best_formula": s.best_formula,
            }
            for s in self.snapshots
        ]

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), indent=2))

    def plot(self, path: str | Path) -> None:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning("matplotlib not installed, skipping plot")
            return

        if len(self.snapshots) < 2:
            return

        steps = [s.step for s in self.snapshots]
        fig, axes = plt.subplots(2, 2, figsize=(14, 8), tight_layout=True)
        fig.suptitle("Neural Formula Strategy — Training Progress", fontsize=13)

        ax = axes[0, 0]
        ax.plot(steps, [s.loss for s in self.snapshots], "k-", lw=1.2)
        ax.set_ylabel("Loss")
        ax.set_xlabel("Step")
        ax.set_title("REINFORCE Loss")
        ax.grid(True, alpha=0.3)

        ax = axes[0, 1]
        ax.plot(steps, [s.avg_reward for s in self.snapshots], "b-", lw=1, alpha=0.6, label="avg")
        ax.plot(steps, [s.best_reward for s in self.snapshots], "g-", lw=1.2, label="best")
        # Running average
        if len(steps) >= 5:
            window = min(10, len(steps) // 3)
            avg = np.convolve([s.avg_reward for s in self.snapshots],
                              np.ones(window) / window, mode="valid")
            ax.plot(steps[window - 1:], avg, "b-", lw=2, label=f"avg (ma{window})")
        ax.axhline(0, color="gray", ls=":", lw=0.8)
        ax.set_ylabel("Reward")
        ax.set_xlabel("Step")
        ax.legend(fontsize=8)
        ax.set_title("Reward")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 0]
        ax.plot(steps, [s.valid_ratio for s in self.snapshots], "steelblue", lw=1.2)
        ax.set_ylabel("Valid / Sampled")
        ax.set_xlabel("Step")
        ax.set_ylim(0, 1)
        ax.set_title("Valid Formula Ratio")
        ax.grid(True, alpha=0.3)

        ax = axes[1, 1]
        ax.plot(steps, [s.unique for s in self.snapshots], "coral", lw=1.2)
        ax.set_ylabel("Unique Candidates")
        ax.set_xlabel("Step")
        ax.set_title("Unique Candidates per Step")
        ax.grid(True, alpha=0.3)

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        logger.info("neural_formula.plot saved to {}", p)


# ---------------------------------------------------------------------------
# NeuralFormulaStrategy
# ---------------------------------------------------------------------------


class NeuralFormulaStrategy:
    """AlphaGPT-style neural formula generation.

    Core loop (aligned with AlphaGPT engine.py):
      1. Sample large batch of RPN sequences (default 4096)
      2. Decode ALL → DSL → compile → execute via StackVM → compute rank-IC
      3. Assign rewards: invalid=-5, constant=-2, valid=IC*scale
      4. Normalize advantage: (r - mean) / (std + eps)
      5. REINFORCE gradient update on the FULL batch
      6. Return top-K unique candidates to orchestrator for CPCV evaluation
    """

    def __init__(
        self,
        registry: OperatorRegistry | None = None,
        schema: TensorSchema | None = None,
        *,
        sample_batch: int = 4096,
        max_candidates: int = 30,
        max_formula_len: int = 12,
        train_steps_per_round: int = 5,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        lr: float = 1e-3,
        lord_decay: float = 1e-3,
        enable_lord: bool = True,
        activation_frequency: int = 1,
        min_round: int = 1,
        device: str | None = None,
        output_dir: str = "data/alpha_lab/neural",
    ) -> None:
        self._registry = registry or OperatorRegistry()
        self._schema = schema or TensorSchema.default_market_schema()

        self._sample_batch = sample_batch
        self._max_candidates = max_candidates
        self._max_len = max_formula_len
        self._train_steps = train_steps_per_round
        self._lr = lr
        self._lord_decay = lord_decay
        self._enable_lord = enable_lord
        self._activation_freq = activation_frequency
        self._min_round = min_round
        self._output_dir = Path(output_dir)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)

        self._vocab = FormulaVocab.from_registry(self._registry, self._schema)
        self._model = FormulaTransformer(
            vocab_size=self._vocab.size,
            d_model=d_model, nhead=nhead, num_layers=num_layers,
            max_len=self._max_len,
        ).to(self._device)
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=lr)

        self._global_step = 0
        self._best_ic = 0.0
        self._best_formula = ""
        self.history = TrainingHistory()

        # Cached forward returns for internal evaluation (built lazily)
        self._fwd_returns: np.ndarray | None = None
        self._store: Any = None

    @property
    def name(self) -> str:
        return "neural_formula"

    def should_activate(self, ctx: SearchContext) -> bool:
        return (
            ctx.round_idx >= self._min_round
            and ctx.round_idx % self._activation_freq == 0
            and ctx.dataset is not None
        )

    # ------------------------------------------------------------------
    # SearchStrategy interface
    # ------------------------------------------------------------------

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        from ..tracing import tracer

        with tracer.start_span(
            "neural_generate", kind="search",
            round=ctx.round_idx, train_steps=self._train_steps,
        ) as span:
            # Lazy init: build internal evaluation cache from dataset
            self._ensure_eval_cache(ctx)

            # Multiple internal training steps
            all_formulas: dict[str, float] = {}  # formula → best_ic
            for _ in range(self._train_steps):
                step_formulas = self._train_step(ctx)
                for f, ic in step_formulas.items():
                    if f not in all_formulas or abs(ic) > abs(all_formulas[f]):
                        all_formulas[f] = ic

            # Build Individuals from best unique formulas
            ranked = sorted(all_formulas.items(), key=lambda x: abs(x[1]), reverse=True)
            candidates: list[Individual] = []
            for formula, ic in ranked:
                if len(candidates) >= self._max_candidates:
                    break
                ind = build_individual(
                    ctx.compiler, ctx.schema, formula,
                    Lineage(origin="neural_formula", screen_ic=round(ic, 5)),
                )
                if ind and ind.expr_hash not in ctx.seen_hashes:
                    candidates.append(ind)

            span.set("train_steps", self._train_steps)
            span.set("unique_formulas", len(all_formulas))
            span.set("candidates", len(candidates))
            logger.info(
                "neural_formula.generate round={} steps={} formulas={} candidates={}",
                ctx.round_idx, self._train_steps, len(all_formulas), len(candidates),
            )

        return candidates

    def on_evaluation_complete(
        self, ctx: SearchContext, evaluated: list[Individual],
    ) -> None:
        # Record to strategy memory (gradient update already done in _train_step)
        if ctx.strategy_memory is not None:
            for ind in evaluated:
                ctx.strategy_memory.record(
                    formula=ind.formula,
                    theme_id="neural_formula",
                    metrics=ind.metrics,
                    is_novel=True,
                    round_idx=ctx.round_idx,
                    all_fields=ctx.schema.fields,
                )

        # Save history periodically
        if ctx.round_idx == ctx.total_rounds - 1 or (ctx.round_idx + 1) % 5 == 0:
            self.history.save(self._output_dir / "training_history.json")
            self.history.plot(self._output_dir / "training_curves.png")

    def get_stats(self) -> dict[str, Any]:
        return {
            "strategy": self.name,
            "global_step": self._global_step,
            "best_ic": round(self._best_ic, 5),
            "best_formula": self._best_formula,
            "vocab_size": self._vocab.size,
            "device": str(self._device),
            "history": self.history.to_dict(),
        }

    # ------------------------------------------------------------------
    # Internal training loop (aligned with AlphaGPT engine.py)
    # ------------------------------------------------------------------

    def _ensure_eval_cache(self, ctx: SearchContext) -> None:
        """Build forward-returns cache from dataset for internal IC evaluation."""
        if self._fwd_returns is not None:
            return
        from ..evaluation import compute_forward_returns
        from ..vm import TensorStore
        ds = ctx.dataset
        close = np.asarray(ds.fields["close"], dtype=np.float32)
        self._fwd_returns = compute_forward_returns(close, periods=5)
        self._store = TensorStore(ds.fields)

    def _train_step(self, ctx: SearchContext) -> dict[str, float]:
        """One REINFORCE training step. Returns {formula: ic} for valid formulas."""
        from ..evaluation import compute_rank_ic
        from ..vm import StackVM

        self._model.train()
        bs = self._sample_batch
        device = self._device
        vocab = self._vocab

        # 1. Autoregressive sampling with vectorized action masking
        sequences = torch.full((bs, 1), vocab.bos_id, dtype=torch.long, device=device)
        all_log_probs: list[torch.Tensor] = []
        stack_depths = torch.zeros(bs, dtype=torch.long, device=device)

        for step in range(self._max_len):
            logits = self._model(sequences)  # [B, V]
            mask = compute_action_mask_batch(stack_depths, step, self._max_len, vocab, device)
            masked_logits = logits + mask
            dist = Categorical(logits=masked_logits)
            action = dist.sample()  # [B]
            all_log_probs.append(dist.log_prob(action))

            # Update stack depths (vectorized)
            tok_arity = vocab.arity_tensor.to(device)[action]  # [B]
            tok_is_op = vocab.is_operator_tensor.to(device)[action]  # [B]
            stack_depths = torch.where(
                tok_is_op,
                stack_depths - tok_arity + 1,
                stack_depths + 1,
            )

            sequences = torch.cat([sequences, action.unsqueeze(1)], dim=1)

        log_probs_tensor = torch.stack(all_log_probs, dim=1)  # [B, L]

        # 2. Decode ALL sequences → DSL → compile → evaluate → reward
        raw_tokens = sequences[:, 1:].cpu().tolist()
        rewards = torch.full((bs,), _REWARD_INVALID, dtype=torch.float32, device=device)
        valid_formulas: dict[str, float] = {}  # formula → IC

        vm = StackVM()
        for i, toks in enumerate(raw_tokens):
            dsl = rpn_to_dsl(toks, vocab)
            if dsl is None:
                continue  # invalid RPN → keep _REWARD_INVALID

            try:
                program = ctx.compiler.compile(dsl, ctx.schema)
            except ValueError:
                continue  # compile error → keep _REWARD_INVALID

            try:
                alpha = vm.run(program, self._store)
            except Exception:
                continue

            if hasattr(alpha, "cpu"):
                alpha_np = alpha.cpu().numpy()
            else:
                alpha_np = np.asarray(alpha, dtype=np.float32)

            # Constant signal check (aligned with AlphaGPT)
            alpha_std = np.nanstd(alpha_np)
            if alpha_std < 1e-6:
                rewards[i] = _REWARD_CONSTANT
                continue

            ic = compute_rank_ic(alpha_np, self._fwd_returns)
            rewards[i] = abs(ic) * _IC_REWARD_SCALE
            valid_formulas[dsl] = ic

            if abs(ic) > abs(self._best_ic):
                self._best_ic = ic
                self._best_formula = dsl

        # 3. Normalize advantage (AlphaGPT style)
        advantage = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # 4. REINFORCE loss (sum over time steps, like AlphaGPT)
        loss = -(log_probs_tensor * advantage.unsqueeze(1)).sum(dim=1).mean()

        self._optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
        self._optimizer.step()

        if self._enable_lord:
            lord_step(self._model, decay_rate=self._lord_decay)

        # 5. Record metrics
        self._global_step += 1
        n_valid = int((rewards > _REWARD_INVALID).sum().item())
        avg_r = rewards.mean().item()
        best_r = rewards.max().item()

        self.history.record(TrainingSnapshot(
            round_idx=ctx.round_idx,
            step=self._global_step,
            loss=loss.item(),
            avg_reward=avg_r,
            best_reward=best_r,
            valid_ratio=n_valid / bs,
            unique=len(valid_formulas),
            best_formula=self._best_formula,
        ))

        if self._global_step % 5 == 0 or self._global_step <= 3:
            logger.info(
                "neural_formula.step {} loss={:.3f} avg_r={:.3f} best_r={:.3f} "
                "valid={}/{} unique={} best_ic={:.4f}",
                self._global_step, loss.item(), avg_r, best_r,
                n_valid, bs, len(valid_formulas), self._best_ic,
            )

        return valid_formulas
