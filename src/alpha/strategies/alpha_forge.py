"""
AlphaForge: 生成-预测代理模型因子挖掘策略.

Paper: "AlphaForge: A Framework to Mine and Dynamically Combine
       Formulaic Alpha Factors" (AAAI 2025, Shi et al.)

核心思想:
  1. Predictor (代理模型) 从 RPN token 序列学习 IC 分布
  2. Generator 将噪声 z 映射为 RPN logit 矩阵, 通过 Gumbel-Softmax 可微采样
  3. Generator 训练目标: 最大化 Predictor 输出 + 多样性损失防止模式坍塌
  4. 有效 RPN 序列解析为 DSL 公式返回给编排器

与 NeuralFormulaStrategy 的区别:
  - 代理模型梯度引导 (vs. REINFORCE 策略梯度)
  - 并行一次性生成 (vs. 自回归序列)
  - 显式多样性损失 (vs. 无显式多样性)
"""

from __future__ import annotations

from typing import Any, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from loguru import logger

from ..search.context import SearchContext
from ..search.evolution import Individual
from ..search.pipeline import Lineage
from .base import BaseStrategy, StrategyMeta
from .neural_formula import FormulaVocab, compute_action_mask_batch, rpn_to_dsl

# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------


class _Predictor(nn.Module):
    """代理模型: 从 RPN token 嵌入预测 IC."""

    def __init__(self, vocab_size: int, max_len: int, d: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(max_len * d, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.net(self.emb(token_ids)).squeeze(-1)

    def forward_soft(self, soft_tokens: torch.Tensor) -> torch.Tensor:
        """soft_tokens: [B, S, V] Gumbel-Softmax 输出 → fitness [B]."""
        return self.net(soft_tokens @ self.emb.weight).squeeze(-1)


class _Generator(nn.Module):
    """生成器: 噪声 z → 位置级 token logit 矩阵."""

    def __init__(self, vocab_size: int, max_len: int, z_dim: int = 32):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_len * vocab_size),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: [B, z_dim] → logits [B, S, V]."""
        return self.net(z).view(-1, self.max_len, self.vocab_size)


# ---------------------------------------------------------------------------
# Strategy
# ---------------------------------------------------------------------------


class AlphaForgeStrategy(BaseStrategy):
    """AlphaForge 生成-预测因子挖掘 (AAAI 2025).

    每轮:
    1. 从 Generator 采样 RPN 序列 → 解析有效公式 → 快速 IC 评估 → 构建样本库
    2. 训练 Predictor 学习 (RPN, IC) 映射
    3. 冻结 Predictor, 通过 Gumbel-Softmax 训练 Generator 最大化预测值 + 多样性
    4. 用训练好的 Generator 生成候选公式
    """

    meta: ClassVar[StrategyMeta] = StrategyMeta(
        registry_name="alpha_forge",
        label="AlphaForge 生成预测",
        brief="代理模型 + Gumbel-Softmax 生成器挖掘因子 (AlphaForge)",
        detail=(
            "训练 Predictor 代理模型学习因子 IC 分布, "
            "Generator 通过 Gumbel-Softmax 梯度优化生成高质量公式, "
            "多样性损失防止模式坍塌"
        ),
    )

    def __init__(
        self,
        compiler: Any,
        vm: Any,
        schema: Any,
        registry: Any,
        *,
        max_len: int = 16,
        z_dim: int = 32,
        lr_p: float = 1e-3,
        lr_g: float = 5e-4,
        lambda_div: float = 0.5,
        gumbel_tau: float = 0.5,
        predictor_epochs: int = 8,
        generator_epochs: int = 15,
        explore_batch: int = 512,
        gen_batch: int = 512,
        top_k: int = 30,
    ) -> None:
        self.compiler = compiler
        self.vm = vm
        self.schema = schema
        self.op_registry = registry

        self.max_len = max_len
        self.z_dim = z_dim
        self.lambda_div = lambda_div
        self.gumbel_tau = gumbel_tau
        self.predictor_epochs = predictor_epochs
        self.generator_epochs = generator_epochs
        self.explore_batch = explore_batch
        self.gen_batch = gen_batch
        self.top_k = top_k

        self.vocab = FormulaVocab.from_registry(registry, schema)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor = _Predictor(self.vocab.size, max_len).to(self.device)
        self.generator = _Generator(self.vocab.size, max_len, z_dim).to(self.device)
        self.opt_p = torch.optim.Adam(self.predictor.parameters(), lr=lr_p)
        self.opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr_g)

        # 样本库: (token_ids_tensor, ic_value)
        self._token_lib: list[torch.Tensor] = []
        self._ic_lib: list[float] = []
        self._seen_formulas: set[str] = set()
        self._stats = {"explored": 0, "valid": 0, "p_loss": 0.0, "g_loss": 0.0}

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------

    def should_activate(self, ctx: SearchContext) -> bool:
        return ctx.evaluator is not None

    def generate_candidates(self, ctx: SearchContext) -> list[Individual]:
        # 1. 探索: 采样 → 解析 → 快速IC → 样本库
        self._explore(ctx)
        if len(self._token_lib) < 20:
            logger.debug("[AlphaForge] 样本库不足, 跳过生成")
            return []

        # 2. 训练代理模型
        self._train_predictor()
        # 3. 训练生成器
        self._train_generator()
        # 4. 生成候选
        formulas = self._generate_formulas()
        logger.info(
            f"[AlphaForge] 样本库={len(self._token_lib)}, "
            f"生成={len(formulas)}, P_loss={self._stats['p_loss']:.4f}, "
            f"G_loss={self._stats['g_loss']:.4f}"
        )
        return self.compile_and_dedup(
            ctx,
            formulas,
            lineage_fn=lambda _f: Lineage(origin="alpha_forge"),
            limit=self.top_k,
        )

    def get_stats(self) -> dict[str, Any]:
        return {"strategy": self.name, **self._stats}

    # ------------------------------------------------------------------
    # 探索: 采样 + IC 评估 → 样本库
    # ------------------------------------------------------------------

    def _explore(self, ctx: SearchContext) -> None:
        """从 Generator 采样 RPN, 解析有效公式, 快速 IC 评估."""
        self.generator.eval()
        with torch.no_grad():
            z = torch.randn(self.explore_batch, self.z_dim, device=self.device)
            all_logits = self.generator(z)

        parsed = self._decode_with_mask(all_logits)
        self._stats["explored"] += self.explore_batch

        # 快速 IC 评估
        new_formulas = [f for f, _ in parsed if f not in self._seen_formulas]
        if not new_formulas or ctx.evaluator is None:
            return

        ic_results = ctx.evaluator.eval_ic_batch(new_formulas)
        for formula, ic in ic_results:
            if formula in self._seen_formulas:
                continue
            self._seen_formulas.add(formula)
            token_ids = self._find_tokens_for(formula, parsed)
            if token_ids is not None:
                self._token_lib.append(token_ids)
                self._ic_lib.append(abs(ic))
                self._stats["valid"] += 1

    def _find_tokens_for(self, formula: str, parsed: list[tuple[str, torch.Tensor]]) -> torch.Tensor | None:
        for f, tids in parsed:
            if f == formula:
                return tids
        return None

    def _decode_with_mask(self, all_logits: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        """自回归解码 + 动作掩码, 使用 Generator 的位置级 logit 作为先验."""
        B = all_logits.shape[0]
        token_ids = torch.full(
            (B, self.max_len),
            self.vocab.bos_id,
            dtype=torch.long,
            device=self.device,
        )
        stack_depths = torch.zeros(B, dtype=torch.long, device=self.device)
        arity_t = self.vocab.arity_tensor.to(self.device)
        is_op_t = self.vocab.is_operator_tensor.to(self.device)

        for step in range(self.max_len):
            logits = all_logits[:, step, :]
            mask = compute_action_mask_batch(
                stack_depths,
                step,
                self.max_len,
                self.vocab,
                self.device,
            )
            logits = logits + mask
            selected = logits.argmax(dim=-1)
            token_ids[:, step] = selected

            sel_arity = arity_t[selected]
            sel_is_op = is_op_t[selected]
            stack_depths = torch.where(
                sel_is_op,
                stack_depths - sel_arity + 1,
                stack_depths + 1,
            )

        results: list[tuple[str, torch.Tensor]] = []
        for i in range(B):
            ids = token_ids[i].cpu().tolist()
            dsl = rpn_to_dsl(ids, self.vocab)
            if dsl is not None:
                results.append((dsl, token_ids[i].clone()))
        return results

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def _train_predictor(self) -> None:
        """训练 Predictor 预测 IC."""
        all_tokens = torch.stack(self._token_lib).to(self.device)
        all_ic = torch.tensor(self._ic_lib, dtype=torch.float32, device=self.device)

        self.predictor.train()
        for _ in range(self.predictor_epochs):
            perm = torch.randperm(len(all_tokens), device=self.device)
            for start in range(0, len(perm), 128):
                idx = perm[start : start + 128]
                pred = self.predictor(all_tokens[idx])
                loss = F.mse_loss(pred, all_ic[idx])
                self.opt_p.zero_grad()
                loss.backward()
                self.opt_p.step()
        self._stats["p_loss"] = loss.item()

    def _train_generator(self) -> None:
        """冻结 Predictor, 训练 Generator: 最大化预测值 + 多样性."""
        self.predictor.eval()
        for p in self.predictor.parameters():
            p.requires_grad_(False)

        self.generator.train()
        last_loss = 0.0
        for _ in range(self.generator_epochs):
            z1 = torch.randn(self.gen_batch, self.z_dim, device=self.device)
            z2 = torch.randn(self.gen_batch, self.z_dim, device=self.device)

            soft1 = F.gumbel_softmax(
                self.generator(z1),
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            )
            soft2 = F.gumbel_softmax(
                self.generator(z2),
                tau=self.gumbel_tau,
                hard=True,
                dim=-1,
            )

            # 适应性损失: 最大化代理模型预测
            fitness_loss = -self.predictor.forward_soft(soft1).mean()

            # 多样性损失: 惩罚两组采样的相似度 (Eq. 4 in paper)
            cos_sim = F.cosine_similarity(
                soft1.flatten(1),
                soft2.flatten(1),
                dim=-1,
            )
            diversity_loss = cos_sim.mean()

            loss = fitness_loss + self.lambda_div * diversity_loss
            self.opt_g.zero_grad()
            loss.backward()
            self.opt_g.step()
            last_loss = loss.item()

        for p in self.predictor.parameters():
            p.requires_grad_(True)
        self._stats["g_loss"] = last_loss

    # ------------------------------------------------------------------
    # 生成
    # ------------------------------------------------------------------

    def _generate_formulas(self) -> list[str]:
        """从训练好的 Generator 生成公式."""
        self.generator.eval()
        formulas: list[str] = []
        with torch.no_grad():
            for _ in range(4):
                z = torch.randn(self.gen_batch, self.z_dim, device=self.device)
                logits = self.generator(z)
                parsed = self._decode_with_mask(logits)
                formulas.extend(f for f, _ in parsed)
        return list(dict.fromkeys(formulas))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
from .registry import StrategyInfra, register_strategy  # noqa: E402


@register_strategy(AlphaForgeStrategy.meta)
def _build_alpha_forge(infra: StrategyInfra):
    return AlphaForgeStrategy(
        compiler=infra.compiler,
        vm=infra.vm,
        schema=infra.schema,
        registry=infra.registry,
    )
