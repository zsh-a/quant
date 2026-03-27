from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class MarketContext:
    liquidity_mask: np.ndarray | None = None
    session_mask: np.ndarray | None = None
    max_abs_weight: float = 1.0
    max_turnover_per_bar: float = 0.25


@dataclass
class CostModel:
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 3.0
    funding_bps_per_event: float = 1.0

    def estimate_fee(self, turnover: np.ndarray) -> np.ndarray:
        return np.abs(turnover) * (self.taker_fee_bps / 10000.0)

    def estimate_slippage(self, turnover: np.ndarray) -> np.ndarray:
        return np.abs(turnover) * (self.slippage_bps / 10000.0)

    def estimate_funding(self, weights: np.ndarray, funding_rate: np.ndarray | None = None) -> np.ndarray:
        if funding_rate is None:
            return np.abs(weights) * (self.funding_bps_per_event / 10000.0)
        return np.abs(weights) * funding_rate


@dataclass
class BacktestResult:
    weights: np.ndarray
    gross_returns: np.ndarray
    net_returns: np.ndarray
    turnover: np.ndarray
    equity_curve: np.ndarray

    def summary(self) -> dict[str, float]:
        mean = float(np.nanmean(self.net_returns)) if self.net_returns.size else 0.0
        std = float(np.nanstd(self.net_returns)) if self.net_returns.size else 0.0
        sharpe = mean / (std + 1e-12)
        total_return = float(np.nansum(self.net_returns))
        avg_turnover = float(np.nanmean(self.turnover)) if self.turnover.size else 0.0
        drawdown = 1.0 - np.divide(
            self.equity_curve,
            np.maximum.accumulate(self.equity_curve),
        )
        return {
            "sharpe": sharpe,
            "total_return": total_return,
            "avg_turnover": avg_turnover,
            "volatility": std,
            "max_drawdown": float(np.nanmax(drawdown)) if drawdown.size else 0.0,
            "final_equity": float(self.equity_curve[-1]) if self.equity_curve.size else 1.0,
        }


class SignalTransformer:
    def to_target_weights(self, alpha: np.ndarray, market_ctx: MarketContext) -> np.ndarray:
        scores = np.asarray(alpha, dtype=float).copy()
        if market_ctx.liquidity_mask is not None:
            scores = np.where(market_ctx.liquidity_mask, scores, np.nan)
        if market_ctx.session_mask is not None:
            scores = np.where(market_ctx.session_mask, scores, 0.0)

        valid_counts = np.sum(~np.isnan(scores), axis=1, keepdims=True)
        row_means = np.divide(
            np.nansum(scores, axis=1, keepdims=True),
            np.maximum(valid_counts, 1),
        )
        centered = scores - row_means
        centered[valid_counts.squeeze(axis=1) == 0] = 0.0
        denom = np.nansum(np.abs(centered), axis=1, keepdims=True)
        weights = centered / (denom + 1e-12)
        return np.clip(weights, -market_ctx.max_abs_weight, market_ctx.max_abs_weight)


class RuleOverlay:
    def apply(self, target_weights: np.ndarray, market_ctx: MarketContext) -> np.ndarray:
        weights = np.asarray(target_weights, dtype=float).copy()
        for idx in range(1, weights.shape[0]):
            prev = weights[idx - 1]
            delta = np.clip(
                weights[idx] - prev,
                -market_ctx.max_turnover_per_bar,
                market_ctx.max_turnover_per_bar,
            )
            weights[idx] = prev + delta
        return weights


class ExecutionSimulator:
    def simulate(
        self,
        weights: np.ndarray,
        prices: dict[str, np.ndarray],
        cost_model: CostModel,
        funding_rate: np.ndarray | None = None,
    ) -> BacktestResult:
        close = np.asarray(prices["close"], dtype=float)
        forward_returns = np.zeros_like(close)
        forward_returns[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0

        position_returns = np.nansum(weights * forward_returns, axis=1)
        turnover = np.nansum(np.abs(np.diff(weights, axis=0, prepend=np.zeros_like(weights[:1]))), axis=1)
        fees = cost_model.estimate_fee(turnover)
        slippage = cost_model.estimate_slippage(turnover)
        funding = np.nansum(cost_model.estimate_funding(weights, funding_rate), axis=1)
        net_returns = position_returns - fees - slippage - funding
        equity_curve = np.cumprod(1.0 + np.nan_to_num(net_returns, nan=0.0))

        return BacktestResult(
            weights=weights,
            gross_returns=position_returns,
            net_returns=net_returns,
            turnover=turnover,
            equity_curve=equity_curve,
        )
