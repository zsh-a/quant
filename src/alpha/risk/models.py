from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import numba
except Exception:  # pragma: no cover - numba is optional for acceleration
    numba = None

try:
    import torch
except Exception:  # pragma: no cover - torch is optional for this scaffold
    torch = None


ArrayLike = Any


if numba is not None:

    @numba.njit(cache=True)
    def _apply_turnover_limit_numba(target_weights: np.ndarray, max_turnover_per_bar: float) -> np.ndarray:
        rows, cols = target_weights.shape
        out = np.empty_like(target_weights)
        if rows == 0:
            return out
        for col in range(cols):
            out[0, col] = target_weights[0, col]
        upper = max_turnover_per_bar
        lower = -max_turnover_per_bar
        for row in range(1, rows):
            for col in range(cols):
                prev = out[row - 1, col]
                delta = target_weights[row, col] - prev
                if delta > upper:
                    delta = upper
                elif delta < lower:
                    delta = lower
                out[row, col] = prev + delta
        return out


@dataclass
class MarketContext:
    liquidity_mask: ArrayLike | None = None
    session_mask: ArrayLike | None = None
    max_abs_weight: float = 1.0
    max_turnover_per_bar: float = 0.25


@dataclass
class CostModel:
    maker_fee_bps: float = 2.0
    taker_fee_bps: float = 5.0
    slippage_bps: float = 3.0
    funding_bps_per_event: float = 1.0
    spread_weight: float = 0.5
    impact_coefficient_bps: float = 1.5

    def estimate_fee(self, turnover: ArrayLike) -> ArrayLike:
        return self._abs(turnover) * (self.taker_fee_bps / 10000.0)

    def estimate_slippage(
        self,
        turnover: ArrayLike,
        spread: ArrayLike | None = None,
        mid_price: ArrayLike | None = None,
    ) -> ArrayLike:
        total_cost = self._abs(turnover) * (self.slippage_bps / 10000.0)
        if spread is not None and mid_price is not None:
            spread_ratio = self._abs(spread) / (self._abs(mid_price) + 1e-12)
            total_cost = total_cost + (self._abs(turnover) * spread_ratio * self.spread_weight)
            total_cost = total_cost + (self._square(self._abs(turnover)) * (self.impact_coefficient_bps / 10000.0))
        if self._ndim(total_cost) > 1:
            return self._sum(total_cost, axis=1)
        return total_cost

    def estimate_funding(self, weights: ArrayLike, funding_rate: ArrayLike | None = None) -> ArrayLike:
        if funding_rate is None:
            return self._abs(weights) * (self.funding_bps_per_event / 10000.0)
        return self._abs(weights) * funding_rate

    def _abs(self, value: ArrayLike) -> ArrayLike:
        if torch is not None and isinstance(value, torch.Tensor):
            return torch.abs(value)
        return np.abs(value)

    def _square(self, value: ArrayLike) -> ArrayLike:
        if torch is not None and isinstance(value, torch.Tensor):
            return torch.square(value)
        return np.square(value)

    def _sum(self, value: ArrayLike, axis: int) -> ArrayLike:
        if torch is not None and isinstance(value, torch.Tensor):
            return torch.nansum(value, dim=axis)
        return np.nansum(value, axis=axis)

    def _ndim(self, value: ArrayLike) -> int:
        if torch is not None and isinstance(value, torch.Tensor):
            return int(value.ndim)
        return int(np.ndim(value))


@dataclass
class BacktestResult:
    weights: ArrayLike
    gross_returns: ArrayLike
    net_returns: ArrayLike
    turnover: ArrayLike
    equity_curve: ArrayLike

    def summary(self) -> dict[str, float]:
        net_returns = _to_numpy(self.net_returns)
        turnover = _to_numpy(self.turnover)
        equity_curve = _to_numpy(self.equity_curve)
        mean = float(np.nanmean(net_returns)) if net_returns.size else 0.0
        std = float(np.nanstd(net_returns)) if net_returns.size else 0.0
        sharpe = mean / (std + 1e-12)
        final_equity = float(equity_curve[-1]) if equity_curve.size else 1.0
        total_return = final_equity - 1.0  # geometric return from equity curve
        avg_turnover = float(np.nanmean(turnover)) if turnover.size else 0.0
        peak = np.maximum.accumulate(equity_curve)
        drawdown = np.where(peak > 1e-12, 1.0 - equity_curve / peak, 0.0)
        return {
            "sharpe": sharpe,
            "total_return": total_return,
            "avg_turnover": avg_turnover,
            "volatility": std,
            "max_drawdown": float(np.clip(np.nanmax(drawdown), 0.0, 1.0)) if drawdown.size else 0.0,
            "final_equity": final_equity,
        }


class SignalTransformer:
    def to_target_weights(self, alpha: ArrayLike, market_ctx: MarketContext) -> ArrayLike:
        if _is_torch(alpha):
            return self._to_target_weights_torch(alpha, market_ctx)
        return self._to_target_weights_numpy(alpha, market_ctx)

    def _to_target_weights_numpy(self, alpha: ArrayLike, market_ctx: MarketContext) -> np.ndarray:
        scores = _to_numpy(alpha).astype(float).copy()
        if market_ctx.liquidity_mask is not None:
            scores = np.where(np.asarray(market_ctx.liquidity_mask, dtype=bool), scores, np.nan)
        if market_ctx.session_mask is not None:
            scores = np.where(np.asarray(market_ctx.session_mask, dtype=bool), scores, 0.0)

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

    def _to_target_weights_torch(self, alpha: Any, market_ctx: MarketContext) -> Any:
        scores = alpha.clone()
        if market_ctx.liquidity_mask is not None:
            liquidity_mask = _as_torch(market_ctx.liquidity_mask, like=alpha, dtype=torch.bool)
            scores = torch.where(liquidity_mask, scores, torch.full_like(scores, torch.nan))
        if market_ctx.session_mask is not None:
            session_mask = _as_torch(market_ctx.session_mask, like=alpha, dtype=torch.bool)
            scores = torch.where(session_mask, scores, torch.zeros_like(scores))

        valid = ~torch.isnan(scores)
        valid_counts = valid.sum(dim=1, keepdim=True)
        row_means = torch.where(
            valid,
            scores,
            torch.zeros_like(scores),
        ).sum(dim=1, keepdim=True) / valid_counts.clamp(min=1).to(dtype=scores.dtype)
        centered = scores - row_means
        empty_rows = (valid_counts.squeeze(dim=1) == 0).unsqueeze(dim=1)
        centered = torch.where(empty_rows, torch.zeros_like(centered), centered)
        denom = torch.where(
            torch.isnan(centered),
            torch.zeros_like(centered),
            torch.abs(centered),
        ).sum(dim=1, keepdim=True)
        weights = centered / (denom + 1e-12)
        weights = torch.where(torch.isnan(weights), torch.zeros_like(weights), weights)
        return torch.clamp(weights, -market_ctx.max_abs_weight, market_ctx.max_abs_weight)


class RuleOverlay:
    def apply(self, target_weights: ArrayLike, market_ctx: MarketContext) -> ArrayLike:
        if _is_torch(target_weights):
            return self._apply_torch(target_weights, market_ctx)
        return self._apply_numpy(target_weights, market_ctx)

    def _apply_numpy(self, target_weights: ArrayLike, market_ctx: MarketContext) -> np.ndarray:
        weights = np.asarray(target_weights, dtype=float)
        original_shape = weights.shape
        if weights.ndim == 1:
            weights = weights.reshape(-1, 1)
            reshaped = True
        else:
            reshaped = False
        if numba is not None and weights.ndim == 2:
            limited = _apply_turnover_limit_numba(
                np.ascontiguousarray(weights),
                float(market_ctx.max_turnover_per_bar),
            )
            return limited.reshape(original_shape) if reshaped else limited
        weights = weights.copy()
        for idx in range(1, weights.shape[0]):
            prev = weights[idx - 1]
            delta = np.clip(
                weights[idx] - prev,
                -market_ctx.max_turnover_per_bar,
                market_ctx.max_turnover_per_bar,
            )
            weights[idx] = prev + delta
        return weights.reshape(original_shape) if reshaped else weights

    def _apply_torch(self, target_weights: Any, market_ctx: MarketContext) -> Any:
        weights = target_weights.clone()
        for idx in range(1, weights.shape[0]):
            prev = weights[idx - 1]
            delta = torch.clamp(
                weights[idx] - prev,
                -market_ctx.max_turnover_per_bar,
                market_ctx.max_turnover_per_bar,
            )
            weights[idx] = prev + delta
        return weights


@dataclass
class RiskConfig:
    """Portfolio-level risk management configuration."""

    vol_target: float = 0.15
    """Annualized target volatility. 0 = disabled."""

    vol_lookback: int = 60
    """Bars for realized vol estimation."""

    max_drawdown: float = 0.15
    """Drawdown threshold for proportional deleveraging. 0 = disabled."""

    trailing_stop_pct: float = 0.05
    """Trailing stop on equity curve. 0 = disabled."""

    trailing_stop_cooldown: int = 12
    """Bars to stay flat after trailing stop triggers."""

    bars_per_year: float = 365.25 * 24 * 12
    """Annualization factor (default: 5-minute bars)."""


class PortfolioManager:
    """
    Dynamic position management layer.

    Sits between RuleOverlay and ExecutionSimulator.  Walks forward bar-by-bar,
    computing a virtual equity curve from weights × forward returns, and applies:

    1. **Volatility targeting** — scale weights so the portfolio's rolling realised
       vol matches ``vol_target``.
    2. **Drawdown control** — linearly deleverage when cumulative drawdown from
       peak exceeds ``max_drawdown``.
    3. **Trailing stop** — zero all weights when equity drops more than
       ``trailing_stop_pct`` from peak; stay flat for ``cooldown`` bars.
    """

    def apply(
        self,
        weights: np.ndarray,
        close: np.ndarray,
        config: RiskConfig,
    ) -> np.ndarray:
        weights = np.asarray(weights, dtype=float).copy()
        close = np.asarray(close, dtype=float)
        n_time = weights.shape[0]
        if n_time < 2:
            return weights

        # Forward returns for virtual equity tracking
        fwd = np.zeros_like(close)
        fwd[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
        fwd = np.clip(fwd, -0.5, 0.5)

        # Pre-compute rolling realised vol (annualised) for vol targeting
        vol_scale = np.ones(n_time, dtype=float)
        if config.vol_target > 0 and config.vol_lookback > 1:
            port_ret = np.nansum(weights * fwd, axis=1)
            vol_scale = self._rolling_vol_scale(port_ret, config)

        # Walk forward: apply vol scaling, then drawdown + trailing stop
        equity = 1.0
        peak_equity = 1.0
        stop_cooldown = 0

        for t in range(n_time):
            # 1) Volatility scaling
            weights[t] *= vol_scale[t]

            # 2) Trailing stop (higher priority — full exit)
            if stop_cooldown > 0:
                weights[t] = 0.0
                stop_cooldown -= 1
            elif config.trailing_stop_pct > 0:
                dd = 1.0 - equity / (peak_equity + 1e-12)
                if dd > config.trailing_stop_pct:
                    weights[t] = 0.0
                    stop_cooldown = config.trailing_stop_cooldown

            # 3) Drawdown control (proportional deleveraging)
            if config.max_drawdown > 0 and stop_cooldown == 0:
                dd = 1.0 - equity / (peak_equity + 1e-12)
                if dd > config.max_drawdown * 0.5:
                    lever = max(1.0 - dd / config.max_drawdown, 0.0)
                    weights[t] *= lever

            # Update virtual equity
            bar_ret = float(np.nansum(weights[t] * fwd[t]))
            equity *= (1.0 + bar_ret)
            if equity > peak_equity:
                peak_equity = equity

        return weights

    def _rolling_vol_scale(self, port_returns: np.ndarray, config: RiskConfig) -> np.ndarray:
        """Compute per-bar scaling factor: vol_target / realized_vol."""
        n = len(port_returns)
        scale = np.ones(n, dtype=float)
        lookback = config.vol_lookback
        ann_factor = np.sqrt(config.bars_per_year)

        for t in range(lookback, n):
            window = port_returns[t - lookback : t]
            realised_vol = float(np.nanstd(window)) * ann_factor
            if realised_vol > 1e-6:
                raw = config.vol_target / realised_vol
                scale[t] = np.clip(raw, 0.1, 3.0)  # prevent extreme leverage
        return scale


class ExecutionSimulator:
    def simulate(
        self,
        weights: ArrayLike,
        prices: dict[str, ArrayLike],
        cost_model: CostModel,
        funding_rate: ArrayLike | None = None,
    ) -> BacktestResult:
        if _is_torch(weights) or any(_is_torch(value) for value in prices.values()) or _is_torch(funding_rate):
            return self._simulate_torch(weights, prices, cost_model, funding_rate)
        return self._simulate_numpy(weights, prices, cost_model, funding_rate)

    def _simulate_numpy(
        self,
        weights: ArrayLike,
        prices: dict[str, ArrayLike],
        cost_model: CostModel,
        funding_rate: ArrayLike | None = None,
    ) -> BacktestResult:
        weights_np = np.asarray(weights, dtype=float)
        close = np.asarray(prices["close"], dtype=float)
        forward_returns = np.zeros_like(close)
        forward_returns[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
        forward_returns = np.clip(forward_returns, -0.5, 0.5)  # cap per-bar returns

        position_returns = np.clip(np.nansum(weights_np * forward_returns, axis=1), -0.5, 0.5)
        trade_sizes = np.abs(np.diff(weights_np, axis=0, prepend=np.zeros_like(weights_np[:1])))
        turnover = np.nansum(trade_sizes, axis=1)
        fees = cost_model.estimate_fee(turnover)
        spread = None if "bid_ask_spread" not in prices else np.asarray(prices["bid_ask_spread"], dtype=float)
        slippage = cost_model.estimate_slippage(trade_sizes, spread=spread, mid_price=close)
        funding = np.nansum(cost_model.estimate_funding(weights_np, funding_rate), axis=1)
        net_returns = position_returns - fees - slippage - funding
        equity_curve = np.cumprod(1.0 + np.nan_to_num(net_returns, nan=0.0))

        return BacktestResult(
            weights=weights_np,
            gross_returns=position_returns,
            net_returns=net_returns,
            turnover=turnover,
            equity_curve=equity_curve,
        )

    def _simulate_torch(
        self,
        weights: ArrayLike,
        prices: dict[str, ArrayLike],
        cost_model: CostModel,
        funding_rate: ArrayLike | None = None,
    ) -> BacktestResult:
        close = _as_torch(prices["close"], like=weights)
        weights_t = _as_torch(weights, like=close)
        forward_returns = torch.zeros_like(close)
        forward_returns[:-1] = close[1:] / (close[:-1] + 1e-12) - 1.0
        forward_returns = torch.clamp(forward_returns, -0.5, 0.5)

        position_returns = torch.clamp(torch.nansum(weights_t * forward_returns, dim=1), -0.5, 0.5)
        trade_sizes = torch.abs(weights_t - torch.cat([torch.zeros_like(weights_t[:1]), weights_t[:-1]], dim=0))
        turnover = torch.nansum(trade_sizes, dim=1)
        fees = cost_model.estimate_fee(turnover)
        spread_tensor = None if "bid_ask_spread" not in prices else _as_torch(prices["bid_ask_spread"], like=weights_t)
        slippage = cost_model.estimate_slippage(trade_sizes, spread=spread_tensor, mid_price=close)
        funding_tensor = None if funding_rate is None else _as_torch(funding_rate, like=weights_t)
        funding = torch.nansum(cost_model.estimate_funding(weights_t, funding_tensor), dim=1)
        net_returns = position_returns - fees - slippage - funding
        equity_curve = torch.cumprod(1.0 + torch.nan_to_num(net_returns, nan=0.0), dim=0)

        return BacktestResult(
            weights=weights_t,
            gross_returns=position_returns,
            net_returns=net_returns,
            turnover=turnover,
            equity_curve=equity_curve,
        )


def _is_torch(value: Any) -> bool:
    return bool(torch is not None and isinstance(value, torch.Tensor))


def _as_torch(value: ArrayLike, like: Any, dtype: Any | None = None) -> Any:
    if torch is None:
        raise RuntimeError("Torch backend requested but torch is not available")
    if isinstance(value, torch.Tensor):
        tensor = value.to(device=like.device)
        if dtype is not None and tensor.dtype != dtype:
            return tensor.to(dtype=dtype)
        return tensor
    array = np.asarray(value)
    target_dtype = dtype
    if target_dtype is None:
        target_dtype = torch.bool if array.dtype == np.bool_ else like.dtype
    return torch.as_tensor(array, device=like.device, dtype=target_dtype)


def _to_numpy(value: ArrayLike) -> np.ndarray:
    from ..core.vm import to_numpy
    return to_numpy(value)
