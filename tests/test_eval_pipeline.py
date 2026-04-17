"""端到端验证因子评测链路正确性。

用 6 天 × 4 只股票的合成数据，手工计算每个环节的预期值，
逐步对比：forward_returns → IC → signal→weights → turnover → cost → equity。
"""

import numpy as np

from src.alpha.core.dataset import AlphaDataset
from src.alpha.eval.metrics import compute_forward_returns, compute_ic_metrics, compute_quantile_returns
from src.alpha.risk.models import (
    CostModel,
    ExecutionSimulator,
    MarketContext,
    RuleOverlay,
    SignalTransformer,
)

# ---------------------------------------------------------------------------
# 合成数据：6 天 × 4 只股票，价格设计为可手算
# ---------------------------------------------------------------------------
#         S0     S1     S2     S3
CLOSE = np.array(
    [
        [10.0, 20.0, 30.0, 40.0],  # t0
        [11.0, 19.0, 33.0, 38.0],  # t1  涨10% 跌5% 涨10% 跌5%
        [11.0, 19.0, 33.0, 38.0],  # t2  平
        [12.1, 17.1, 36.3, 34.2],  # t3  涨10% 跌10% 涨10% 跌10%
        [12.1, 17.1, 36.3, 34.2],  # t4  平
        [13.31, 15.39, 39.93, 30.78],  # t5  涨10% 跌10% 涨10% 跌10%
    ],
    dtype=np.float32,
)
T, S = CLOSE.shape  # 6, 4


class TestForwardReturns:
    """Step 1: 验证 forward returns = close[t+1]/close[t] - 1"""

    def test_basic(self):
        fwd = compute_forward_returns(CLOSE, periods=1)
        # t0→t1: S0 涨10%, S1 跌5%, S2 涨10%, S3 跌5%
        np.testing.assert_allclose(fwd[0, 0], 0.10, atol=1e-4)
        np.testing.assert_allclose(fwd[0, 1], -0.05, atol=1e-4)
        np.testing.assert_allclose(fwd[0, 2], 0.10, atol=1e-4)
        np.testing.assert_allclose(fwd[0, 3], -0.05, atol=1e-4)
        # t1→t2: 全平
        np.testing.assert_allclose(fwd[1], [0, 0, 0, 0], atol=1e-4)
        # 最后一行应为 nan
        assert np.all(np.isnan(fwd[-1]))

    def test_shape(self):
        fwd = compute_forward_returns(CLOSE, periods=1)
        assert fwd.shape == CLOSE.shape

    def test_multi_period(self):
        fwd5 = compute_forward_returns(CLOSE, periods=2)
        # t0→t2: close相同为t1
        expected = CLOSE[2] / CLOSE[0] - 1.0
        np.testing.assert_allclose(fwd5[0], expected, atol=1e-4)
        # 最后两行应为 nan
        assert np.all(np.isnan(fwd5[-2:]))


class TestSignalTransformer:
    """Step 2: 验证 alpha → L1 归一化权重"""

    def setup_method(self):
        self.transformer = SignalTransformer()
        self.ctx = MarketContext()  # 无 mask，全部可交易

    def test_l1_normalization(self):
        # 简单 alpha: [1, 2, 3, 4]
        alpha = np.array([[1.0, 2.0, 3.0, 4.0]])  # (1, 4)
        weights = self.transformer.to_target_weights(alpha, self.ctx)
        # mean = 2.5, centered = [-1.5, -0.5, 0.5, 1.5]
        # L1 = 4.0, weights = [-0.375, -0.125, 0.125, 0.375]
        expected = np.array([[-0.375, -0.125, 0.125, 0.375]])
        np.testing.assert_allclose(weights, expected, atol=1e-6)

    def test_weights_sum_near_zero(self):
        """多空对冲权重之和应接近 0"""
        alpha = np.array([[10.0, 20.0, 30.0, 40.0]])
        weights = self.transformer.to_target_weights(alpha, self.ctx)
        assert abs(weights.sum()) < 1e-6

    def test_liquidity_mask(self):
        """被 mask 的股票权重为 0 或 nan"""
        alpha = np.array([[1.0, 2.0, 3.0, 4.0]])
        mask = np.array([[True, True, False, True]])  # S2 不可交易
        ctx = MarketContext(liquidity_mask=mask)
        weights = self.transformer.to_target_weights(alpha, ctx)
        # S2 被排除，剩下 S0,S1,S3: mean=7/3≈2.333
        assert np.isnan(weights[0, 2]) or abs(weights[0, 2]) < 1e-9


class TestRuleOverlay:
    """Step 3: 验证换手约束"""

    def test_turnover_limit(self):
        overlay = RuleOverlay()
        ctx = MarketContext(max_turnover_per_bar=0.1)
        # t0: 目标 [0.5, -0.5], t1: 目标 [-0.5, 0.5]（完全反转）
        target = np.array([[0.5, -0.5], [-0.5, 0.5]])
        result = overlay.apply(target, ctx)
        # t0 不变（第一行）
        np.testing.assert_allclose(result[0], [0.5, -0.5])
        # t1: delta = [-1, 1], clip to [-0.1, 0.1]
        # result[1] = [0.5-0.1, -0.5+0.1] = [0.4, -0.4]
        np.testing.assert_allclose(result[1], [0.4, -0.4], atol=1e-9)


class TestCostModel:
    """Step 4: 验证费用计算"""

    def test_a_share_costs(self):
        # A股参数
        cost = CostModel(
            taker_fee_bps=1.0,
            slippage_bps=1.5,
            funding_bps_per_event=0.0,
        )
        turnover = np.array([0.5])  # 换手 50%
        fee = cost.estimate_fee(turnover)
        # 0.5 * 1/10000 = 0.00005
        np.testing.assert_allclose(fee, 0.00005)

        slippage = cost.estimate_slippage(turnover, spread=None, mid_price=None)
        # 0.5 * 1.5/10000 = 0.000075
        np.testing.assert_allclose(slippage, 0.000075)

    def test_no_funding_for_a_share(self):
        cost = CostModel(funding_bps_per_event=0.0)
        weights = np.array([[0.5, -0.5]])
        funding = cost.estimate_funding(weights, funding_rate=None)
        # funding_bps = 0 → 全为 0
        np.testing.assert_allclose(funding, 0.0)


class TestExecutionSimulator:
    """Step 5: 验证回测模拟器端到端"""

    def test_known_returns(self):
        """用已知权重和价格手算净收益"""
        close = np.array([[100.0, 100.0], [110.0, 90.0]], dtype=np.float64)
        # t0→t1: S0涨10% S1跌10%
        # 权重: 全部做多 S0
        weights = np.array([[1.0, 0.0], [1.0, 0.0]])
        cost = CostModel(taker_fee_bps=0, slippage_bps=0, funding_bps_per_event=0)
        sim = ExecutionSimulator()
        result = sim.simulate(weights, {"close": close}, cost, funding_rate=None)

        # gross = w0*r0 + w1*r1 = 1*0.1 + 0*(-0.1) = 0.1
        np.testing.assert_allclose(result.gross_returns[0], 0.1, atol=1e-6)
        # 最后一行 forward return = 0
        np.testing.assert_allclose(result.gross_returns[1], 0.0, atol=1e-6)

    def test_cost_deduction(self):
        """验证费用被正确扣除"""
        close = np.array([[100.0], [110.0]], dtype=np.float64)
        weights = np.array([[1.0], [1.0]])
        cost = CostModel(taker_fee_bps=100, slippage_bps=0, funding_bps_per_event=0)
        sim = ExecutionSimulator()
        result = sim.simulate(weights, {"close": close}, cost, funding_rate=None)

        # turnover[0] = |1-0| = 1.0, fee = 1.0 * 100/10000 = 0.01
        # gross = 0.1, net = 0.1 - 0.01 = 0.09
        np.testing.assert_allclose(result.net_returns[0], 0.09, atol=1e-6)

    def test_equity_curve(self):
        """验证 equity curve 是 net returns 的累积乘积"""
        close = np.array(
            [[100.0, 100.0], [110.0, 90.0], [121.0, 81.0]],
            dtype=np.float64,
        )
        weights = np.array([[0.5, -0.5], [0.5, -0.5], [0.5, -0.5]])
        cost = CostModel(taker_fee_bps=0, slippage_bps=0, funding_bps_per_event=0)
        sim = ExecutionSimulator()
        result = sim.simulate(weights, {"close": close}, cost, funding_rate=None)

        # t0: gross = 0.5*0.1 + (-0.5)*(-0.1) = 0.1
        np.testing.assert_allclose(result.gross_returns[0], 0.1, atol=1e-6)
        # equity = cumprod(1 + net_returns)
        expected_eq = np.cumprod(1.0 + result.net_returns)
        np.testing.assert_allclose(result.equity_curve, expected_eq, atol=1e-9)


class TestICMetrics:
    """Step 6: 验证 IC 计算"""

    def test_perfect_positive_ic(self):
        """alpha 完全预测未来收益方向 → IC ≈ 1"""
        # alpha 和 forward_return 完美正相关
        alpha = np.array(
            [[1.0, 2.0, 3.0, 4.0]] * 20,
            dtype=np.float64,
        )
        # 构造 close 使得高 alpha 的股票涨、低 alpha 的跌
        close = np.ones((21, 4), dtype=np.float64) * 100
        for t in range(20):
            close[t + 1] = close[t] * (1 + np.array([-0.02, -0.01, 0.01, 0.02]))

        metrics = compute_ic_metrics(alpha, close[:20], fwd_windows=[1])
        assert metrics["rank_ic"] > 0.9, f"Expected IC > 0.9, got {metrics['rank_ic']}"

    def test_zero_ic_random(self):
        """随机 alpha 和不相关收益 → IC ≈ 0"""
        rng = np.random.default_rng(42)
        alpha = rng.standard_normal((100, 20))
        close = np.cumprod(1.0 + rng.standard_normal((100, 20)) * 0.01, axis=0)
        metrics = compute_ic_metrics(alpha, close, fwd_windows=[1])
        assert abs(metrics["rank_ic"]) < 0.15, f"Expected IC ≈ 0, got {metrics['rank_ic']}"


class TestQuantileReturns:
    """Step 7: 验证分层回测"""

    def test_monotonicity(self):
        """好因子应有单调递增的分组收益"""
        T = 100
        S = 50
        rng = np.random.default_rng(123)
        # alpha 就是真实的未来收益方向（加噪声）
        true_signal = rng.standard_normal((T, S))
        alpha = true_signal + rng.standard_normal((T, S)) * 0.3
        close = np.ones((T + 1, S)) * 100.0
        for t in range(T):
            close[t + 1] = close[t] * (1 + true_signal[t] * 0.01)

        result = compute_quantile_returns(alpha, close[:T], n_quantiles=5)
        stats = result["quantile_stats"]

        # Q5 收益应 > Q1
        assert stats[4]["total_return"] > stats[0]["total_return"]
        # 单调性 > 0.5
        assert result["monotonicity"] > 0.5


class TestEndToEnd:
    """完整链路: alpha → weights → backtest → metrics"""

    def test_pipeline_consistency(self):
        """验证 AlphaService 的各环节串联后数值一致"""
        T = 50
        S = 10
        rng = np.random.default_rng(0)

        close = np.cumprod(1 + rng.standard_normal((T, S)) * 0.02, axis=0).astype(np.float32) * 100
        alpha = rng.standard_normal((T, S)).astype(np.float32)

        fields = {
            "close": close,
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "volume": np.ones_like(close),
        }
        AlphaDataset(
            interval="1d",
            symbols=[f"S{i}" for i in range(S)],
            timestamps=[f"2025-01-{d + 1:02d}" for d in range(T)],
            fields=fields,
            liquidity_mask=np.ones((T, S), dtype=bool),
            session_mask=np.ones((T, S), dtype=bool),
        )

        # 手动执行各步骤
        ctx = MarketContext()
        transformer = SignalTransformer()
        overlay = RuleOverlay()
        cost = CostModel(taker_fee_bps=1.0, slippage_bps=1.5, funding_bps_per_event=0.0)
        sim = ExecutionSimulator()

        weights = transformer.to_target_weights(alpha, ctx)
        weights = overlay.apply(weights, ctx)
        result = sim.simulate(weights, {"close": close}, cost, funding_rate=None)

        # 基本不变量检查
        assert result.equity_curve.shape == (T,)
        assert result.turnover.shape == (T,)
        assert np.all(np.isfinite(result.equity_curve))
        assert np.all(result.turnover >= 0)

        # L1 归一化后 turnover overlay 可能轻微超 1（不重新归一化）
        abs_sum = np.nansum(np.abs(weights), axis=1)
        assert np.all(abs_sum <= 1.05), f"Max weight L1 = {abs_sum.max()}"

        # 净收益 ≤ 毛收益（有费用时）
        has_trade = result.turnover > 1e-9
        assert np.all(result.net_returns[has_trade] <= result.gross_returns[has_trade] + 1e-9)

        # IC 应可计算
        metrics = compute_ic_metrics(alpha, close, fwd_windows=[1])
        assert "rank_ic" in metrics
        assert np.isfinite(metrics["rank_ic"])
