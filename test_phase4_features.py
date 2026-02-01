#!/usr/bin/env python3
"""
Test script for Phase 4 advanced features.
"""

import sys
sys.path.insert(0, '.')

from loguru import logger


def test_portfolio_manager():
    """Test PortfolioManager"""
    print("\n" + "="*50)
    print("Testing PortfolioManager")
    print("="*50)
    
    from src.portfolio.portfolio_manager import (
        PortfolioManager,
        StrategyConfig,
        WeightMethod
    )
    
    # Create mock strategy class
    class MockStrategy:
        def __init__(self, **params):
            self.params = params
        
        def on_bar(self, bar):
            return {
                'symbol': bar.get('symbol'),
                'direction': 'buy' if bar.get('close', 0) > 10 else 'hold',
                'strength': 0.8
            }
    
    # Create configs
    configs = [
        StrategyConfig(name='strat_a', strategy_class=MockStrategy, params={'x': 1}),
        StrategyConfig(name='strat_b', strategy_class=MockStrategy, params={'x': 2}),
    ]
    
    # Test equal weights
    pm = PortfolioManager(configs, weight_method=WeightMethod.EQUAL)
    weights = pm.get_weights()
    
    assert len(weights) == 2, "Should have 2 strategies"
    assert abs(weights['strat_a'] - 0.5) < 0.01, "Equal weights should be 0.5"
    print(f"✅ Equal weights: {weights}")
    
    # Test signal collection
    bar = {'symbol': 'TEST', 'close': 15.0}
    signals = pm.collect_signals(bar)
    assert len(signals) == 2, "Should collect 2 signals"
    print(f"✅ Signal collection: {len(signals)} signals")
    
    # Test signal combination
    combined = pm.combine_signals(signals)
    print(f"✅ Signal combination: {len(combined)} combined signals")
    
    print("✅ PortfolioManager tests passed!")
    return True


def test_parameter_optimizer():
    """Test ParameterOptimizer"""
    print("\n" + "="*50)
    print("Testing ParameterOptimizer")
    print("="*50)
    
    from src.optimizer.optimizer import (
        ParameterOptimizer,
        ParamSpec,
        OptimizationObjective
    )
    
    # Create mock strategy
    class MockStrategy:
        def __init__(self, ma_short=5, ma_long=20):
            self.ma_short = ma_short
            self.ma_long = ma_long
    
    # Define param space
    param_space = {
        'ma_short': ParamSpec(name='ma_short', param_type='int', low=5, high=10, step=1),
        'ma_long': ParamSpec(name='ma_long', param_type='int', low=20, high=30, step=5)
    }
    
    optimizer = ParameterOptimizer(
        strategy_class=MockStrategy,
        param_space=param_space,
        objective=OptimizationObjective.MAX_SHARPE
    )
    
    # Mock backtest function
    def mock_backtest(params):
        # Simulate: smaller ma_short with larger ma_long is better
        score = (10 - params['ma_short']) * 0.1 + (params['ma_long'] - 20) * 0.05
        return {
            'sharpe_ratio': score,
            'total_return': score * 0.2,
            'max_drawdown': 0.1,
            'n_trades': 50
        }
    
    # Test grid search (small space)
    print("Running grid search...")
    report = optimizer.grid_search(mock_backtest, max_combinations=20)
    
    assert report.n_iterations > 0, "Should have some iterations"
    assert report.best_params is not None, "Should find best params"
    print(f"✅ Grid search: {report.n_iterations} iterations")
    print(f"   Best params: {report.best_params}")
    print(f"   Best score: {report.best_score:.4f}")
    
    # Test random search
    print("Running random search...")
    report = optimizer.random_search(mock_backtest, n_iterations=10)
    assert report.n_iterations == 10, "Should run 10 iterations"
    print(f"✅ Random search: {report.n_iterations} iterations")
    
    print("✅ ParameterOptimizer tests passed!")
    return True


def test_attribution():
    """Test Attribution Analysis"""
    print("\n" + "="*50)
    print("Testing Attribution Analysis")
    print("="*50)
    
    from src.analysis.attribution import ReturnAttribution, RiskAttribution
    
    # Mock data
    trades = [
        {'symbol': 'sz.300750', 'side': 'buy', 'quantity': 100, 'price': 10.0, 'timestamp': '2024-01-01'},
        {'symbol': 'sz.300750', 'side': 'sell', 'quantity': 100, 'price': 11.0, 'timestamp': '2024-01-15'},
        {'symbol': 'sz.000858', 'side': 'buy', 'quantity': 50, 'price': 20.0, 'timestamp': '2024-01-10'},
        {'symbol': 'sz.000858', 'side': 'sell', 'quantity': 50, 'price': 19.0, 'timestamp': '2024-01-20'},
    ]
    
    equity_history = [
        {'date': '2024-01-01', 'total_equity': 100000},
        {'date': '2024-01-15', 'total_equity': 100100},
        {'date': '2024-02-01', 'total_equity': 100050},
        {'date': '2024-02-15', 'total_equity': 100200},
    ]
    
    # Test return attribution
    attr = ReturnAttribution(trades, equity_history)
    result = attr.analyze()
    
    assert 'sz.300750' in result.by_asset, "Should have asset attribution"
    print(f"✅ By asset: {result.by_asset}")
    print(f"✅ By sector: {result.by_sector}")
    print(f"✅ Win rate: {result.win_rate:.1%}")
    
    # Test risk attribution
    risk_attr = RiskAttribution(equity_history, {})
    risk = risk_attr.analyze()
    
    assert 'volatility' in risk, "Should have volatility"
    print(f"✅ Volatility: {risk['volatility']:.2%}")
    print(f"✅ Max Drawdown: {risk['max_drawdown']:.2%}")
    
    print("✅ Attribution tests passed!")
    return True


def test_report_generator():
    """Test Report Generator"""
    print("\n" + "="*50)
    print("Testing Report Generator")
    print("="*50)
    
    from src.reports.generator import ReportGenerator
    import os
    
    # Mock data
    equity_history = [
        {'date': '2024-01-01', 'total_equity': 100000},
        {'date': '2024-06-01', 'total_equity': 110000},
        {'date': '2024-12-01', 'total_equity': 115000},
    ]
    
    trades = [
        {'symbol': 'TEST', 'side': 'buy', 'quantity': 100, 'price': 10.0, 'timestamp': '2024-01-15'},
        {'symbol': 'TEST', 'side': 'sell', 'quantity': 100, 'price': 11.0, 'timestamp': '2024-03-01'},
    ]
    
    # Generate report
    generator = ReportGenerator(output_dir='data/reports')
    report_path = generator.generate(
        session_id='test_session',
        strategy_name='TestStrategy',
        equity_history=equity_history,
        trades=trades,
        positions={},
        params={'param1': 10, 'param2': 20}
    )
    
    assert os.path.exists(report_path), "Report file should exist"
    
    # Check content
    with open(report_path, 'r') as f:
        content = f.read()
    
    assert '回测报告' in content, "Should have report title"
    assert '执行摘要' in content, "Should have executive summary"
    assert '绩效指标' in content, "Should have performance metrics"
    
    print(f"✅ Report generated: {report_path}")
    print(f"✅ Report size: {len(content)} bytes")
    
    # Cleanup
    os.remove(report_path)
    
    print("✅ Report Generator tests passed!")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("   Phase 4 Advanced Features - Test Suite")
    print("="*60)
    
    tests = [
        ("PortfolioManager", test_portfolio_manager),
        ("ParameterOptimizer", test_parameter_optimizer),
        ("Attribution", test_attribution),
        ("ReportGenerator", test_report_generator),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, success))
        except Exception as e:
            logger.error(f"{name} failed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("   Test Summary")
    print("="*60)
    
    passed = sum(1 for _, s in results if s)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {name}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
