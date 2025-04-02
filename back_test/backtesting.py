import backtrader as bt

class MoneyDrawDownAnalyzer(bt.Analyzer):
    """
    自定义分析器：在回测过程中记录资金峰值，并计算最大回撤的“金额”。
    """
    def create_analysis(self):
        self.max_value = None
        self.max_drawdown = 0.0

    def next(self):
        current_value = self.strategy.broker.getvalue()
        if self.max_value is None or current_value > self.max_value:
            self.max_value = current_value
        current_drawdown = self.max_value - current_value
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

    def get_analysis(self):
        return {'max_drawdown_money': self.max_drawdown}


def run_backtest(ticker, df, start_date, end_date, strategy, 
                 initial_cash=100000, strategy_params=None, print_log=True,
                 timeframe=bt.TimeFrame.Minutes, compression=5,
                 market_params=None):
    """
    回测函数：加载数据，执行策略，输出资金、回撤、收益率、夏普等。
    """
    if strategy_params is None:
        strategy_params = {}
    # 如果没有指定市场参数，则使用默认值（假设为美国市场）
    if market_params is None:
        market_params = {'trading_days': 252, 'minutes_per_day': 390}

    cerebro = bt.Cerebro()

    # 根据数据时间尺度和市场参数计算年化因子
    if timeframe == bt.TimeFrame.Minutes:
        bars_per_day = market_params['minutes_per_day'] / compression
        factor = int(market_params['trading_days'] * bars_per_day)
    elif timeframe == bt.TimeFrame.Days:
        factor = market_params['trading_days']
    elif timeframe == bt.TimeFrame.Weeks:
        # 可以简单假设一年 52 周
        factor = 52
    else:
        factor = 1  # 默认值，根据需求调整

    # 添加分析器，使用动态计算的 factor
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe', 
                        timeframe=timeframe, annualize=True, factor=factor)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(MoneyDrawDownAnalyzer, _name='mydd')


    # 添加策略
    cerebro.addstrategy(strategy, **strategy_params)

    # 添加数据
    data_feed = bt.feeds.PandasData(
        dataname=df,
        timeframe=timeframe,
        compression=compression,
        fromdate=start_date,
        todate=end_date
    )
    data_feed._name = ticker
    cerebro.adddata(data_feed)

    # 设置初始资金
    cerebro.broker.setcash(initial_cash)
    if print_log:
        print(f"初始资金: {cerebro.broker.getvalue():.2f}")

    # 设置手续费和滑点
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_slippage_fixed(0.05, slip_open=True)

    # 运行回测
    results = cerebro.run()
    strat = results[0]

    final_value = cerebro.broker.getvalue()
    if print_log:
        print(f"回测结束资金: {final_value:.2f}")
        print("=== 回测分析报告 ===")

    # 从分析器获取结果
    sharpe = strat.analyzers.sharpe.get_analysis()
    drawdown = strat.analyzers.drawdown.get_analysis()
    returns = strat.analyzers.returns.get_analysis()
    trades = strat.analyzers.trades.get_analysis()
    mydd = strat.analyzers.mydd.get_analysis()

    # 指标提取
    sharpe_ratio = sharpe.get('sharperatio', None)
    max_dd_pct = drawdown.get('max', {}).get('drawdown', None)
    max_dd_money = mydd.get('max_drawdown_money', 0)
    rtot = returns.get('rtot', None)
    rnorm = returns.get('rnorm', None)

    # 打印日志
    if print_log:
        if sharpe_ratio is not None:
            print(f"夏普比率: {sharpe_ratio:.4f}")
        else:
            print("夏普比率: N/A")
        if max_dd_pct is not None:
            print(f"最大回撤比例: {max_dd_pct:.2f}%")
        else:
            print("最大回撤比例: N/A")
        print(f"最大回撤金额(自定义): {max_dd_money:.2f}")
        if rtot is not None:
            print(f"累计收益率: {rtot*100:.2f}%")
        else:
            print("累计收益率: N/A")
        if rnorm is not None:
            print(f"年化收益率: {rnorm*100:.2f}%")
        else:
            print("年化收益率: N/A")

        total_trades = trades.get('total', {}).get('total', 0)
        won_trades = trades.get('won', {}).get('total', 0)
        print("=== 交易详情 ===")
        print(f"总交易笔数: {total_trades}")
        print(f"胜率: {won_trades} / {total_trades}")

    # 返回结果
    result_dict = {
        'final_value': final_value,
        'sharpe_ratio': sharpe_ratio,
        'max_dd_pct': max_dd_pct,
        'max_dd_money': max_dd_money,
        'rtot': rtot,
        'rnorm': rnorm
    }

    return result_dict, cerebro