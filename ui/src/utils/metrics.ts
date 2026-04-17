/**
 * Unified backtest performance metrics calculation.
 * This module mirrors the Python backend implementation at src/analysis/backtest_metrics.py
 * Both implementations should produce consistent results.
 */

export interface Trade {
    timestamp: string;
    symbol: string;
    name: string;
    type: string;
    price: number;
    quantity: number;
    amount: number;
    commission?: number;
}

export interface EquityPoint {
    timestamp: string;
    total_equity: number;
    daily_pnl?: number;
    daily_return?: number;
    cash: number;
}

export interface PerformanceMetrics {
    // Return metrics
    totalReturn: number;
    annualizedReturn: number;

    // Risk metrics
    maxDrawdown: number;
    maxDrawdownDurationDays: number;
    volatility: number;
    downsideDeviation: number;

    // Risk-adjusted metrics
    sharpeRatio: number;
    sortinoRatio: number;
    calmarRatio: number;

    // Trade statistics (based on daily P&L)
    totalTrades: number;
    winRate: number;
    profitFactor: number;
    avgWin: number;
    avgLoss: number;

    // Period info
    tradingDays: number;
    startDate: string | null;
    endDate: string | null;
}

// Constants matching backend
const RISK_FREE_RATE = 0.02; // 2% annual
const TRADING_DAYS_PER_YEAR = 252;

/**
 * Calculate comprehensive backtest performance metrics.
 * This matches the Python implementation for consistency.
 */
export const calculateMetrics = (
    equityHistory: EquityPoint[],
    trades: Trade[]
): PerformanceMetrics => {
    const emptyMetrics: PerformanceMetrics = {
        totalReturn: 0,
        annualizedReturn: 0,
        maxDrawdown: 0,
        maxDrawdownDurationDays: 0,
        volatility: 0,
        downsideDeviation: 0,
        sharpeRatio: 0,
        sortinoRatio: 0,
        calmarRatio: 0,
        totalTrades: 0,
        winRate: 0,
        profitFactor: 0,
        avgWin: 0,
        avgLoss: 0,
        tradingDays: 0,
        startDate: null,
        endDate: null,
    };

    if (!equityHistory || equityHistory.length < 2) {
        return emptyMetrics;
    }

    // Sort by timestamp
    const sortedEquity = [...equityHistory].sort(
        (a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime()
    );

    const equities = sortedEquity.map(e => e.total_equity);
    const dailyPnls = sortedEquity.map(e => e.daily_pnl || 0);
    const timestamps = sortedEquity.map(e => e.timestamp);

    const initialEquity = equities[0] > 0 ? equities[0] : 1;
    const finalEquity = equities[equities.length - 1];

    // ========== Return Metrics ==========
    const totalReturn = (finalEquity - initialEquity) / initialEquity;

    // Trading period
    let tradingDays: number;
    let startDate: string | null = timestamps[0];
    let endDate: string | null = timestamps[timestamps.length - 1];

    try {
        const start = new Date(timestamps[0].split(' ')[0]);
        const end = new Date(timestamps[timestamps.length - 1].split(' ')[0]);
        tradingDays = Math.max(1, Math.round((end.getTime() - start.getTime()) / (1000 * 3600 * 24)));
    } catch {
        tradingDays = sortedEquity.length;
    }

    // Annualized return
    const annualizedReturn = tradingDays > 0
        ? Math.pow(1 + totalReturn, 365 / tradingDays) - 1
        : 0;

    // ========== Daily Returns ==========
    const dailyReturns: number[] = [];
    for (let i = 1; i < equities.length; i++) {
        if (equities[i - 1] > 0) {
            dailyReturns.push((equities[i] - equities[i - 1]) / equities[i - 1]);
        }
    }

    if (dailyReturns.length === 0) {
        return {
            ...emptyMetrics,
            totalReturn,
            tradingDays,
            startDate,
            endDate,
        };
    }

    // ========== Risk Metrics ==========
    // Volatility (annualized)
    const meanReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / dailyReturns.length;
    const stdDev = Math.sqrt(variance);
    const volatility = stdDev * Math.sqrt(TRADING_DAYS_PER_YEAR);

    // Downside deviation (for Sortino)
    const negativeReturns = dailyReturns.filter(r => r < 0);
    let downsideDeviation = 0;
    if (negativeReturns.length > 0) {
        const downsideVariance = negativeReturns.reduce((sum, r) => sum + r * r, 0) / dailyReturns.length;
        downsideDeviation = Math.sqrt(downsideVariance) * Math.sqrt(TRADING_DAYS_PER_YEAR);
    }

    // Max Drawdown
    let peak = equities[0];
    let maxDrawdown = 0;
    let maxDdDuration = 0;
    let currentDdStart = 0;
    let inDrawdown = false;

    for (let i = 0; i < equities.length; i++) {
        const equity = equities[i];
        if (equity > peak) {
            peak = equity;
            if (inDrawdown) {
                const ddDuration = i - currentDdStart;
                maxDdDuration = Math.max(maxDdDuration, ddDuration);
            }
            inDrawdown = false;
        } else {
            if (!inDrawdown) {
                currentDdStart = i;
                inDrawdown = true;
            }
            const drawdown = (peak - equity) / peak;
            maxDrawdown = Math.max(maxDrawdown, drawdown);
        }
    }

    // Check final drawdown duration
    if (inDrawdown) {
        const ddDuration = equities.length - currentDdStart;
        maxDdDuration = Math.max(maxDdDuration, ddDuration);
    }

    // ========== Risk-Adjusted Metrics ==========
    const dailyRf = RISK_FREE_RATE / TRADING_DAYS_PER_YEAR;

    // Sharpe Ratio
    const sharpeRatio = stdDev > 0
        ? (meanReturn - dailyRf) / stdDev * Math.sqrt(TRADING_DAYS_PER_YEAR)
        : 0;

    // Sortino Ratio
    let sortinoRatio = 0;
    if (downsideDeviation > 0) {
        sortinoRatio = (annualizedReturn - RISK_FREE_RATE) / downsideDeviation;
    } else if (annualizedReturn > RISK_FREE_RATE) {
        sortinoRatio = 999.99;
    }

    // Calmar Ratio
    let calmarRatio = 0;
    if (maxDrawdown > 0) {
        calmarRatio = annualizedReturn / maxDrawdown;
    } else if (annualizedReturn > 0) {
        calmarRatio = 999.99;
    }

    // ========== Trade Statistics (based on daily P&L) ==========
    const positivePnls = dailyPnls.filter(p => p > 0);
    const negativePnls = dailyPnls.filter(p => p < 0);

    const winRate = dailyPnls.length > 0 ? positivePnls.length / dailyPnls.length : 0;

    const grossProfit = positivePnls.reduce((sum, p) => sum + p, 0);
    const grossLoss = Math.abs(negativePnls.reduce((sum, p) => sum + p, 0));

    let profitFactor = 0;
    if (grossLoss > 0) {
        profitFactor = grossProfit / grossLoss;
    } else if (grossProfit > 0) {
        profitFactor = 999.99;
    }

    const avgWin = positivePnls.length > 0
        ? positivePnls.reduce((sum, p) => sum + p, 0) / positivePnls.length
        : 0;
    const avgLoss = negativePnls.length > 0
        ? negativePnls.reduce((sum, p) => sum + p, 0) / negativePnls.length
        : 0;

    // Clamp infinite values
    return {
        totalReturn,
        annualizedReturn,
        maxDrawdown,
        maxDrawdownDurationDays: maxDdDuration,
        volatility,
        downsideDeviation,
        sharpeRatio,
        sortinoRatio: Math.min(sortinoRatio, 999.99),
        calmarRatio: Math.min(calmarRatio, 999.99),
        totalTrades: trades.length,
        winRate,
        profitFactor: Math.min(profitFactor, 999.99),
        avgWin,
        avgLoss,
        tradingDays,
        startDate,
        endDate,
    };
};

/**
 * Convert API response metrics (snake_case) to frontend format (camelCase)
 */
export const fromApiMetrics = (apiMetrics: Record<string, unknown>): PerformanceMetrics => ({
    totalReturn: (apiMetrics.total_return as number) || 0,
    annualizedReturn: (apiMetrics.annualized_return as number) || 0,
    maxDrawdown: (apiMetrics.max_drawdown as number) || 0,
    maxDrawdownDurationDays: (apiMetrics.max_drawdown_duration_days as number) || 0,
    volatility: (apiMetrics.volatility as number) || 0,
    downsideDeviation: (apiMetrics.downside_deviation as number) || 0,
    sharpeRatio: (apiMetrics.sharpe_ratio as number) || 0,
    sortinoRatio: (apiMetrics.sortino_ratio as number) || 0,
    calmarRatio: (apiMetrics.calmar_ratio as number) || 0,
    totalTrades: (apiMetrics.total_trades as number) || 0,
    winRate: (apiMetrics.win_rate as number) || 0,
    profitFactor: (apiMetrics.profit_factor as number) || 0,
    avgWin: (apiMetrics.avg_win as number) || 0,
    avgLoss: (apiMetrics.avg_loss as number) || 0,
    tradingDays: (apiMetrics.trading_days as number) || 0,
    startDate: (apiMetrics.start_date as string) || null,
    endDate: (apiMetrics.end_date as string) || null,
});

// Legacy alias for backward compatibility
export type BacktestMetrics = PerformanceMetrics;
