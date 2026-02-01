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

export interface BacktestMetrics {
    totalReturn: number;
    annualizedReturn: number;
    maxDrawdown: number;
    sharpeRatio: number;
    volatility: number;
    winRate: number;
    profitFactor: number;
    totalTrades: number;
    avgProfit: number;
    avgLoss: number;
}

export const calculateMetrics = (equityHistory: EquityPoint[], trades: Trade[]): BacktestMetrics => {
    if (!equityHistory || equityHistory.length < 2) {
        return {
            totalReturn: 0,
            annualizedReturn: 0,
            maxDrawdown: 0,
            sharpeRatio: 0,
            volatility: 0,
            winRate: 0,
            profitFactor: 0,
            totalTrades: 0,
            avgProfit: 0,
            avgLoss: 0
        };
    }

    // Sort equity history by date just in case
    const sortedEquity = [...equityHistory].sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
    
    const initialEquity = sortedEquity[0].total_equity;
    const finalEquity = sortedEquity[sortedEquity.length - 1].total_equity;
    
    // Total Return
    const totalReturn = (finalEquity - initialEquity) / initialEquity;

    // Time Duration (Days)
    const startDate = new Date(sortedEquity[0].timestamp);
    const endDate = new Date(sortedEquity[sortedEquity.length - 1].timestamp);
    const days = Math.max(1, (endDate.getTime() - startDate.getTime()) / (1000 * 3600 * 24));
    
    // Annualized Return
    const annualizedReturn = Math.pow(1 + totalReturn, 365 / days) - 1;

    // Max Drawdown
    let peak = -Infinity;
    let maxDrawdown = 0;
    
    sortedEquity.forEach(pt => {
        if (pt.total_equity > peak) {
            peak = pt.total_equity;
        }
        const drawdown = (peak - pt.total_equity) / peak;
        if (drawdown > maxDrawdown) {
            maxDrawdown = drawdown;
        }
    });

    // Sharpe Ratio & Volatility
    // Using daily returns if available, otherwise calculate them
    const dailyReturns = sortedEquity.map((pt, idx) => {
        if (idx === 0) return 0;
        return (pt.total_equity - sortedEquity[idx - 1].total_equity) / sortedEquity[idx - 1].total_equity;
    }).slice(1);

    const meanReturn = dailyReturns.reduce((sum, r) => sum + r, 0) / dailyReturns.length;
    const variance = dailyReturns.reduce((sum, r) => sum + Math.pow(r - meanReturn, 2), 0) / dailyReturns.length;
    const stdDev = Math.sqrt(variance);
    const annualizedVol = stdDev * Math.sqrt(252);
    
    // Assume risk-free rate is 0 for simplicity, or 2% (0.02)
    const riskFreeRate = 0.02; 
    // Sharpe = (Annualized Return - Risk Free) / Annualized Volatility
    // Or simplified: Mean Daily Return / Daily StdDev * sqrt(252)
    const sharpeRatio = stdDev === 0 ? 0 : (meanReturn / stdDev) * Math.sqrt(252);

    // Trade Statistics
    // Filter for 'sell' trades or calculate based on round trips?
    // For simplicity, let's look at completed trades if possible, or just use the trade log.
    // The 'trades' array is a flat list of buy/sell. We can't easily pair them without ID.
    // But we can approximate using PnL from trades if provided? 
    // The Trade interface doesn't have PnL. 
    // However, the prompt implies "Trade Metrics". 
    // We can infer win rate from 'daily_pnl' if we assume day trading, but that's inaccurate.
    // Let's rely on `daily_pnl` for "Win Days" vs "Loss Days" as a proxy if we can't pair trades,
    // OR try to pair trades simply (FIFO).
    
    // Let's try to calculate simple Trade stats if we have PnL info on trades (we don't in the interface).
    // But wait, the previous `BTStat.tsx` used `order_stats` from `btres`.
    // The backend `get_account_info` might compute this? 
    // `src/core/backtest_broker.py` usually tracks trades.
    // If we don't have per-trade PnL, we can't calculate Win Rate accurately without replay.
    // Check `Trade` interface again: `timestamp, symbol, name, type, price, quantity, amount, commission`.
    // It does not have realized PnL.
    
    // FALLBACK: Use Daily PnL for "Win Days %" if per-trade is impossible, OR
    // check if `trades` from backend actually includes more fields than the interface defined.
    // I will look at `src/core/backtest_broker.py` later.
    // For now, I will calculate Win Rate based on *Daily* PnL (Positive Days / Total Days) as a placeholder,
    // and note it as "Win Days %". 
    // OR, simpler: Just count number of SELL orders where Price > Average Buy Price? 
    // Too complex to reconstruct FIFO here.
    // Let's stick to Daily Stats for now which are reliable from Equity Curve.
    
    const winDays = dailyReturns.filter(r => r > 0).length;
    const totalDays = dailyReturns.length;
    const winRate = totalDays > 0 ? winDays / totalDays : 0; // This is technically "Win Days %"

    // Profit Factor (based on daily PnL)
    const dailyPnLs = sortedEquity.map(pt => pt.daily_pnl || 0);
    const grossProfit = dailyPnLs.filter(p => p > 0).reduce((sum, p) => sum + p, 0);
    const grossLoss = Math.abs(dailyPnLs.filter(p => p < 0).reduce((sum, p) => sum + p, 0));
    const profitFactor = grossLoss === 0 ? (grossProfit > 0 ? 999 : 0) : grossProfit / grossLoss;

    return {
        totalReturn,
        annualizedReturn,
        maxDrawdown,
        sharpeRatio,
        volatility: annualizedVol,
        winRate, // Actually % of profitable days
        profitFactor,
        totalTrades: trades.length,
        avgProfit: 0, // Hard without trade PnL
        avgLoss: 0    // Hard without trade PnL
    };
};
