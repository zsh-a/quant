import React, { useMemo, useState } from 'react';
import {
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ComposedChart
} from 'recharts';
import { lttb } from '../lttb';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics, BacktestMetrics } from '../utils/metrics';

interface ComparisonProps {
    selectedSessionIds: string[];
    sessionDataCache: Record<string, { equity: EquityPoint[], trades: Trade[], positions: Record<string, Position> }>;
    allSessions: SessionSummary[];
    benchmarksData: Record<string, BenchmarkData[]>;
    availableBenchmarks: { code: string, name: string }[];
}

const COLORS = [
    '#6366f1', // Indigo (Primary)
    '#ec4899', // Pink
    '#10b981', // Emerald
    '#f59e0b', // Amber
    '#8b5cf6', // Violet
    '#3b82f6', // Blue
    '#ef4444', // Red
];

const METRIC_LABELS: Record<keyof BacktestMetrics, string> = {
    totalReturn: 'Total Return',
    annualizedReturn: 'CAGR (Annualized)',
    maxDrawdown: 'Max Drawdown',
    sharpeRatio: 'Sharpe Ratio',
    volatility: 'Volatility (Ann.)',
    winRate: 'Win Days %',
    profitFactor: 'Profit Factor',
    totalTrades: 'Total Trades',
    avgProfit: 'Avg Profit',
    avgLoss: 'Avg Loss'
};

const FORMATTERS: Record<keyof BacktestMetrics, (val: number) => string> = {
    totalReturn: (v) => `${(v * 100).toFixed(2)}%`,
    annualizedReturn: (v) => `${(v * 100).toFixed(2)}%`,
    maxDrawdown: (v) => `${(v * 100).toFixed(2)}%`,
    sharpeRatio: (v) => v.toFixed(2),
    volatility: (v) => `${(v * 100).toFixed(2)}%`,
    winRate: (v) => `${(v * 100).toFixed(2)}%`,
    profitFactor: (v) => v.toFixed(2),
    totalTrades: (v) => v.toString(),
    avgProfit: (v) => `$${v.toFixed(2)}`,
    avgLoss: (v) => `$${v.toFixed(2)}`
};

const Comparison: React.FC<ComparisonProps> = ({
    selectedSessionIds,
    sessionDataCache,
    allSessions,
    benchmarksData,
    availableBenchmarks
}) => {
    const [useLttb, setUseLttb] = useState(true);

    // Prepare data for each selected session
    const sessionMetrics = useMemo(() => {
        return selectedSessionIds.map(id => {
            const session = allSessions.find(s => s.id === id);
            const data = sessionDataCache[id];
            
            if (!session || !data) return null;

            const metrics = calculateMetrics(data.equity, data.trades);
            return {
                id,
                name: session.strategy,
                mode: session.mode,
                metrics,
                equity: data.equity
            };
        }).filter(item => item !== null) as { id: string, name: string, mode: string, metrics: BacktestMetrics, equity: EquityPoint[] }[];
    }, [selectedSessionIds, sessionDataCache, allSessions]);

    // Prepare Chart Data
    const chartData = useMemo(() => {
        if (sessionMetrics.length === 0) return [];

        const dataMap = new Map<string, any>();

        sessionMetrics.forEach(item => {
            if (item.equity.length === 0) return;

            let processedData = item.equity;
            if (useLttb && item.equity.length > 2000) {
                processedData = lttb(item.equity, 2000, 'total_equity');
            }

            const initialEquity = processedData[0].total_equity || 1;

            processedData.forEach(pt => {
                const ts = pt.timestamp;
                const dateStr = ts.split(' ')[0];

                if (!dataMap.has(dateStr)) {
                    dataMap.set(dateStr, { timestamp: ts });
                }
                const entry = dataMap.get(dateStr);
                const ret = ((pt.total_equity - initialEquity) / initialEquity) * 100;
                entry[`session_${item.id}`] = ret;
            });
        });

        // Add Benchmark Data if available (optional, maybe just show sessions for clarity or add benchmark selector later)
        // For now, let's skip benchmarks in this specific view to focus on session comparison, 
        // OR we can add them if they are in global state. The prompt asks for "Result Comparison", implying sessions.
        
        return Array.from(dataMap.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    }, [sessionMetrics, useLttb]);

    if (selectedSessionIds.length === 0) {
        return (
            <div className="dashboard-view" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', color: 'var(--text-dim)' }}>
                <h2>No Sessions Selected</h2>
                <p>Go to Strategy Lab to select sessions for comparison.</p>
            </div>
        );
    }

    return (
        <div className="dashboard-view">
            <header style={{ marginBottom: '2rem' }}>
                <h2>Strategy Comparison</h2>
                <p className="tagline">Comparing {sessionMetrics.length} sessions</p>
            </header>

            {/* Metrics Comparison Table */}
            <div className="glass card" style={{ overflowX: 'auto', marginBottom: '2rem' }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th style={{ width: '200px' }}>Metric</th>
                            {sessionMetrics.map((s, idx) => (
                                <th key={s.id} style={{ minWidth: '150px' }}>
                                    <div style={{ color: COLORS[idx % COLORS.length], fontWeight: 700 }}>{s.name}</div>
                                    <div className="tagline" style={{ fontSize: '0.7rem' }}>{s.mode}</div>
                                    <div className="tagline" style={{ fontSize: '0.6rem', fontFamily: 'monospace' }}>{s.id.slice(0, 8)}</div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {(Object.keys(METRIC_LABELS) as Array<keyof BacktestMetrics>).map(key => (
                            <tr key={key}>
                                <td style={{ color: 'var(--text-dim)' }}>{METRIC_LABELS[key]}</td>
                                {sessionMetrics.map(s => {
                                    const val = s.metrics[key];
                                    let color = 'inherit';
                                    if (key === 'totalReturn' || key === 'annualizedReturn' || key === 'sharpeRatio' || key === 'profitFactor') {
                                        color = val > 0 ? 'var(--success)' : (val < 0 ? 'var(--danger)' : 'inherit');
                                        if (key === 'sharpeRatio' && val < 1) color = 'var(--text-dim)'; // Neutral if low sharpe
                                    }
                                    if (key === 'maxDrawdown') {
                                        color = val > 0.2 ? 'var(--danger)' : 'inherit';
                                    }

                                    return (
                                        <td key={s.id} style={{ fontWeight: 600, color }}>
                                            {FORMATTERS[key](val)}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>

            {/* Comparison Chart */}
            <div className="glass card chart-container" style={{ height: '500px', padding: '2rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '1rem' }}>
                    <h3>Equity Curve Comparison (%)</h3>
                    <button
                        onClick={() => setUseLttb(!useLttb)}
                        className="glass"
                        style={{
                            padding: '0.3rem 0.6rem',
                            fontSize: '0.75rem',
                            background: useLttb ? 'var(--primary)' : 'rgba(255,255,255,0.05)',
                            color: 'white',
                            border: '1px solid rgba(255,255,255,0.1)',
                            cursor: 'pointer',
                            borderRadius: '4px'
                        }}
                    >
                        LTTB: {useLttb ? 'ON' : 'OFF'}
                    </button>
                </div>
                
                <ResponsiveContainer width="100%" height="90%">
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={['auto', 'auto']} stroke="var(--text-dim)" fontSize={12} tickFormatter={(val) => `${val.toFixed(0)}%`} />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: 'var(--text)' }}
                            formatter={(value: any, name: string) => {
                                const sessionId = name.replace('session_', '');
                                const session = sessionMetrics.find(s => s.id === sessionId);
                                return [`${value.toFixed(2)}%`, session ? session.name : name];
                            }}
                            labelFormatter={(label) => label.split(' ')[0]}
                        />
                        <Legend wrapperStyle={{ paddingTop: '10px' }} />
                        {sessionMetrics.map((s, idx) => (
                            <Line
                                key={s.id}
                                type="monotone"
                                dataKey={`session_${s.id}`}
                                name={s.name} // Legend uses this
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2}
                                dot={false}
                            />
                        ))}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
        </div>
    );
};

export default Comparison;
