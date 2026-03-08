import React, { useMemo, useState } from 'react';
import {
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend, ComposedChart
} from 'recharts';
import { lttb } from '../lttb';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics, PerformanceMetrics } from '../utils/metrics';
import { formatMoney, formatPercent } from '../utils/format';
import { EmptyState } from './layout/EmptyState';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';

interface ComparisonProps {
    selectedSessionIds: string[];
    sessionDataCache: Record<string, { equity: EquityPoint[], trades: Trade[], positions: Record<string, Position> }>;
    allSessions: SessionSummary[];
    benchmarksData: Record<string, BenchmarkData[]>;
    availableBenchmarks: { code: string, name: string }[];
}

const COLORS = [
    '#22d3ee', // Cyan
    '#fb7185', // Rose
    '#f59e0b', // Amber
    '#34d399', // Emerald
    '#a78bfa', // Violet
    '#38bdf8', // Sky
    '#f97316', // Orange
];

// Subset of metrics to display in comparison table
type DisplayableMetricKey = 'totalReturn' | 'annualizedReturn' | 'maxDrawdown' | 'sharpeRatio' | 
    'sortinoRatio' | 'volatility' | 'winRate' | 'profitFactor' | 'totalTrades' | 'avgWin' | 'avgLoss';

const METRIC_LABELS: Record<DisplayableMetricKey, string> = {
    totalReturn: 'Total Return',
    annualizedReturn: 'CAGR (Annualized)',
    maxDrawdown: 'Max Drawdown',
    sharpeRatio: 'Sharpe Ratio',
    sortinoRatio: 'Sortino Ratio',
    volatility: 'Volatility (Ann.)',
    winRate: 'Win Days %',
    profitFactor: 'Profit Factor',
    totalTrades: 'Total Trades',
    avgWin: 'Avg Win',
    avgLoss: 'Avg Loss'
};

const FORMATTERS: Record<DisplayableMetricKey, (val: number) => string> = {
    totalReturn: (v) => formatPercent(v, 2),
    annualizedReturn: (v) => formatPercent(v, 2),
    maxDrawdown: (v) => formatPercent(v, 2),
    sharpeRatio: (v) => v.toFixed(2),
    sortinoRatio: (v) => v.toFixed(2),
    volatility: (v) => formatPercent(v, 2),
    winRate: (v) => formatPercent(v, 2),
    profitFactor: (v) => v.toFixed(2),
    totalTrades: (v) => v.toString(),
    avgWin: (v) => formatMoney(v),
    avgLoss: (v) => formatMoney(v)
};

const Comparison: React.FC<ComparisonProps> = ({
    selectedSessionIds,
    sessionDataCache,
    allSessions,
    benchmarksData: _benchmarksData,
    availableBenchmarks: _availableBenchmarks
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
        }).filter(item => item !== null) as { id: string, name: string, mode: string, metrics: PerformanceMetrics, equity: EquityPoint[] }[];
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
            <EmptyState title="未选择对比会话" description="前往策略实验室选择需要对比的会话。" />
        );
    }

    return (
        <div className="dashboard-view space-y-6">
            <PageHeader
                eyebrow="Cross-run Analytics"
                title="策略对比"
                description={`当前正在比较 ${sessionMetrics.length} 个会话`}
                actions={
                    <Button variant={useLttb ? 'default' : 'outline'} size="sm" onClick={() => setUseLttb(!useLttb)}>
                        LTTB: {useLttb ? 'ON' : 'OFF'}
                    </Button>
                }
            />

            <SectionCard title="指标矩阵" description="对齐展示各会话的收益、风险与交易效率指标。">
            <div className="overflow-x-auto">
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
                        {(Object.keys(METRIC_LABELS) as Array<DisplayableMetricKey>).map(key => (
                            <tr key={key}>
                                <td style={{ color: 'var(--text-dim)' }}>{METRIC_LABELS[key]}</td>
                                {sessionMetrics.map(s => {
                                    const val = s.metrics[key];
                                    const numVal = typeof val === 'number' ? val : 0;
                                    let color = 'inherit';
                                    if (key === 'totalReturn' || key === 'annualizedReturn' || key === 'sharpeRatio' || key === 'profitFactor') {
                                        color = numVal > 0 ? 'var(--success)' : (numVal < 0 ? 'var(--danger)' : 'inherit');
                                        if (key === 'sharpeRatio' && numVal < 1) color = 'var(--text-dim)'; // Neutral if low sharpe
                                    }
                                    if (key === 'maxDrawdown') {
                                        color = numVal > 0.2 ? 'var(--danger)' : 'inherit';
                                    }

                                    return (
                                        <td key={s.id} style={{ fontWeight: 600, color }}>
                                            {FORMATTERS[key](numVal)}
                                        </td>
                                    );
                                })}
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
            </SectionCard>

            <SectionCard title="收益曲线对比 (%)" description="查看各会话的相对收益轨迹变化。">
            <div className="chart-container h-[500px]">
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
                                strokeWidth={2.4}
                                dot={false}
                            />
                        ))}
                    </ComposedChart>
                </ResponsiveContainer>
            </div>
            </SectionCard>
        </div>
    );
};

export default Comparison;
