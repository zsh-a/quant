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
    '#5B5FD9',
    '#C4626A',
    '#3D8EB8',
    '#368A72',
    '#8B75C6',
    '#C97C3A',
    '#5B93B7',
];

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

        return Array.from(dataMap.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    }, [sessionMetrics, useLttb]);

    if (selectedSessionIds.length === 0) {
        return (
            <EmptyState title="No Sessions Selected for Comparison" description="Go to the Strategy Lab to select sessions for comparison." />
        );
    }

    return (
        <div className="dashboard-view space-y-7">
            <PageHeader
                eyebrow="Cross-run Analytics"
                title="Strategy Comparison"
                description={`Currently comparing ${sessionMetrics.length} session(s)`}
                actions={
                    <Button variant={useLttb ? 'default' : 'outline'} size="sm" onClick={() => setUseLttb(!useLttb)}>
                        LTTB: {useLttb ? 'ON' : 'OFF'}
                    </Button>
                }
            />

            <SectionCard title="Metrics Matrix" description="Side-by-side comparison of return, risk, and trading efficiency metrics across sessions.">
            <div className="overflow-x-auto">
                <table className="data-table">
                    <thead>
                        <tr>
                            <th style={{ width: '200px' }}>Metric</th>
                            {sessionMetrics.map((s, idx) => (
                                <th key={s.id} style={{ minWidth: '150px' }}>
                                    <div style={{ color: COLORS[idx % COLORS.length], fontWeight: 700 }}>{s.name}</div>
                                    <div className="mt-0.5 text-xs font-normal normal-case tracking-normal text-muted-foreground">{s.mode}</div>
                                    <div className="mt-0.5 font-mono text-[10px] font-normal normal-case tracking-normal text-muted-foreground/60">{s.id.slice(0, 8)}</div>
                                </th>
                            ))}
                        </tr>
                    </thead>
                    <tbody>
                        {(Object.keys(METRIC_LABELS) as Array<DisplayableMetricKey>).map(key => (
                            <tr key={key}>
                                <td style={{ color: 'var(--color-text-dim)' }}>{METRIC_LABELS[key]}</td>
                                {sessionMetrics.map(s => {
                                    const val = s.metrics[key];
                                    const numVal = typeof val === 'number' ? val : 0;
                                    let color = 'inherit';
                                    if (key === 'totalReturn' || key === 'annualizedReturn' || key === 'sharpeRatio' || key === 'profitFactor') {
                                        color = numVal > 0 ? 'var(--color-success)' : (numVal < 0 ? 'var(--color-danger)' : 'inherit');
                                        if (key === 'sharpeRatio' && numVal < 1) color = 'var(--color-text-dim)';
                                    }
                                    if (key === 'maxDrawdown') {
                                        color = numVal > 0.2 ? 'var(--color-danger)' : 'inherit';
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

            <SectionCard title="Return Curve Comparison (%)" description="View the relative return trajectory for each session.">
            <div className="chart-container h-[500px]">
                <ResponsiveContainer width="100%" height="90%">
                    <ComposedChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={['auto', 'auto']} stroke="var(--color-text-dim)" fontSize={12} tickFormatter={(val) => `${val.toFixed(0)}%`} />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--color-border)', borderRadius: '10px' }}
                            itemStyle={{ color: 'var(--color-text)' }}
                            formatter={(value: any, name: string) => {
                                const sessionId = name.replace('session_', '');
                                const session = sessionMetrics.find(s => s.id === sessionId);
                                return [`${value.toFixed(2)}%`, session ? session.name : name];
                            }}
                            labelFormatter={(label) => label.split(' ')[0]}
                        />
                        <Legend wrapperStyle={{ paddingTop: '12px' }} />
                        {sessionMetrics.map((s, idx) => (
                            <Line
                                key={s.id}
                                type="monotone"
                                dataKey={`session_${s.id}`}
                                name={s.name}
                                stroke={COLORS[idx % COLORS.length]}
                                strokeWidth={2.2}
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
