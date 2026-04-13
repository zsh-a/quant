import React, { useMemo, useState } from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { LineChart as ELineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { lttb } from '../lttb';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics, PerformanceMetrics } from '../utils/metrics';
import { formatMoney, formatPercent } from '../utils/format';
import { EmptyState } from './layout/EmptyState';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { useChartTheme } from '../hooks/useChartTheme';

echarts.use([ELineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, CanvasRenderer]);

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

function ComparisonChart({
    chartData,
    chart,
    sessionMetrics,
}: {
    chartData: any[];
    chart: ReturnType<typeof useChartTheme>;
    sessionMetrics: { id: string; name: string; mode: string; metrics: PerformanceMetrics; equity: EquityPoint[] }[];
}) {
    const option = useMemo(() => {
        const timestamps = chartData.map((d) => d.timestamp);
        const series = sessionMetrics.map((s, idx) => ({
            name: s.name,
            type: 'line' as const,
            data: chartData.map((d) => d[`session_${s.id}`] ?? null),
            smooth: 0.3,
            symbol: 'none' as const,
            lineStyle: { width: 2.2, color: COLORS[idx % COLORS.length] },
        }));

        return {
            backgroundColor: 'transparent',
            grid: { left: 52, right: 16, top: 16, bottom: 56 },
            tooltip: {
                trigger: 'axis' as const,
                backgroundColor: chart.tooltipBg,
                borderColor: chart.tooltipBorder,
                textStyle: { color: 'var(--color-foreground)', fontSize: 12 },
                formatter: (params: any) => {
                    const label = params[0]?.axisValue?.split(' ')[0] ?? '';
                    const lines = params.map(
                        (p: any) =>
                            `<span style="color:${p.color}">●</span> ${p.seriesName}: ${p.value != null ? p.value.toFixed(2) : '-'}%`,
                    );
                    return `${label}<br/>${lines.join('<br/>')}`;
                },
                axisPointer: { type: 'cross' as const, lineStyle: { type: 'dashed' as const } },
            },
            legend: {
                bottom: 0,
                textStyle: { color: chart.textDim, fontSize: 12 },
                itemWidth: 16,
                itemHeight: 3,
            },
            xAxis: {
                type: 'category' as const,
                data: timestamps,
                axisLabel: { show: false },
                axisLine: { show: false },
                axisTick: { show: false },
            },
            yAxis: {
                type: 'value' as const,
                splitLine: { lineStyle: { color: chart.grid, type: 'dashed' as const } },
                axisLabel: {
                    color: chart.textDim,
                    fontSize: 12,
                    formatter: (v: number) => `${v.toFixed(0)}%`,
                },
            },
            dataZoom: [{ type: 'inside' as const, xAxisIndex: 0 }],
            series,
        };
    }, [chartData, chart, sessionMetrics]);

    return (
        <ReactEChartsCore
            echarts={echarts}
            option={option}
            style={{ height: 450 }}
            notMerge
            lazyUpdate
        />
    );
}

const Comparison: React.FC<ComparisonProps> = ({
    selectedSessionIds,
    sessionDataCache,
    allSessions,
    benchmarksData: _benchmarksData,
    availableBenchmarks: _availableBenchmarks
}) => {
    const chart = useChartTheme();
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
            <ComparisonChart chartData={chartData} chart={chart} sessionMetrics={sessionMetrics} />
            </SectionCard>
        </div>
    );
};

export default Comparison;
