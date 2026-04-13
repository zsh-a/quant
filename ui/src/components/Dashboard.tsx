import React, { useEffect, useState, useMemo } from 'react';
import ReactEChartsCore from 'echarts-for-react/lib/core';
import * as echarts from 'echarts/core';
import { LineChart as ELineChart } from 'echarts/charts';
import { GridComponent, TooltipComponent, LegendComponent, DataZoomComponent } from 'echarts/components';
import { CanvasRenderer } from 'echarts/renderers';
import { lttb } from '../lttb';
import { MetricCard } from './layout/MetricCard';
import { EmptyState } from './layout/EmptyState';
import { VirtualizedTradeList } from './VirtualizedTradeList';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics } from '../utils/metrics';
import { formatMoney, formatSignedMoney, formatSigned, formatPercent, colorFromSign, colorFromValue } from '../utils/format';
import { formatModeLabel } from '../utils/display';
import { Button } from './ui/button';
import { useChartTheme } from '../hooks/useChartTheme';

echarts.use([ELineChart, GridComponent, TooltipComponent, LegendComponent, DataZoomComponent, CanvasRenderer]);

interface DashboardProps {
    primarySession: SessionSummary | undefined;
    equityHistory: EquityPoint[];
    trades: Trade[];
    positions: Record<string, Position>;
    comparisonData: { id: string, name: string, data: EquityPoint[] }[];
    benchmarksData: Record<string, BenchmarkData[]>;
    selectedBenchmarks: string[];
    onToggleBenchmark: (code: string) => void;
    availableBenchmarks: { code: string, name: string }[];
    onSelectSession: (id: string) => void;
    allSessions: SessionSummary[];
}

const COLORS: Record<string, string> = {
    'sh.000300': '#C07D2F',
    'sh.000905': '#3D8EB8',
    'sz.399006': '#368A72',
};

const SESSION_COMPARE_COLORS = ['#C4626A', '#C07D2F', '#8B75C6', '#3D8EB8', '#C97C3A', '#368A72'];

const PAGE_SIZE = 10;

/** ECharts-based return curve with area gradient, comparison sessions, and benchmarks. */
function ReturnChart({
    chartData,
    chart,
    visibleComparisonData,
    selectedBenchmarks,
    availableBenchmarks,
}: {
    chartData: any[];
    chart: ReturnType<typeof useChartTheme>;
    visibleComparisonData: { id: string; name: string; data: EquityPoint[] }[];
    selectedBenchmarks: string[];
    availableBenchmarks: { code: string; name: string }[];
}) {
    const option = useMemo(() => {
        const timestamps = chartData.map((d) => d.timestamp);

        const series: any[] = [
            {
                name: 'Primary',
                type: 'line',
                data: chartData.map((d) => d.equityReturn ?? null),
                smooth: 0.3,
                symbol: 'none',
                lineStyle: { width: 2.5, color: chart.stroke },
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: chart.stroke + '38' },
                        { offset: 1, color: chart.stroke + '00' },
                    ]),
                },
                z: 2,
            },
        ];

        visibleComparisonData.forEach((c, idx) => {
            series.push({
                name: `${c.name} (${c.id.slice(0, 4)})`,
                type: 'line',
                data: chartData.map((d) => d[`session_${c.id}`] ?? null),
                smooth: 0.3,
                symbol: 'none',
                lineStyle: {
                    width: 2,
                    color: SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length],
                    type: 'dashed',
                },
            });
        });

        selectedBenchmarks.forEach((code) => {
            series.push({
                name: availableBenchmarks.find((b) => b.code === code)?.name ?? code,
                type: 'line',
                data: chartData.map((d) => d[code] ?? null),
                smooth: 0.3,
                symbol: 'none',
                lineStyle: { width: 2, color: COLORS[code] || 'var(--color-secondary)' },
            });
        });

        return {
            backgroundColor: 'transparent',
            grid: { left: 56, right: 20, top: 16, bottom: 56 },
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
            dataZoom: [
                {
                    type: 'inside' as const,
                    xAxisIndex: 0,
                },
            ],
            series,
        };
    }, [chartData, chart, visibleComparisonData, selectedBenchmarks, availableBenchmarks]);

    return (
        <ReactEChartsCore
            echarts={echarts}
            option={option}
            style={{ height: 400 }}
            notMerge
            lazyUpdate
        />
    );
}

const Dashboard: React.FC<DashboardProps> = ({
    primarySession,
    equityHistory,
    trades,
    positions,
    comparisonData,
    benchmarksData,
    selectedBenchmarks,
    onToggleBenchmark,
    availableBenchmarks,
    onSelectSession,
    allSessions
}) => {
    const chart = useChartTheme();
    const [useLttb, setUseLttb] = useState(true);
    const [selectedDay, setSelectedDay] = useState<EquityPoint | null>(null);
    const [equityPage, setEquityPage] = useState(1);
    const [holdingsPage, setHoldingsPage] = useState(1);
    const [visibleComparisonIds, setVisibleComparisonIds] = useState<string[]>([]);

    const metrics = useMemo(() => calculateMetrics(equityHistory, trades), [equityHistory, trades]);

    useEffect(() => {
        setVisibleComparisonIds((prev) => prev.filter((id) => comparisonData.some((item) => item.id === id)));
    }, [comparisonData]);

    const chartData = useMemo(() => {
        if (!primarySession && comparisonData.length === 0) return [];

        const relevantCurves: { id: string, data: EquityPoint[] }[] = [];
        if (equityHistory.length > 0) {
            relevantCurves.push({ id: 'Primary', data: equityHistory });
        }
        comparisonData
            .filter((c) => visibleComparisonIds.includes(c.id))
            .forEach(c => relevantCurves.push({ id: c.id, data: c.data }));

        if (relevantCurves.length === 0) return [];

        const dataMap = new Map<string, any>();
        relevantCurves.forEach(curve => {
            if (curve.data.length === 0) return;
            let processedData = curve.data;
            if (useLttb && curve.data.length > 2000) {
                processedData = lttb(curve.data, 2000, 'total_equity');
            }
            const initialEquity = processedData[0].total_equity || 1;
            processedData.forEach(pt => {
                const dateStr = pt.timestamp.split(' ')[0];
                if (!dataMap.has(dateStr)) dataMap.set(dateStr, { timestamp: pt.timestamp });
                const entry = dataMap.get(dateStr);
                const ret = ((pt.total_equity - initialEquity) / initialEquity) * 100;
                if (curve.id === 'Primary') {
                    entry.equityReturn = ret;
                    entry.equityValue = pt.total_equity;
                } else {
                    entry[`session_${curve.id}`] = ret;
                }
            });
        });

        const bmMaps: Record<string, { map: Map<string, number>, initial: number }> = {};
        Object.keys(benchmarksData).forEach(code => {
            const data = benchmarksData[code];
            if (data?.length > 0) {
                const map = new Map<string, number>();
                data.forEach(d => map.set(d.timestamp.split(' ')[0], d.value));
                bmMaps[code] = { map, initial: data[0].value };
            }
        });
        dataMap.forEach((entry, dateStr) => {
            Object.keys(bmMaps).forEach(code => {
                const { map, initial } = bmMaps[code];
                const val = map.get(dateStr);
                if (val !== undefined && initial > 0) {
                    entry[code] = ((val - initial) / initial) * 100;
                }
            });
        });

        return Array.from(dataMap.values()).sort((a, b) => a.timestamp.localeCompare(b.timestamp));
    }, [equityHistory, comparisonData, benchmarksData, useLttb, primarySession, visibleComparisonIds]);

    const visibleComparisonData = useMemo(
        () => comparisonData.filter((item) => visibleComparisonIds.includes(item.id)),
        [comparisonData, visibleComparisonIds],
    );

    const toggleComparisonVisibility = (sessionId: string) => {
        setVisibleComparisonIds((prev) =>
            prev.includes(sessionId) ? prev.filter((id) => id !== sessionId) : [...prev, sessionId]
        );
    };

    const currentPositions = (selectedDay ? selectedDay.positions : positions) || {};
    const positionKeys = Object.keys(currentPositions);
    const visiblePositions = positionKeys.slice((holdingsPage - 1) * PAGE_SIZE, holdingsPage * PAGE_SIZE);
    const totalHoldingsPages = Math.ceil(positionKeys.length / PAGE_SIZE) || 1;

    const filteredTrades = useMemo(() => {
        const list = selectedDay
            ? trades.filter(t => t.timestamp.split(' ')[0] === selectedDay.timestamp.split(' ')[0])
            : trades;
        return [...list].reverse();
    }, [trades, selectedDay]);

    const totalEquityPages = Math.ceil(equityHistory.length / PAGE_SIZE) || 1;

    if (!primarySession) {
        return <EmptyState title="No Session Selected" description="Open a session from the overview or lab to view details." />;
    }

    return (
        <div className="space-y-7">
            {/* Header */}
            <div className="flex flex-wrap items-center justify-between gap-4">
                <div className="space-y-2.5">
                    <div className="flex items-center gap-4">
                        <h2 className="text-xl font-semibold tracking-tight text-foreground">
                            {primarySession.strategy || 'Unknown'}
                            <span className="ml-2.5 text-base font-normal text-muted-foreground">({formatModeLabel(primarySession.mode)})</span>
                        </h2>
                        <select className="glass-input w-auto" value={primarySession.id} onChange={e => onSelectSession(e.target.value)}>
                            {allSessions.map(s => <option key={s.id} value={s.id}>{s.strategy} - {formatModeLabel(s.mode)} ({s.id.slice(0, 6)}...)</option>)}
                        </select>
                    </div>
                    {primarySession.params && Object.keys(primarySession.params).length > 0 && (
                        <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 text-sm text-muted-foreground">
                            <span className="text-xs font-medium uppercase tracking-wider">Params:</span>
                            {Object.entries(primarySession.params).map(([k, v]) => (
                                <span key={k} className="rounded-md bg-white/[0.06] px-2.5 py-1 text-xs">
                                    {k}: <strong className="text-foreground">{String(v)}</strong>
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Stats Grid */}
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                <MetricCard
                    label="Total Equity"
                    value={
                        <span style={{ color: equityHistory.length > 1 ? colorFromSign(formatPercent(metrics.totalReturn, 2)) : undefined }}>
                            {equityHistory.length > 0 ? formatMoney(equityHistory[equityHistory.length - 1].total_equity) : "--"}
                        </span>
                    }
                    hint={equityHistory.length > 1 ? `${formatPercent(metrics.totalReturn, 2)} total` : undefined}
                />
                <MetricCard
                    label="Annualized"
                    value={<span style={{ color: colorFromSign(formatPercent(metrics.annualizedReturn, 2)) }}>{formatPercent(metrics.annualizedReturn, 2)}</span>}
                    hint="Compounded annual return"
                />
                <MetricCard
                    label="Sharpe Ratio"
                    value={<span style={{ color: colorFromValue(metrics.sharpeRatio) }}>{metrics.sharpeRatio.toFixed(2)}</span>}
                    hint={`Vol: ${formatPercent(metrics.volatility, 2)}`}
                />
                <MetricCard
                    label="Max Drawdown"
                    value={<span style={{ color: colorFromValue(-metrics.maxDrawdown) }}>{formatPercent(metrics.maxDrawdown, 2)}</span>}
                    hint={metrics.maxDrawdown > 0.2 ? 'Elevated risk' : 'Within limits'}
                />
                <MetricCard
                    label="Daily P&L"
                    value={
                        <span style={{ color: equityHistory.length > 0 ? colorFromValue(equityHistory[equityHistory.length - 1].daily_pnl) : undefined }}>
                            {equityHistory.length > 0 ? formatSignedMoney(equityHistory[equityHistory.length - 1].daily_pnl) : "--"}
                        </span>
                    }
                    hint={equityHistory.length > 0 ? formatSigned((equityHistory[equityHistory.length - 1].daily_return ?? 0) * 100, { asPercent: true }) : undefined}
                />
            </div>

            {/* Return Chart */}
            <div className="glass card chart-container mt-0 flex min-h-[520px] flex-col gap-5 p-7">
                <div className="flex flex-wrap items-start justify-between gap-4">
                    <div>
                        <h3 className="mb-1.5 text-base font-semibold tracking-tight">Return Curve (%)</h3>
                        <div className="text-sm text-muted-foreground">Primary session shown by default. Overlay benchmarks or comparison sessions as needed.</div>
                    </div>
                    <div className="flex flex-col items-end gap-3">
                        <Button variant={useLttb ? 'default' : 'outline'} size="sm" onClick={() => setUseLttb(!useLttb)}>
                            LTTB: {useLttb ? 'On' : 'Off'}
                        </Button>
                        {availableBenchmarks.length > 0 && (
                            <div className="flex flex-wrap justify-end gap-2">
                                {availableBenchmarks.map((bm) => (
                                    <Button
                                        key={bm.code}
                                        variant={selectedBenchmarks.includes(bm.code) ? 'default' : 'outline'}
                                        size="sm"
                                        onClick={() => onToggleBenchmark(bm.code)}
                                        style={selectedBenchmarks.includes(bm.code) ? {
                                            borderColor: COLORS[bm.code],
                                            background: COLORS[bm.code],
                                            color: '#08111f',
                                        } : undefined}
                                    >
                                        {bm.name}
                                    </Button>
                                ))}
                            </div>
                        )}
                        {comparisonData.length > 0 && (
                            <div className="flex flex-wrap justify-end gap-2">
                                {comparisonData.map((s, idx) => (
                                    <Button
                                        key={s.id}
                                        variant={visibleComparisonIds.includes(s.id) ? 'default' : 'outline'}
                                        size="sm"
                                        onClick={() => toggleComparisonVisibility(s.id)}
                                        style={visibleComparisonIds.includes(s.id) ? {
                                            borderColor: SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length],
                                            background: SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length],
                                            color: '#08111f',
                                        } : undefined}
                                    >
                                        {s.name}
                                    </Button>
                                ))}
                            </div>
                        )}
                    </div>
                </div>
                <ReturnChart
                    chartData={chartData}
                    chart={chart}
                    visibleComparisonData={visibleComparisonData}
                    selectedBenchmarks={selectedBenchmarks}
                    availableBenchmarks={availableBenchmarks}
                />
            </div>

            {/* Holdings & Trades */}
            <div className="grid gap-6 lg:grid-cols-2">
                {/* Holdings */}
                <div className="glass card">
                    <div className="mb-5 flex items-center justify-between">
                        <h3 className="text-base font-semibold tracking-tight">
                            {selectedDay ? `Holdings · ${selectedDay.timestamp.split(' ')[0]}` : 'Current Holdings'}
                        </h3>
                        <div className="flex items-center gap-2">
                            {selectedDay && (
                                <Button variant="ghost" size="sm" onClick={() => { setSelectedDay(null); setHoldingsPage(1); }}>
                                    Live View
                                </Button>
                            )}
                            <Button variant="ghost" size="sm" onClick={() => setHoldingsPage(p => Math.max(1, p - 1))} disabled={holdingsPage === 1}>Prev</Button>
                            <span className="text-xs tabular-nums text-muted-foreground">{holdingsPage}/{totalHoldingsPages}</span>
                            <Button variant="ghost" size="sm" onClick={() => setHoldingsPage(p => Math.min(totalHoldingsPages, p + 1))} disabled={holdingsPage >= totalHoldingsPages}>Next</Button>
                        </div>
                    </div>
                    <div className="overflow-x-auto">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Symbol</th>
                                    <th className="text-right">Qty</th>
                                    <th className="text-right">Avg Cost</th>
                                    <th className="text-right">Price</th>
                                    <th className="text-right">Value</th>
                                    <th className="text-right">P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                {visiblePositions.map((sym) => {
                                    const pos = currentPositions[sym];
                                    const qty = typeof pos === 'number' ? pos : pos.qty;
                                    const price = typeof pos === 'number' ? 0 : (pos.price || 0);
                                    const value = typeof pos === 'number' ? 0 : (pos.value || qty * price);
                                    const avgCost = pos.avg_cost || 0;
                                    const pnl = pos.unrealized_pnl || 0;
                                    const pnlPct = (pos.pnl_pct || 0) * 100;
                                    return (
                                        <tr key={sym}>
                                            <td>
                                                <div className="font-medium">{sym}</div>
                                                <div className="text-xs text-muted-foreground">{pos.name}</div>
                                            </td>
                                            <td className="text-right tabular-nums">{qty}</td>
                                            <td className="text-right tabular-nums">{avgCost > 0 ? formatMoney(avgCost) : '-'}</td>
                                            <td className="text-right tabular-nums">{formatMoney(price)}</td>
                                            <td className="text-right tabular-nums font-semibold">{formatMoney(value)}</td>
                                            <td className="text-right tabular-nums">
                                                <div style={{ color: colorFromValue(pnl) }}>{formatSignedMoney(pnl)}</div>
                                                <div className="text-xs" style={{ color: colorFromValue(pnlPct) }}>{formatSigned(pnlPct, { asPercent: true })}</div>
                                            </td>
                                        </tr>
                                    );
                                })}
                                {positionKeys.length === 0 && (
                                    <tr><td colSpan={6} className="p-5 text-center text-sm text-muted-foreground">No holdings</td></tr>
                                )}
                            </tbody>
                        </table>
                    </div>
                </div>

                {/* Trades */}
                <div className="glass card flex flex-col">
                    <div className="mb-3 flex items-center justify-between">
                        <h3 className="text-base font-semibold tracking-tight">
                            {selectedDay ? `Trades · ${selectedDay.timestamp.split(' ')[0]}` : 'All Trades'}
                        </h3>
                        <span className="text-xs tabular-nums text-muted-foreground">{filteredTrades.length} trades</span>
                    </div>
                    <VirtualizedTradeList trades={filteredTrades} height={350} showDate={!selectedDay} />
                </div>
            </div>

            {/* Daily History */}
            <div className="glass card">
                <div className="mb-5 flex items-center justify-between">
                    <h3 className="text-base font-semibold tracking-tight">Daily History</h3>
                    <div className="flex items-center gap-2">
                        <Button variant="ghost" size="sm" onClick={() => setEquityPage(p => Math.max(1, p - 1))} disabled={equityPage === 1}>Prev</Button>
                        <span className="text-xs tabular-nums text-muted-foreground">{equityPage}/{totalEquityPages}</span>
                        <Button variant="ghost" size="sm" onClick={() => setEquityPage(p => Math.min(totalEquityPages, p + 1))} disabled={equityPage >= totalEquityPages}>Next</Button>
                    </div>
                </div>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>Date</th>
                            <th className="text-right">Equity</th>
                            <th className="text-right">Daily P&L</th>
                            <th className="text-right">Return</th>
                            <th>Holdings</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[...equityHistory].reverse().slice((equityPage - 1) * PAGE_SIZE, equityPage * PAGE_SIZE).map((day, idx) => (
                            <tr
                                key={idx}
                                className={`cursor-pointer ${selectedDay?.timestamp === day.timestamp ? 'bg-primary/8' : ''}`}
                                onClick={() => { setSelectedDay(day); setHoldingsPage(1); }}
                            >
                                <td className="tabular-nums">{day.timestamp.split(' ')[0]}</td>
                                <td className="text-right tabular-nums">{formatMoney(day.total_equity)}</td>
                                <td className="text-right tabular-nums" style={{ color: colorFromValue(day.daily_pnl) }}>
                                    {formatSignedMoney(day.daily_pnl)}
                                </td>
                                <td className="text-right tabular-nums font-semibold" style={{ color: colorFromValue((day.daily_return ?? 0) * 100) }}>
                                    {formatSigned((day.daily_return ?? 0) * 100, { asPercent: true })}
                                </td>
                                <td className="text-xs text-muted-foreground">
                                    {Object.values(day.positions || {}).map(p => `${p.name} (${p.qty})`).slice(0, 3).join(', ')}
                                    {Object.keys(day.positions || {}).length > 3 ? '...' : ''}
                                </td>
                            </tr>
                        ))}
                        {equityHistory.length === 0 && (
                            <tr><td colSpan={5} className="p-5 text-center text-sm text-muted-foreground">No history data</td></tr>
                        )}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default Dashboard;
