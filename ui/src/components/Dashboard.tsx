import React, { useEffect, useState, useMemo } from 'react';
import {
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, Legend, ComposedChart
} from 'recharts';
import { lttb } from '../lttb';
import { MetricCard } from './layout/MetricCard';
import { EmptyState } from './layout/EmptyState';
import { VirtualizedTradeList } from './VirtualizedTradeList';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics } from '../utils/metrics';
import { formatMoney, formatSignedMoney, formatSigned, formatPercent, colorFromSign, colorFromValue } from '../utils/format';
import { formatModeLabel } from '../utils/display';
import { Button } from './ui/button';

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
                <div className="min-h-[360px] flex-1">
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData} margin={{ top: 8, right: 20, bottom: 28, left: 4 }}>
                            <defs>
                                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="var(--chart-stroke)" stopOpacity={0.22} />
                                    <stop offset="95%" stopColor="var(--chart-stroke)" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="var(--chart-grid)" vertical={false} />
                            <XAxis dataKey="timestamp" hide />
                            <YAxis domain={['auto', 'auto']} stroke="var(--color-text-dim)" fontSize={12} tickFormatter={(val) => `${val.toFixed(0)}%`} width={56} />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'var(--color-border)', borderRadius: '10px' }}
                                itemStyle={{ color: 'var(--color-text)' }}
                                formatter={(value: any, name: string) => [
                                    `${value.toFixed(2)}%`,
                                    name === 'equityReturn' ? 'Strategy' : availableBenchmarks.find(b => b.code === name)?.name || name
                                ]}
                                labelFormatter={(label) => label.split(' ')[0]}
                            />
                            <Legend wrapperStyle={{ paddingTop: '12px' }} verticalAlign="bottom" />
                            <Area type="monotone" dataKey="equityReturn" name="Primary" stroke="var(--chart-stroke)" fillOpacity={1} fill="url(#colorEquity)" strokeWidth={2.5} />
                            {visibleComparisonData.map((c, idx) => (
                                <Line
                                    key={c.id}
                                    type="monotone"
                                    dataKey={`session_${c.id}`}
                                    name={`${c.name} (${c.id.slice(0, 4)})`}
                                    stroke={SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length]}
                                    strokeWidth={2}
                                    dot={false}
                                    strokeDasharray="6 5"
                                />
                            ))}
                            {selectedBenchmarks.map(code => (
                                <Line
                                    key={code}
                                    type="monotone"
                                    dataKey={code}
                                    name={availableBenchmarks.find(b => b.code === code)?.name}
                                    stroke={COLORS[code] || 'var(--color-secondary)'}
                                    strokeWidth={2}
                                    dot={false}
                                />
                            ))}
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
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
