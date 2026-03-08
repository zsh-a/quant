import React, { useEffect, useState, useMemo } from 'react';
import {
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, Legend, ComposedChart
} from 'recharts';
import { lttb } from '../lttb';
import StatCard from './StatCard';
import { VirtualizedTradeList } from './VirtualizedTradeList';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics } from '../utils/metrics';
import { formatMoney, formatSignedMoney, formatSigned, formatPercent, colorFromValue } from '../utils/format';
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

const COLORS = {
    'sh.000300': '#f59e0b', // Amber
    'sh.000905': '#38bdf8', // Sky
    'sz.399006': '#34d399', // Emerald
};

const SESSION_COMPARE_COLORS = ['#fb7185', '#f59e0b', '#a78bfa', '#22d3ee', '#f97316', '#4ade80'];

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
            .forEach(c => {
            relevantCurves.push({ id: c.id, data: c.data });
        });

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
                const ts = pt.timestamp;
                const dateStr = ts.split(' ')[0];

                if (!dataMap.has(dateStr)) {
                    dataMap.set(dateStr, { timestamp: ts });
                }
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

        // Benchmarks
        const bmMaps: Record<string, { map: Map<string, number>, initial: number }> = {};
        Object.keys(benchmarksData).forEach(code => {
            const data = benchmarksData[code];
            if (data && data.length > 0) {
                const map = new Map();
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
            prev.includes(sessionId)
                ? prev.filter((id) => id !== sessionId)
                : [...prev, sessionId]
        );
    };

    // Derived Display Data
    const currentPositions = (selectedDay ? selectedDay.positions : positions) || {};
    const positionKeys = Object.keys(currentPositions);
    const visiblePositions = positionKeys.slice((holdingsPage - 1) * PAGE_SIZE, holdingsPage * PAGE_SIZE);

    // Virtualized trades - no pagination needed
    const filteredTrades = useMemo(() => {
        const list = selectedDay
            ? trades.filter(t => t.timestamp.split(' ')[0] === selectedDay.timestamp.split(' ')[0])
            : trades;
        return [...list].reverse();
    }, [trades, selectedDay]);

    // Alias for backward compatibility in JSX
    const sortedTrades = filteredTrades;

    const handleDaySelect = (day: EquityPoint) => {
        setSelectedDay(day);
        setHoldingsPage(1);
    };

    const clearDaySelection = () => {
        setSelectedDay(null);
        setHoldingsPage(1);
    };

    if (!primarySession) {
        return (
            <div className="dashboard-view" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', color: 'var(--text-dim)' }}>
                <h2>未选择会话</h2>
                <p>可从总览或实验室中打开一个会话查看详情。</p>
            </div>
        );
    }

    return (
        <div className="dashboard-view">
            {/* Header Controls */}
            <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
                        <h2>
                            会话：{primarySession.strategy || '未知'}
                            <span className="tagline" style={{ fontSize: '1rem' }}> ({formatModeLabel(primarySession.mode)})</span>
                        </h2>
                        <select className="glass-input" style={{ width: 'auto' }} value={primarySession.id} onChange={e => onSelectSession(e.target.value)}>
                            {allSessions.map(s => <option key={s.id} value={s.id}>{s.strategy} - {formatModeLabel(s.mode)} ({s.id.slice(0, 6)}...)</option>)}
                        </select>
                    </div>
                    {(primarySession.params && Object.keys(primarySession.params).length > 0) && (
                        <div style={{ fontSize: '0.85rem', color: 'var(--text-dim)', display: 'flex', flexWrap: 'wrap', gap: '0.5rem 1rem', alignItems: 'center' }}>
                            <span className="tagline">策略参数:</span>
                            {Object.entries(primarySession.params).map(([k, v]) => (
                                <span key={k} style={{ background: 'rgba(255,255,255,0.08)', padding: '0.2rem 0.5rem', borderRadius: '4px' }}>
                                    {k}: <strong style={{ color: 'var(--text)' }}>{String(v)}</strong>
                                </span>
                            ))}
                        </div>
                    )}
                </div>
            </div>

            {/* Key Stats Grid */}
            <div className="grid">
                <StatCard 
                    label="Total Equity" 
                    value={equityHistory.length > 0 ? formatMoney(equityHistory[equityHistory.length - 1].total_equity) : "--"} 
                    delta={equityHistory.length > 1 ? `${formatPercent(metrics.totalReturn, 2)} total` : undefined}
                />
                <StatCard 
                    label="年化收益" 
                    value={formatPercent(metrics.annualizedReturn, 2)}
                    subtext="按年度折算"
                />
                <StatCard 
                    label="夏普比率" 
                    value={metrics.sharpeRatio.toFixed(2)}
                    subtext={`Vol: ${formatPercent(metrics.volatility, 2)}`}
                />
                <StatCard 
                    label="最大回撤" 
                    value={formatPercent(metrics.maxDrawdown, 2)}
                    delta={metrics.maxDrawdown > 0.2 ? '风险偏高' : '风险可控'}
                />
                 <StatCard 
                    label="当日盈亏" 
                    value={equityHistory.length > 0 ? formatSignedMoney(equityHistory[equityHistory.length - 1].daily_pnl) : "--"}
                    delta={equityHistory.length > 0 ? formatSigned((equityHistory[equityHistory.length - 1].daily_return ?? 0) * 100, { asPercent: true }) : undefined}
                />
            </div>

            {/* Chart */}
            <div
                className="glass card chart-container"
                style={{
                    marginTop: '2rem',
                    minHeight: '520px',
                    padding: '2rem',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '1rem',
                }}
            >
                <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', gap: '1rem', alignItems: 'flex-start', flexWrap: 'wrap' }}>
                    <div>
                        <h3 style={{ marginBottom: '0.35rem' }}>收益曲线 (%)</h3>
                        <div className="tagline">默认仅显示当前会话；可按需叠加基准线与对比会话。</div>
                    </div>
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', alignItems: 'flex-end' }}>
                        <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                            <Button variant={useLttb ? 'default' : 'outline'} size="sm" onClick={() => setUseLttb(!useLttb)}>
                                LTTB: {useLttb ? '开' : '关'}
                            </Button>
                        </div>
                        {availableBenchmarks.length > 0 ? (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'flex-end' }}>
                                {availableBenchmarks.map((benchmark) => (
                                    <Button
                                        key={benchmark.code}
                                        variant={selectedBenchmarks.includes(benchmark.code) ? 'default' : 'outline'}
                                        size="sm"
                                        onClick={() => onToggleBenchmark(benchmark.code)}
                                        style={{
                                            borderColor: selectedBenchmarks.includes(benchmark.code)
                                                ? COLORS[benchmark.code as keyof typeof COLORS]
                                                : undefined,
                                            background: selectedBenchmarks.includes(benchmark.code)
                                                ? COLORS[benchmark.code as keyof typeof COLORS]
                                                : undefined,
                                            color: selectedBenchmarks.includes(benchmark.code) ? '#08111f' : undefined,
                                        }}
                                    >
                                        {benchmark.name}
                                    </Button>
                                ))}
                            </div>
                        ) : null}
                        {comparisonData.length > 0 ? (
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem', justifyContent: 'flex-end' }}>
                                {comparisonData.map((session, idx) => (
                                    <Button
                                        key={session.id}
                                        variant={visibleComparisonIds.includes(session.id) ? 'default' : 'outline'}
                                        size="sm"
                                        onClick={() => toggleComparisonVisibility(session.id)}
                                        style={{
                                            borderColor: visibleComparisonIds.includes(session.id)
                                                ? SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length]
                                                : undefined,
                                            background: visibleComparisonIds.includes(session.id)
                                                ? SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length]
                                                : undefined,
                                            color: visibleComparisonIds.includes(session.id) ? '#08111f' : undefined,
                                        }}
                                    >
                                        {session.name}
                                    </Button>
                                ))}
                            </div>
                        ) : null}
                    </div>
                </div>
                <div style={{ flex: 1, minHeight: '360px' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <ComposedChart data={chartData} margin={{ top: 8, right: 20, bottom: 28, left: 4 }}>
                            <defs>
                                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.38} />
                                    <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                                </linearGradient>
                            </defs>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                            <XAxis dataKey="timestamp" hide />
                            <YAxis domain={['auto', 'auto']} stroke="var(--text-dim)" fontSize={12} tickFormatter={(val) => `${val.toFixed(0)}%`} width={56} />
                            <Tooltip
                                contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                                itemStyle={{ color: 'var(--text)' }}
                                formatter={(value: any, name: string) => [
                                    `${value.toFixed(2)}%`,
                                    name === 'equityReturn' ? '策略收益' : availableBenchmarks.find(b => b.code === name)?.name || name
                                ]}
                                labelFormatter={(label) => label.split(' ')[0]}
                            />
                            <Legend wrapperStyle={{ paddingTop: '12px' }} verticalAlign="bottom" />
                            <Area type="monotone" dataKey="equityReturn" name="Primary" stroke="#22d3ee" fillOpacity={1} fill="url(#colorEquity)" strokeWidth={3} />
                            
                            {visibleComparisonData.map((c, idx) => {
                                 const color = SESSION_COMPARE_COLORS[idx % SESSION_COMPARE_COLORS.length];
                                 return (
                                     <Line
                                         key={c.id}
                                         type="monotone"
                                         dataKey={`session_${c.id}`}
                                         name={`${c.name} (${c.id.slice(0,4)})`}
                                         stroke={color}
                                         strokeWidth={2.25}
                                         dot={false}
                                         strokeDasharray="6 5"
                                     />
                                 );
                            })}

                            {selectedBenchmarks.map(code => (
                                <Line
                                    key={code}
                                    type="monotone"
                                    dataKey={code}
                                    name={availableBenchmarks.find(b => b.code === code)?.name}
                                    stroke={COLORS[code as keyof typeof COLORS] || 'var(--secondary)'}
                                    strokeWidth={2.25}
                                    dot={false}
                                />
                            ))}
                        </ComposedChart>
                    </ResponsiveContainer>
                </div>
            </div>

             {/* Holdings & Trades Table Sections */}
             <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>

                {/* Holdings Card */}
                <div className="glass card">
                <h3 style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {selectedDay ? `持仓快照 · ${selectedDay.timestamp.split(' ')[0]}` : '当前持仓'}
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    {selectedDay && <button className="tagline" style={{ marginRight: '0.5rem', padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={clearDaySelection}>返回实时视图</button>}
                    <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => setHoldingsPage(p => Math.max(1, p - 1))} disabled={holdingsPage === 1}>Prev</button>
                    <span className="tagline" style={{ fontSize: '0.7rem' }}>{holdingsPage} / {Math.ceil(positionKeys.length / PAGE_SIZE) || 1}</span>
                    <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => setHoldingsPage(p => Math.min(Math.ceil(positionKeys.length / PAGE_SIZE), p + 1))} disabled={holdingsPage >= Math.ceil(positionKeys.length / PAGE_SIZE)}>Next</button>
                    </div>
                </h3>
                <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                    <thead>
                        <tr>
                        <th>Symbol</th>
                        <th style={{ textAlign: 'right' }}>Qty</th>
                        <th style={{ textAlign: 'right' }}>Avg Cost</th>
                        <th style={{ textAlign: 'right' }}>Price</th>
                        <th style={{ textAlign: 'right' }}>Value</th>
                        <th style={{ textAlign: 'right' }}>P&L</th>
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
                                <div>{sym}</div>
                                <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{pos.name}</div>
                            </td>
                            <td style={{ textAlign: 'right' }}>{qty}</td>
                            <td style={{ textAlign: 'right' }}>{avgCost > 0 ? formatMoney(avgCost) : '-'}</td>
                            <td style={{ textAlign: 'right' }}>{formatMoney(price)}</td>
                            <td style={{ textAlign: 'right', fontWeight: 700 }}>{formatMoney(value)}</td>
                            <td style={{ textAlign: 'right' }}>
                                <div style={{ color: colorFromValue(pnl) }}>
                                {formatSignedMoney(pnl)}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: colorFromValue(pnlPct) }}>
                                {formatSigned(pnlPct, { asPercent: true })}
                                </div>
                            </td>
                            </tr>
                        );
                        })}
                        {positionKeys.length === 0 && <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>暂无持仓</td></tr>}
                    </tbody>
                    </table>
                </div>
                </div>

                {/* Trades Card - Virtualized */}
                <div className="glass card" style={{ display: 'flex', flexDirection: 'column' }}>
                <h3 style={{ marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {selectedDay ? `成交记录 · ${selectedDay.timestamp.split(' ')[0]}` : '全部成交'}
                    <span className="tagline" style={{ fontSize: '0.7rem' }}>
                        {sortedTrades.length} trades
                    </span>
                </h3>
                <VirtualizedTradeList 
                    trades={sortedTrades} 
                    height={350}
                    showDate={!selectedDay}
                />
                </div>
            </div>

            {/* Daily Evolution Card */}
            <div className="glass card" style={{ marginTop: '2rem' }}>
                <h3 style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                每日历史
                <div style={{ display: 'flex', gap: '0.5rem' }}>
                    <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => setEquityPage(p => Math.max(1, p - 1))} disabled={equityPage === 1}>Prev</button>
                    <span className="tagline" style={{ fontSize: '0.7rem' }}>{equityPage} / {Math.ceil(equityHistory.length / PAGE_SIZE) || 1}</span>
                    <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => setEquityPage(p => Math.min(Math.ceil(equityHistory.length / PAGE_SIZE), p + 1))} disabled={equityPage >= Math.ceil(equityHistory.length / PAGE_SIZE)}>Next</button>
                </div>
                </h3>
                <table className="data-table">
                <thead>
                    <tr>
                    <th>Date</th>
                    <th style={{ textAlign: 'right' }}>Equity</th>
                    <th style={{ textAlign: 'right' }}>Daily P&L</th>
                    <th style={{ textAlign: 'right' }}>Return</th>
                    <th>Holdings Summary</th>
                    </tr>
                </thead>
                <tbody>
                    {[...equityHistory].reverse().slice((equityPage - 1) * PAGE_SIZE, equityPage * PAGE_SIZE).map((day, idx) => (
                    <tr key={idx} style={{ cursor: 'pointer', backgroundColor: selectedDay?.timestamp === day.timestamp ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }} onClick={() => handleDaySelect(day)}>
                        <td>{day.timestamp.split(' ')[0]}</td>
                        <td style={{ textAlign: 'right' }}>{formatMoney(day.total_equity)}</td>
                        <td style={{ textAlign: 'right', color: colorFromValue(day.daily_pnl) }}>
                        {formatSignedMoney(day.daily_pnl)}
                        </td>
                        <td style={{ textAlign: 'right', color: colorFromValue((day.daily_return ?? 0) * 100), fontWeight: 600 }}>
                        {formatSigned((day.daily_return ?? 0) * 100, { asPercent: true })}
                        </td>
                        <td style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                        {Object.values(day.positions || {}).map(p => `${p.name} (${p.qty})`).slice(0, 3).join(', ')}{Object.keys(day.positions || {}).length > 3 ? '...' : ''}
                        </td>
                    </tr>
                    ))}
                    {equityHistory.length === 0 && <tr><td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>暂无历史数据</td></tr>}
                </tbody>
                </table>
            </div>
        </div>
    );
};

export default Dashboard;
