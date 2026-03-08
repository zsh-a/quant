import React, { useState, useMemo } from 'react';
import {
    Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, Legend, ComposedChart
} from 'recharts';
import { lttb } from '../lttb';
import StatCard from './StatCard';
import { VirtualizedTradeList } from './VirtualizedTradeList';
import { SessionSummary, EquityPoint, Trade, Position, BenchmarkData } from '../types';
import { calculateMetrics } from '../utils/metrics';
import { formatMoney, formatSignedMoney, formatSigned, formatPercent, colorFromValue } from '../utils/format';

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

    const metrics = useMemo(() => calculateMetrics(equityHistory, trades), [equityHistory, trades]);

    const chartData = useMemo(() => {
        if (!primarySession && comparisonData.length === 0) return [];

        const relevantCurves: { id: string, data: EquityPoint[] }[] = [];

        if (equityHistory.length > 0) {
            relevantCurves.push({ id: 'Primary', data: equityHistory });
        }

        comparisonData.forEach(c => {
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

    }, [equityHistory, comparisonData, benchmarksData, useLttb, primarySession]);

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
                <h2>No Session Selected</h2>
                <p>Open a session from Overview or Lab to inspect its details.</p>
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
                            Session: {primarySession.strategy || 'Unknown'}
                            <span className="tagline" style={{ fontSize: '1rem' }}> ({primarySession.mode})</span>
                        </h2>
                        <select className="glass-input" style={{ width: 'auto' }} value={primarySession.id} onChange={e => onSelectSession(e.target.value)}>
                            {allSessions.map(s => <option key={s.id} value={s.id}>{s.strategy} - {s.mode} ({s.id.slice(0, 6)}...)</option>)}
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
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="tagline">Benchmarks:</span>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                        {availableBenchmarks.map(bm => (
                            <button
                                key={bm.code}
                                onClick={() => onToggleBenchmark(bm.code)}
                                style={{
                                    padding: '0.3rem 0.6rem',
                                    fontSize: '0.75rem',
                                    background: selectedBenchmarks.includes(bm.code) ? COLORS[bm.code as keyof typeof COLORS] || 'var(--secondary)' : 'rgba(255,255,255,0.1)',
                                    color: selectedBenchmarks.includes(bm.code) ? '#08111f' : 'white',
                                    border: selectedBenchmarks.includes(bm.code) ? '1px solid transparent' : '1px solid rgba(255,255,255,0.12)',
                                    opacity: selectedBenchmarks.includes(bm.code) ? 1 : 0.84,
                                    fontWeight: 700,
                                    borderRadius: '999px'
                                }}
                            >
                                {bm.name}
                            </button>
                        ))}
                        <div style={{ width: '1px', height: '16px', background: 'rgba(255,255,255,0.1)', margin: '0 4px' }} />
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
                                borderRadius: '4px',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '4px'
                            }}
                        >
                            <span style={{
                                width: '6px',
                                height: '6px',
                                borderRadius: '50%',
                                background: useLttb ? '#4ade80' : '#94a3b8',
                                display: 'inline-block'
                            }} />
                            LTTB: {useLttb ? 'ON' : 'OFF'}
                        </button>
                    </div>
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
                    label="CAGR" 
                    value={formatPercent(metrics.annualizedReturn, 2)}
                    subtext="Annualized Return"
                />
                <StatCard 
                    label="Sharpe Ratio" 
                    value={metrics.sharpeRatio.toFixed(2)}
                    subtext={`Vol: ${formatPercent(metrics.volatility, 2)}`}
                />
                <StatCard 
                    label="Max Drawdown" 
                    value={formatPercent(metrics.maxDrawdown, 2)}
                    delta={metrics.maxDrawdown > 0.2 ? 'High Risk' : 'Acceptable'}
                />
                 <StatCard 
                    label="Daily P&L" 
                    value={equityHistory.length > 0 ? formatSignedMoney(equityHistory[equityHistory.length - 1].daily_pnl) : "--"}
                    delta={equityHistory.length > 0 ? formatSigned((equityHistory[equityHistory.length - 1].daily_return ?? 0) * 100, { asPercent: true }) : undefined}
                />
            </div>

            {/* Chart */}
            <div className="glass card chart-container" style={{ marginTop: '2rem', height: '400px', padding: '2rem' }}>
                <h3 style={{ marginBottom: '1rem' }}>Equity Curve (%)</h3>
                <ResponsiveContainer width="100%" height="90%">
                    <ComposedChart data={chartData}>
                        <defs>
                            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.38} />
                                <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                            </linearGradient>
                        </defs>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.1)" vertical={false} />
                        <XAxis dataKey="timestamp" hide />
                        <YAxis domain={['auto', 'auto']} stroke="var(--text-dim)" fontSize={12} tickFormatter={(val) => `${val.toFixed(0)}%`} />
                        <Tooltip
                            contentStyle={{ backgroundColor: 'var(--card-bg)', borderColor: 'rgba(255,255,255,0.1)', borderRadius: '8px' }}
                            itemStyle={{ color: 'var(--text)' }}
                            formatter={(value: any, name: string) => [
                                `${value.toFixed(2)}%`,
                                name === 'equityReturn' ? 'Strategy' : availableBenchmarks.find(b => b.code === name)?.name || name
                            ]}
                            labelFormatter={(label) => label.split(' ')[0]}
                        />
                        <Legend wrapperStyle={{ paddingTop: '10px' }} />
                        <Area type="monotone" dataKey="equityReturn" name="Primary" stroke="#22d3ee" fillOpacity={1} fill="url(#colorEquity)" strokeWidth={3} />
                        
                        {comparisonData.map((c, idx) => {
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

             {/* Holdings & Trades Table Sections */}
             <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '2rem' }}>

                {/* Holdings Card */}
                <div className="glass card">
                <h3 style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {selectedDay ? `Holdings: ${selectedDay.timestamp.split(' ')[0]}` : 'Current Holdings'}
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    {selectedDay && <button className="tagline" style={{ marginRight: '0.5rem', padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={clearDaySelection}>Back to Live</button>}
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
                        {positionKeys.length === 0 && <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>No positions</td></tr>}
                    </tbody>
                    </table>
                </div>
                </div>

                {/* Trades Card - Virtualized */}
                <div className="glass card" style={{ display: 'flex', flexDirection: 'column' }}>
                <h3 style={{ marginBottom: '0.5rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {selectedDay ? `Trades: ${selectedDay.timestamp.split(' ')[0]}` : 'All Trades'}
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
                Daily History
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
                    {equityHistory.length === 0 && <tr><td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>No history yet</td></tr>}
                </tbody>
                </table>
            </div>
        </div>
    );
};

export default Dashboard;
