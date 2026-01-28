import React, { useState, useEffect, useRef } from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, Legend, ComposedChart
} from 'recharts';
import { lttb } from './lttb';
import './App.css';

const API_BASE = "http://localhost:8000";

interface Trade {
  timestamp: string;
  symbol: string;
  name: string;
  type: string;
  price: number;
  quantity: number;
  amount: number;
  commission?: number;
}

interface Position {
  qty: number;
  name: string;
  price: number;
  value: number;
  avg_cost?: number;
  unrealized_pnl?: number;
  pnl_pct?: number;
}

interface EquityPoint {
  timestamp: string;
  total_equity: number;
  daily_pnl?: number;
  daily_return?: number;
  cash: number;
  positions: Record<string, Position>;
}

interface SessionSummary {
  id: string;
  strategy: string;
  symbol: string;
  status: string;
  mode: string;
  progress: number;
  start_date: string;
  end_date?: string;
}

interface BenchmarkData {
  timestamp: string;
  value: number;
}

const AVAILABLE_BENCHMARKS = [
  { code: 'sh.000300', name: 'HS300' },
  { code: 'sh.000905', name: 'ZZ500' },
  { code: 'sz.399006', name: 'ChiNext' }
];

const COLORS = {
  'sh.000300': '#ec4899', // Pink
  'sh.000905': '#f59e0b', // Amber
  'sz.399006': '#10b981', // Emerald
};

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  
  // New Session Form State
  const [strategy, setStrategy] = useState('rotation');
  const [symbol, setSymbol] = useState('sh.000300');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState<string>(''); // Optional end date
  const [mode, setMode] = useState('backtest');
  
  // Session State
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSessionIds, setSelectedSessionIds] = useState<string[]>([]);
  
  // Detailed Data State (for primary selected session)
  const [primarySessionId, setPrimarySessionId] = useState<string | null>(null);
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Record<string, Position>>({});
  const [selectedDay, setSelectedDay] = useState<EquityPoint | null>(null);
  const [error, setError] = useState<string | null>(null);

  // Benchmark State
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [benchmarksData, setBenchmarksData] = useState<Record<string, BenchmarkData[]>>({});

  // Pagination states
  const [equityPage, setEquityPage] = useState(1);
  const [tradePage, setTradePage] = useState(1);
  const [holdingsPage, setHoldingsPage] = useState(1);
  const PAGE_SIZE = 10;

  const pollInterval = useRef<number | null>(null);
  const lastUpdatedRef = useRef<string | null>(null);

  const fetchSessions = async () => {
    try {
      const resp = await fetch(`${API_BASE}/sessions`);
      const data = await resp.json();
      setSessions(data);
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const startSession = async () => {
    setError(null);
    try {
      const payload: any = { strategy, symbol, start_date: startDate, mode };
      if (endDate) payload.end_date = endDate;

      const resp = await fetch(`${API_BASE}/session/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();
      fetchSessions();
      setPrimarySessionId(data.session_id);
      setSelectedSessionIds(prev => [...prev, data.session_id]);
      setActiveTab('dashboard');
    } catch (err) {
      setError("Failed to start session");
    }
  };

  const stopSession = async (id: string) => {
    try {
      await fetch(`${API_BASE}/session/${id}/stop`, { method: 'POST' });
      fetchSessions();
    } catch (err) {
      console.error("Failed to stop", err);
    }
  };

  // Poll for sessions list and active session details
  useEffect(() => {
    // Reset incremental state on session switch
    setEquityHistory([]);
    setTrades([]);
    setPositions({});
    lastUpdatedRef.current = null;
    
    fetchSessions();
    if (primarySessionId) {
        fetchSessionDetails(primarySessionId);
    }

    pollInterval.current = window.setInterval(() => {
        fetchSessions();
        if (primarySessionId) {
            fetchSessionDetails(primarySessionId);
        }
    }, 1000);
    return () => {
      if (pollInterval.current) clearInterval(pollInterval.current);
    };
  }, [primarySessionId]);

  const fetchSessionDetails = async (id: string) => {
    try {
      let url = `${API_BASE}/session/${id}/status`;
      if (lastUpdatedRef.current) {
          url += `?since=${lastUpdatedRef.current}`;
      }
      const resp = await fetch(url);
      const data = await resp.json();
      
      if (data.equity_history && data.equity_history.length > 0) {
          setEquityHistory(prev => [...prev, ...data.equity_history]);
          lastUpdatedRef.current = data.equity_history[data.equity_history.length - 1].timestamp;
      }
      
      if (data.trades && data.trades.length > 0) {
          setTrades(prev => [...prev, ...data.trades]);
      }
      
      setPositions(data.positions || {});
    } catch (err) {
      console.error("Fetch details error", err);
    }
  };

  // Fetch Benchmark Data when primary session or benchmark selection changes
  useEffect(() => {
    if (!primarySessionId) {
        setBenchmarksData({});
        return;
    }
    
    const session = sessions.find(s => s.id === primarySessionId);
    // FIX: Ensure session exists AND start_date is present before fetching
    if (!session || !session.start_date) return;

    const fetchBenchmarks = async () => {
        const newData: Record<string, BenchmarkData[]> = {};
        
        await Promise.all(selectedBenchmarks.map(async (bmCode) => {
            try {
                let url = `${API_BASE}/market/benchmark?symbol=${bmCode}&start_date=${session.start_date}`;
                if (session.end_date) url += `&end_date=${session.end_date}`;
                
                const resp = await fetch(url);
                if (resp.ok) {
                    const data = await resp.json();
                    newData[bmCode] = data;
                }
            } catch (err) {
                console.error(`Failed to fetch benchmark ${bmCode}`, err);
            }
        }));
        
        setBenchmarksData(newData);
    };

    if (selectedBenchmarks.length > 0) {
        fetchBenchmarks();
    } else {
        setBenchmarksData({});
    }
  }, [primarySessionId, selectedBenchmarks, sessions]);

  const toggleSessionSelection = (id: string) => {
      if (selectedSessionIds.includes(id)) {
          setSelectedSessionIds(selectedSessionIds.filter(s => s !== id));
          if (primarySessionId === id) setPrimarySessionId(null);
      } else {
          setSelectedSessionIds([...selectedSessionIds, id]);
          setPrimarySessionId(id); // Make newly selected primary
      }
  };

  const toggleBenchmark = (code: string) => {
      if (selectedBenchmarks.includes(code)) {
          setSelectedBenchmarks(prev => prev.filter(c => c !== code));
      } else {
          setSelectedBenchmarks(prev => [...prev, code]);
      }
  };

  const handleDaySelect = (day: EquityPoint) => {
      setSelectedDay(day);
      setTradePage(1); 
      setHoldingsPage(1);
  };

  const clearDaySelection = () => {
      setSelectedDay(null);
      setTradePage(1);
      setHoldingsPage(1);
  };

  const activeSessions = sessions.filter(s => s.status === 'running');

  // Chart Data Preparation
  const chartData = React.useMemo(() => {
      if (equityHistory.length === 0) return [];
      
      // LTTB Downsampling
      let processedHistory = equityHistory;
      if (equityHistory.length > 2000) {
          processedHistory = lttb(equityHistory, 2000, 'total_equity');
      }
      
      const initialEquity = processedHistory[0].total_equity;
      
      // Pre-process benchmarks into maps for O(1) lookup
      const bmMaps: Record<string, { map: Map<string, number>, initial: number }> = {};
      
      Object.keys(benchmarksData).forEach(code => {
          const data = benchmarksData[code];
          if (data && data.length > 0) {
              const map = new Map();
              data.forEach(d => map.set(d.timestamp.split(' ')[0], d.value));
              bmMaps[code] = { map, initial: data[0].value };
          }
      });

      return processedHistory.map(pt => {
          const dateStr = pt.timestamp.split(' ')[0];
          
          const point: any = {
              timestamp: pt.timestamp,
              equityReturn: ((pt.total_equity - initialEquity) / initialEquity) * 100,
              equityValue: pt.total_equity
          };

          // Add benchmark returns
          Object.keys(bmMaps).forEach(code => {
              const { map, initial } = bmMaps[code];
              const val = map.get(dateStr);
              if (val !== undefined && initial > 0) {
                  point[code] = ((val - initial) / initial) * 100;
              }
          });

          return point;
      });
  }, [equityHistory, benchmarksData]);

  // Derived Data for Display
  const currentPositions = selectedDay ? selectedDay.positions : positions;
  const positionKeys = Object.keys(currentPositions);
  const visiblePositions = positionKeys.slice((holdingsPage - 1) * PAGE_SIZE, holdingsPage * PAGE_SIZE);

  const visibleTradesList = selectedDay 
    ? trades.filter(t => t.timestamp.split(' ')[0] === selectedDay.timestamp.split(' ')[0])
    : trades;
  
  const sortedTrades = [...visibleTradesList].reverse(); 
  const visibleTrades = sortedTrades.slice((tradePage - 1) * PAGE_SIZE, tradePage * PAGE_SIZE);

  return (
    <div className="app-container">
      <nav className="glass sidebar">
        <div className="logo" style={{ fontSize: '1.5rem', fontWeight: 900, marginBottom: '2rem' }}>
          QUENT<span style={{ color: 'var(--primary)' }}>AI</span>
        </div>
        <div className="nav-items">
          <NavItem icon="📊" label="Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
          <NavItem icon="🧪" label="Lab & Sessions" active={activeTab === 'lab'} onClick={() => setActiveTab('lab')} />
        </div>
        
        <div style={{ marginTop: 'auto' }}>
            <div className="tagline">Active Sessions ({activeSessions.length})</div>
            {activeSessions.slice(0, 5).map(s => (
                <div key={s.id} style={{ fontSize: '0.8rem', padding: '0.5rem', background: 'rgba(255,255,255,0.05)', borderRadius: '4px', marginBottom: '0.5rem' }}>
                    <div style={{ display:'flex', justifyContent:'space-between'}}>
                        <span>{s.strategy}</span>
                        <span className={`status-badge ${s.mode === 'live' ? 'status-live' : 'status-backtest'}`}>{s.mode}</span>
                    </div>
                    <div style={{ height:'4px', background:'rgba(255,255,255,0.1)', marginTop:'4px', borderRadius:'2px'}}>
                        <div style={{width: `${s.progress}%`, height:'100%', background:'var(--primary)'}}></div>
                    </div>
                </div>
            ))}
        </div>
      </nav>

      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <h1>{activeTab === 'lab' ? 'Strategy Lab' : 'Dashboard'}</h1>
          <div>
            <span className="tagline">Connected: </span> 
            <span style={{ color: 'var(--success)', fontWeight: 700 }}>Localhost</span>
          </div>
        </header>

        {activeTab === 'dashboard' && (
          <div className="dashboard-view">
            {primarySessionId ? (
                <>
                <div style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '1rem' }}>
                    <div style={{ display:'flex', alignItems:'center', gap:'1rem'}}>
                        <h2>Session: {sessions.find(s => s.id === primarySessionId)?.strategy} <span className="tagline" style={{fontSize:'1rem'}}>({sessions.find(s => s.id === primarySessionId)?.mode})</span></h2>
                        <select className="glass-input" style={{ width: 'auto' }} value={primarySessionId || ''} onChange={e => setPrimarySessionId(e.target.value)}>
                            {sessions.map(s => <option key={s.id} value={s.id}>{s.strategy} - {s.mode} ({s.id.slice(0,6)}...)</option>)}
                        </select>
                    </div>
                    
                    {/* Benchmark Multi-Selector */}
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                        <span className="tagline">Benchmarks:</span>
                        <div style={{ display: 'flex', gap: '0.5rem' }}>
                            {AVAILABLE_BENCHMARKS.map(bm => (
                                <button
                                    key={bm.code}
                                    onClick={() => toggleBenchmark(bm.code)}
                                    style={{
                                        padding: '0.3rem 0.6rem',
                                        fontSize: '0.75rem',
                                        background: selectedBenchmarks.includes(bm.code) ? COLORS[bm.code as keyof typeof COLORS] || 'var(--secondary)' : 'rgba(255,255,255,0.1)',
                                        color: 'white',
                                        border: '1px solid rgba(255,255,255,0.1)',
                                        opacity: selectedBenchmarks.includes(bm.code) ? 1 : 0.7
                                    }}
                                >
                                    {bm.name}
                                </button>
                            ))}
                        </div>
                    </div>
                </div>

                <div className="grid">
                <StatCard label="Total Equity" value={equityHistory.length > 0 ? `$${equityHistory[equityHistory.length - 1].total_equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : "--"} delta={equityHistory.length > 1 ? `${((equityHistory[equityHistory.length - 1].total_equity / equityHistory[0].total_equity - 1) * 100).toFixed(2)}% total` : undefined} />
                <StatCard label="Current Cash" value={equityHistory.length > 0 ? `$${equityHistory[equityHistory.length - 1].cash.toLocaleString(undefined, { minimumFractionDigits: 2 })}` : "--"} />
                <StatCard label="Daily P&L" value={equityHistory.length > 0 ? `$${(equityHistory[equityHistory.length - 1].daily_pnl || 0).toLocaleString(undefined, { minimumFractionDigits: 2 })}` : "--"} delta={equityHistory.length > 0 ? `${((equityHistory[equityHistory.length - 1].daily_return || 0) * 100).toFixed(2)}%` : undefined} />
                </div>

                <div className="glass card chart-container" style={{ marginTop: '2rem', height: '400px', padding: '2rem' }}>
                <h3 style={{ marginBottom: '1rem' }}>Equity Curve (%)</h3>
                <ResponsiveContainer width="100%" height="90%">
                    <ComposedChart data={chartData}>
                    <defs>
                        <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3} />
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
                            name === 'equityReturn' ? 'Strategy' : AVAILABLE_BENCHMARKS.find(b => b.code === name)?.name || name
                        ]}
                        labelFormatter={(label) => label.split(' ')[0]}
                    />
                    <Legend wrapperStyle={{ paddingTop: '10px' }}/>
                    <Area type="monotone" dataKey="equityReturn" name="Strategy" stroke="var(--primary)" fillOpacity={1} fill="url(#colorEquity)" strokeWidth={3} />
                    
                    {/* Render active benchmarks */}
                    {selectedBenchmarks.map(code => (
                        <Line 
                            key={code} 
                            type="monotone" 
                            dataKey={code} 
                            name={AVAILABLE_BENCHMARKS.find(b => b.code === code)?.name} 
                            stroke={COLORS[code as keyof typeof COLORS] || 'var(--secondary)'} 
                            strokeWidth={2} 
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
                            <td style={{ textAlign: 'right' }}>{avgCost > 0 ? `$${avgCost.toFixed(2)}` : '-'}</td>
                            <td style={{ textAlign: 'right' }}>${price.toFixed(2)}</td>
                            <td style={{ textAlign: 'right', fontWeight: 700 }}>${value.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                            <td style={{ textAlign: 'right' }}>
                                <div style={{ color: pnl >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                                    {pnl >= 0 ? '+' : ''}{pnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
                                </div>
                                <div style={{ fontSize: '0.7rem', color: pnl >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                                    {pnlPct.toFixed(2)}%
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

                {/* Trades Card */}
                <div className="glass card">
                    <h3 style={{ marginBottom: '1rem', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    {selectedDay ? `Trades: ${selectedDay.timestamp.split(' ')[0]}` : 'All Trades'}
                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                        <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => setTradePage(p => Math.max(1, p - 1))} disabled={tradePage === 1}>Prev</button>
                        <span className="tagline" style={{ fontSize: '0.7rem' }}>{tradePage} / {Math.ceil(sortedTrades.length / PAGE_SIZE) || 1}</span>
                        <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => setTradePage(p => Math.min(Math.ceil(sortedTrades.length / PAGE_SIZE), p + 1))} disabled={tradePage >= Math.ceil(sortedTrades.length / PAGE_SIZE)}>Next</button>
                    </div>
                    </h3>
                    <div style={{ overflowX: 'auto' }}>
                    <table className="data-table">
                    <thead>
                        <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th style={{ textAlign: 'right' }}>Price</th>
                        <th style={{ textAlign: 'right' }}>Amt</th>
                        <th style={{ textAlign: 'right' }}>Comm</th>
                        </tr>
                    </thead>
                    <tbody>
                        {visibleTrades.map((t, idx) => (
                        <tr key={idx}>
                            <td className="tagline" style={{ fontSize: '0.7rem' }}>{t.timestamp.split(' ')[0]}</td>
                            <td>
                            <div style={{ fontWeight: 600 }}>{t.symbol}</div>
                            </td>
                            <td><span className={`status-badge ${t.type === 'buy' ? 'status-live' : 'status-danger'}`}>{t.type}</span></td>
                            <td style={{ textAlign: 'right' }}>${t.price.toFixed(2)}</td>
                            <td style={{ textAlign: 'right' }}>${t.amount.toLocaleString(undefined, { maximumFractionDigits: 0 })}</td>
                            <td style={{ textAlign: 'right', color: 'var(--text-dim)' }}>
                                {t.commission ? `$${t.commission.toFixed(1)}` : '-'}
                            </td>
                        </tr>
                        ))}
                        {sortedTrades.length === 0 && <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>No trades {selectedDay ? 'on this day' : ''}</td></tr>}
                    </tbody>
                    </table>
                    </div>
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
                            <td style={{ textAlign: 'right' }}>${day.total_equity.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td>
                            <td style={{ textAlign: 'right', color: (day.daily_pnl || 0) >= 0 ? 'var(--success)' : 'var(--danger)' }}>
                                {(day.daily_pnl || 0) >= 0 ? '+' : ''}{day.daily_pnl?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
                            </td>
                            <td style={{ textAlign: 'right', color: (day.daily_return || 0) >= 0 ? 'var(--success)' : 'var(--danger)', fontWeight: 600 }}>
                                {(day.daily_return || 0) >= 0 ? '+' : ''}{((day.daily_return || 0) * 100).toFixed(2)}%
                            </td>
                            <td style={{ color: 'var(--text-dim)', fontSize: '0.8rem' }}>
                                {Object.values(day.positions).map(p => `${p.name} (${p.qty})`).slice(0, 3).join(', ')}{Object.keys(day.positions).length > 3 ? '...' : ''}
                            </td>
                            </tr>
                        ))}
                        {equityHistory.length === 0 && <tr><td colSpan={5} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>No history yet</td></tr>}
                        </tbody>
                    </table>
                </div>
                </>
            ) : (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', color: 'var(--text-dim)' }}>
                    <h2>No Session Selected</h2>
                    <p>Go to Lab & Sessions to start or select a session.</p>
                </div>
            )}
          </div>
        )}

        {activeTab === 'lab' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
            {/* Control Panel */}
            <div className="glass card">
              <h3 style={{ marginBottom: '1.5rem' }}>Start New Session</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="input-group">
                  <label className="tagline">Mode</label>
                  <select className="glass-input" value={mode} onChange={(e) => setMode(e.target.value)}>
                    <option value="backtest">Backtest (Historical)</option>
                    <option value="simulation">Simulation (Paper Trade)</option>
                    <option value="live">Live Trading</option>
                  </select>
                </div>
                <div className="input-group">
                  <label className="tagline">Strategy</label>
                  <select className="glass-input" value={strategy} onChange={(e) => setStrategy(e.target.value)}>
                    <option value="rotation">Advanced Rotation</option>
                    <option value="jsg">JSG Quantitative</option>
                  </select>
                </div>
                <div className="input-group">
                  <label className="tagline">Symbol</label>
                  <input type="text" value={symbol} onChange={(e) => setSymbol(e.target.value)} className="glass-input" />
                </div>
                <div className="input-group">
                  <label className="tagline">Start Date</label>
                  <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="glass-input" />
                </div>
                <div className="input-group">
                  <label className="tagline">End Date (Optional)</label>
                  <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="glass-input" />
                </div>
                <button onClick={startSession}>
                  Launch {mode.charAt(0).toUpperCase() + mode.slice(1)} Session
                </button>
                {error && <div style={{ color: 'var(--danger)', fontSize: '0.8rem' }}>{error}</div>}
              </div>
            </div>

            {/* Sessions List */}
            <div className="glass card">
                <h3 style={{ marginBottom: '1.5rem' }}>All Sessions</h3>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Strategy</th>
                            <th>Timeframe</th>
                            <th>Mode</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sessions.map(s => (
                            <tr key={s.id} style={{ backgroundColor: selectedSessionIds.includes(s.id) ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }}>
                                <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{s.id.slice(0, 8)}...</td>
                                <td>
                                    {s.strategy}
                                    <div className="tagline" style={{ fontSize: '0.7rem' }}>{s.symbol}</div>
                                </td>
                                <td style={{ fontSize: '0.8rem' }}>
                                    {s.start_date}<br/>
                                    {s.end_date || 'Ongoing'}
                                </td>
                                <td><span className={`status-badge ${s.mode === 'live' ? 'status-live' : s.mode === 'simulation' ? 'status-backtest' : ''}`}>{s.mode}</span></td>
                                <td>
                                    {s.status}
                                    {s.status === 'running' && <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem' }}>({s.progress.toFixed(0)}%)</span>}
                                </td>
                                <td>
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => toggleSessionSelection(s.id)}>
                                            {selectedSessionIds.includes(s.id) ? 'Deselect' : 'Select'}
                                        </button>
                                        {s.status === 'running' && (
                                            <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(239, 68, 68, 0.2)', color: 'var(--danger)', fontSize: '0.6rem' }} onClick={() => stopSession(s.id)}>
                                                Stop
                                            </button>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

const NavItem: React.FC<{ icon: string, label: string, active: boolean, onClick: () => void }> = ({ icon, label, active, onClick }) => (
  <div
    className={`nav-item ${active ? 'active' : ''}`}
    onClick={onClick}
    style={{
      padding: '0.75rem 1rem',
      borderRadius: '12px',
      cursor: 'pointer',
      display: 'flex',
      alignItems: 'center',
      gap: '0.75rem',
      backgroundColor: active ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
      color: active ? 'var(--primary)' : 'var(--text-dim)',
      fontWeight: active ? 700 : 500,
      transition: 'all 0.2s'
    }}
  >
    <span>{icon}</span>
    {label}
  </div>
);

const StatCard: React.FC<{ label: string, value: string, delta?: string }> = ({ label, value, delta }) => (
  <div className="glass card">
    <div className="tagline">{label}</div>
    <div style={{ fontSize: '1.5rem', fontWeight: 800, marginTop: '0.5rem' }}>{value}</div>
    {delta && <div style={{ color: 'var(--success)', fontSize: '0.75rem', marginTop: '0.25rem', fontWeight: 700 }}>{delta}</div>}
  </div>
);

export default App;
