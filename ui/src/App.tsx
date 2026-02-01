import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import {
  SessionSummary,
  Trade,
  Position,
  EquityPoint,
  BenchmarkData,
  StrategyMeta
} from './types';
import { useWebSocket } from './hooks/useWebSocket';

import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Comparison from './components/Comparison';
import NewSessionForm from './components/NewSessionForm';
import SessionList from './components/SessionList';
import { RiskPanel } from './components/RiskPanel';
import { CheckpointList } from './components/CheckpointList';

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? "http://localhost:8000"
  : `http://${window.location.hostname}:8000`;

const AVAILABLE_BENCHMARKS = [
  { code: 'sh.000300', name: 'HS300' },
  { code: 'sh.000905', name: 'ZZ500' },
  { code: 'sz.399006', name: 'ChiNext' }
];

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');

  // New Session Form State
  const [strategies, setStrategies] = useState<StrategyMeta[]>([]);

  // Session State
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [selectedSessionIds, setSelectedSessionIds] = useState<string[]>(() => {
    const saved = localStorage.getItem('selectedSessionIds');
    return saved ? JSON.parse(saved) : [];
  });

  // Multi-session Data Cache
  const [sessionDataCache, setSessionDataCache] = useState<Record<string, { equity: EquityPoint[], trades: Trade[], positions: Record<string, Position> }>>({});

  // Detailed Data State (for primary selected session)
  const [primarySessionId, setPrimarySessionId] = useState<string | null>(() => {
    return localStorage.getItem('primarySessionId');
  });
  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Record<string, Position>>({});
  const [error, setError] = useState<string | null>(null);

  // Persist state to localStorage
  useEffect(() => {
    localStorage.setItem('selectedSessionIds', JSON.stringify(selectedSessionIds));
  }, [selectedSessionIds]);

  useEffect(() => {
    if (primarySessionId) {
      localStorage.setItem('primarySessionId', primarySessionId);
    } else {
      localStorage.removeItem('primarySessionId');
    }
  }, [primarySessionId]);

  // Benchmark State
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [benchmarksData, setBenchmarksData] = useState<Record<string, BenchmarkData[]>>({});

  const pollInterval = useRef<number | null>(null);
  const lastUpdatedRef = useRef<string | null>(null);

  const fetchStrategies = async () => {
    try {
      const resp = await fetch(`${API_BASE}/strategies`);
      const data = await resp.json();
      setStrategies(data);
    } catch (err) {
      console.error("Failed to fetch strategies", err);
    }
  };

  const fetchSessions = async () => {
    try {
      const resp = await fetch(`${API_BASE}/sessions`);
      const data = await resp.json();
      setSessions(data);
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const startSession = async (payload: any) => {
    setError(null);
    try {
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

  // WebSocket for real-time updates
  const { isConnected, usePolling } = useWebSocket({
    sessionId: primarySessionId || '',
    enabled: !!primarySessionId,
    onMessage: (message) => {
      console.log('[WebSocket] Received:', message.type);

      switch (message.type) {
        case 'session_progress':
          // Update progress in sessions list
          setSessions(prev => prev.map(s =>
            s.id === message.session_id
              ? { ...s, progress: message.data.progress, status: message.data.status }
              : s
          ));
          break;

        case 'equity_update':
          // Add new equity point
          const equity = message.data.equity;
          setEquityHistory(prev => [...prev, equity]);
          lastUpdatedRef.current = equity.timestamp;
          break;

        case 'trade_executed':
          // Add new trade
          setTrades(prev => [...prev, message.data.trade]);
          break;

        case 'session_completed':
          // Update session status
          setSessions(prev => prev.map(s =>
            s.id === message.session_id
              ? { ...s, status: 'completed', progress: 100 }
              : s
          ));
          break;

        case 'error_occurred':
          console.error('[Session Error]:', message.data.error);
          setError(message.data.error);
          break;
      }
    },
    onConnect: () => {
      console.log('[WebSocket] Connected');
    },
    onDisconnect: () => {
      console.log('[WebSocket] Disconnected');
    },
    fallbackToPolling: true,
    pollingInterval: 2000
  });

  // Initial data fetch and periodic refresh for sessions list
  useEffect(() => {
    // Reset incremental state on session switch
    setEquityHistory([]);
    setTrades([]);
    setPositions({});
    lastUpdatedRef.current = null;

    fetchStrategies();
    fetchSessions();

    if (primarySessionId) {
      fetchSessionDetails(primarySessionId);
    }

    // Only poll for sessions list, not individual session data (WebSocket handles that)
    pollInterval.current = window.setInterval(() => {
      fetchSessions();

      // If WebSocket is not connected and not using polling fallback, fetch manually
      if (!isConnected && !usePolling && primarySessionId) {
        fetchSessionDetails(primarySessionId);
      }
    }, 5000); // Reduced frequency since WebSocket handles real-time updates

    return () => {
      if (pollInterval.current) clearInterval(pollInterval.current);
    };
  }, [primarySessionId, isConnected, usePolling]);

  const fetchSessionDataFull = async (id: string) => {
    try {
      const session = sessions.find(s => s.id === id);
      if (session && session.status === 'completed' && sessionDataCache[id]) {
        return; // Already have full data for completed session
      }

      const resp = await fetch(`${API_BASE}/session/${id}/status`);
      const data = await resp.json();

      setSessionDataCache(prev => ({
        ...prev,
        [id]: {
          equity: data.equity_history || [],
          trades: data.trades || [],
          positions: data.positions || {}
        }
      }));
    } catch (err) {
      console.error("Error fetching full session data", err);
    }
  };

  // Watch selectedSessionIds to fetch missing data
  useEffect(() => {
    selectedSessionIds.forEach(id => {
      if (!sessionDataCache[id]) {
        fetchSessionDataFull(id);
      }
    });
  }, [selectedSessionIds]);

  const fetchSessionDetails = async (id: string) => {
    try {
      let url = `${API_BASE}/session/${id}/status`;
      if (lastUpdatedRef.current) {
        url += `?since=${encodeURIComponent(lastUpdatedRef.current)}`;
      }
      const resp = await fetch(url);
      const data = await resp.json();

      // Incremental merge: only append new data
      if (data.equity_history && data.equity_history.length > 0) {
        setEquityHistory(prev => [...prev, ...data.equity_history]);
        // Update last timestamp for next incremental fetch
        lastUpdatedRef.current = data.equity_history[data.equity_history.length - 1].timestamp;
      }

      if (data.trades && data.trades.length > 0) {
        setTrades(prev => [...prev, ...data.trades]);
      }

      // Positions are always full snapshot (not incremental)
      setPositions(data.positions || {});
    } catch (err) {
      console.error("Fetch details error", err);
    }
  };

  // Fetch Benchmark Data
  useEffect(() => {
    if (!primarySessionId) {
      setBenchmarksData({});
      return;
    }

    const session = sessions.find(s => s.id === primarySessionId);
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

  const handleViewSession = (id: string) => {
    setPrimarySessionId(id);
    if (!selectedSessionIds.includes(id)) {
      setSelectedSessionIds(prev => [...prev, id]);
    }
    setActiveTab('dashboard');
  };

  const activeSessions = sessions.filter(s => s.status === 'running');
  const primarySession = sessions.find(s => s.id === primarySessionId);
  const comparisonData = selectedSessionIds
    .filter(id => id !== primarySessionId && sessionDataCache[id])
    .map(id => ({
      id,
      name: sessions.find(s => s.id === id)?.strategy || id,
      data: sessionDataCache[id].equity
    }));

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        activeSessions={activeSessions}
        onSessionSelect={(id) => {
          setPrimarySessionId(id);
          if (!selectedSessionIds.includes(id)) {
            setSelectedSessionIds(prev => [...prev, id]);
          }
          setActiveTab('dashboard');
        }}
      />

      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <h1>{activeTab === 'lab' ? 'Strategy Lab' : activeTab === 'analysis' ? 'Analysis' : 'Dashboard'}</h1>
          <div>
            <span className="tagline">Connected: </span>
            <span style={{ color: 'var(--success)', fontWeight: 700 }}>Localhost</span>
          </div>
        </header>

        {activeTab === 'dashboard' && (
          <Dashboard
            primarySession={primarySession}
            equityHistory={equityHistory}
            trades={trades}
            positions={positions}
            comparisonData={comparisonData}
            benchmarksData={benchmarksData}
            selectedBenchmarks={selectedBenchmarks}
            onToggleBenchmark={toggleBenchmark}
            availableBenchmarks={AVAILABLE_BENCHMARKS}
            onSelectSession={setPrimarySessionId}
            allSessions={sessions}
          />
        )}

        {activeTab === 'analysis' && (
          <Comparison
            selectedSessionIds={selectedSessionIds}
            sessionDataCache={sessionDataCache}
            allSessions={sessions}
            benchmarksData={benchmarksData}
            availableBenchmarks={AVAILABLE_BENCHMARKS}
          />
        )}

        {activeTab === 'lab' && (
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
            <NewSessionForm
              strategies={strategies}
              onStart={startSession}
              error={error}
            />
            <SessionList
              sessions={sessions}
              selectedSessionIds={selectedSessionIds}
              onToggleSelection={toggleSessionSelection}
              onViewSession={handleViewSession}
              onStopSession={stopSession}
            />
          </div>
        )}

        {activeTab === 'risk' && primarySessionId && (
          <div className="risk-tab">
            <RiskPanel sessionId={primarySessionId} />
            <div style={{ marginTop: '20px' }}>
              <CheckpointList
                sessionId={primarySessionId}
                onRestore={() => {
                  fetchSessionData(primarySessionId);
                }}
              />
            </div>
          </div>
        )}
      </main>
    </div>
  );
};

export default App;