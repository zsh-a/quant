import React, { useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import {
  Trade,
  Position,
  EquityPoint,
  BenchmarkData,
  StrategyMeta
} from './types';
import { useWebSocket } from './hooks/useWebSocket';
import {
  useSessions,
  useSelectedSessions,
  usePrimarySession,
  useSessionDataCache,
  useSessionActions,
  useSessionStore
} from './store';

import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import Comparison from './components/Comparison';
import NewSessionForm from './components/NewSessionForm';
import SessionList from './components/SessionList';
import { RiskPanel } from './components/RiskPanel';
import { CheckpointList } from './components/CheckpointList';
import { PortfolioManager } from './components/PortfolioManager';
import { OptimizerPanel } from './components/OptimizerPanel';
import { AttributionPanel } from './components/AttributionPanel';
import { StrategyLogViewer } from './components/StrategyLogViewer';
import { IndustryHeatmap } from './components/IndustryHeatmap';
import { AutomationPanel } from './components/AutomationPanel';

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? "http://localhost:8000"
  : `http://${window.location.hostname}:8000`;

const AVAILABLE_BENCHMARKS = [
  { code: 'sh.000300', name: 'HS300' },
  { code: 'sh.000905', name: 'ZZ500' },
  { code: 'sz.399006', name: 'ChiNext' }
];

function mergeEquity(prev: EquityPoint[], next: EquityPoint[]): EquityPoint[] {
  if (next.length === 0) return prev;
  const seen = new Set(prev.map((p) => p.timestamp));
  const added = next.filter((p) => !seen.has(p.timestamp));
  return added.length === 0 ? prev : [...prev, ...added];
}

function mergeTrades(prev: Trade[], next: Trade[]): Trade[] {
  if (next.length === 0) return prev;
  const key = (t: Trade) => `${t.timestamp}-${t.symbol}-${t.type}-${t.quantity}`;
  const seen = new Set(prev.map(key));
  const added = next.filter((t) => !seen.has(key(t)));
  return added.length === 0 ? prev : [...prev, ...added];
}

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [strategies, setStrategies] = useState<StrategyMeta[]>([]);
  const [error, setError] = useState<string | null>(null);

  const sessions = useSessions()
  const selectedSessionIds = useSelectedSessions()
  const primarySessionId = usePrimarySession()
  const sessionDataCache = useSessionDataCache()
  const { selectSession, toggleSession, updateSession, setSessions: setStoreSessions, addSessionData, removeSession } = useSessionActions()

  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Record<string, Position>>({});
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [benchmarksData, setBenchmarksData] = useState<Record<string, BenchmarkData[]>>({});

  const primarySession = sessions.find(s => s.id === primarySessionId);
  const primarySessionSource = primarySession?.source || 'manual';
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
      setStoreSessions(data);
    } catch (err) {
      console.error("Failed to fetch sessions", err);
    }
  };

  const startSession = async (payload: any) => {
    setError(null);
    try {
      const useAsync = payload.async === true;
      delete payload.async;

      const endpoint = useAsync ? '/session/run_async' : '/session/run';
      const resp = await fetch(`${API_BASE}${endpoint}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      fetchSessions();
      selectSession(data.session_id);
      setActiveTab('dashboard');

      if (useAsync && data.task_id) {
        pollTaskProgress(data.session_id, data.task_id);
      }
    } catch (err) {
      setError("Failed to start session");
    }
  };

  const pollTaskProgress = (sessionId: string, taskId: string) => {
    const poll = async () => {
      try {
        const resp = await fetch(`${API_BASE}/tasks/backtest/${taskId}`);
        const data = await resp.json();

        updateSession(sessionId, {
          progress: data.progress || 0,
          status: data.status === 'SUCCESS' ? 'completed' : data.status === 'FAILURE' ? 'failed' : 'running'
        });

        if (data.status !== 'SUCCESS' && data.status !== 'FAILURE') {
          setTimeout(poll, 2000);
        } else {
          fetchSessions();
          if (sessionId === primarySessionId) {
            fetchSessionDataFull(sessionId);
          }
        }
      } catch (err) {
        console.error('Task polling error:', err);
      }
    };
    poll();
  };

  const stopSession = async (id: string) => {
    try {
      await fetch(`${API_BASE}/session/${id}/stop`, { method: 'POST' });
      fetchSessions();
    } catch (err) {
      console.error("Failed to stop", err);
    }
  };

  const fetchSessionDataFull = async (id: string) => {
    try {
      const session = sessions.find(s => s.id === id);
      if (session && session.status === 'completed' && sessionDataCache[id]) {
        return;
      }

      const resp = await fetch(`${API_BASE}/session/${id}/status`);
      if (resp.status === 404) {
        console.warn(`Session ${id} not found, removing from store`);
        removeSession(id);
        return;
      }
      const data = await resp.json();

      addSessionData(id, {
        equity: data.equity_history || [],
        trades: data.trades || [],
        positions: data.positions || {}
      });
    } catch (err) {
      console.error("Error fetching full session data", err);
    }
  };

  const fetchSessionDetails = async (id: string) => {
    try {
      const since = lastUpdatedRef.current;
      const url = since
        ? `${API_BASE}/session/${id}/status?since=${encodeURIComponent(since)}`
        : `${API_BASE}/session/${id}/status`;
      const resp = await fetch(url);
      
      if (resp.status === 404) {
        console.warn(`Session ${id} not found, removing from store`);
        removeSession(id);
        return;
      }
      
      const data = await resp.json();

      const equityList = data.equity_history || [];
      const tradeList = data.trades || [];

      if (!since) {
        setEquityHistory(equityList);
        setTrades(tradeList);
      } else {
        setEquityHistory((prev) => mergeEquity(prev, equityList));
        setTrades((prev) => mergeTrades(prev, tradeList));
      }
      if (equityList.length > 0) {
        lastUpdatedRef.current = equityList[equityList.length - 1].timestamp;
      }
      setPositions(data.positions || {});
    } catch (err) {
      console.error("Fetch details error", err);
    }
  };

  const fetchBenchmarks = useCallback(async () => {
    if (!primarySessionId) {
      setBenchmarksData({});
      return;
    }

    const session = sessions.find(s => s.id === primarySessionId);
    if (!session || !session.start_date) return;

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
  }, [primarySessionId, selectedBenchmarks, sessions]);

  useEffect(() => {
    if (selectedBenchmarks.length > 0) {
      fetchBenchmarks();
    } else {
      setBenchmarksData({});
    }
  }, [selectedBenchmarks, fetchBenchmarks]);

  const { isConnected, usePolling } = useWebSocket({
    sessionId: primarySessionId || '',
    enabled: !!primarySessionId,
    onMessage: (message) => {
      console.log('[WebSocket] Received:', message.type);

      switch (message.type) {
        case 'session_progress':
          updateSession(message.session_id, {
            progress: message.data.progress,
            status: message.data.status
          });
          break;

        case 'equity_update':
          const equity = message.data.equity;
          setEquityHistory((prev) => mergeEquity(prev, [equity]));
          lastUpdatedRef.current = equity.timestamp;
          break;

        case 'trade_executed':
          setTrades((prev) => mergeTrades(prev, [message.data.trade]));
          break;

        case 'session_completed':
          updateSession(message.session_id, { status: 'completed', progress: 100 });
          break;

        case 'error_occurred':
          console.error('[Session Error]:', message.data.error);
          setError(message.data.error);
          break;
      }
    },
    onConnect: () => console.log('[WebSocket] Connected'),
    onDisconnect: () => console.log('[WebSocket] Disconnected'),
    fallbackToPolling: true,
    pollingInterval: 2000
  });

  useEffect(() => {
    // 1. One-time parallel initialization
    const initData = async () => {
      console.log('[App] Initializing data...');
      try {
        await Promise.all([
          fetchStrategies(),
          fetchSessions()
        ]);
      } catch (err) {
        console.error('Initial data fetch failed', err);
      }
    };
    
    initData();

    // 2. Separate interval for background list updates
    const listInterval = window.setInterval(fetchSessions, 15000);
    
    return () => clearInterval(listInterval);
  }, []); // Run only once on mount

  useEffect(() => {
    // 3. Reactive effect for specific session details
    if (primarySessionId) {
      setEquityHistory([]);
      setTrades([]);
      setPositions({});
      lastUpdatedRef.current = null;
      fetchSessionDetails(primarySessionId);
    }

    // Interval for dynamic session updates when NOT using WebSocket/Polling
    let detailInterval: number | null = null;
    if (primarySessionId && ((!isConnected && !usePolling) || primarySessionSource === 'automation')) {
      detailInterval = window.setInterval(() => {
        fetchSessionDetails(primarySessionId);
      }, primarySessionSource === 'automation' ? 3000 : 5000);
    }

    return () => {
      if (detailInterval) clearInterval(detailInterval);
    };
  }, [primarySessionId, isConnected, usePolling, primarySessionSource]);

  useEffect(() => {
    selectedSessionIds.forEach(id => {
      if (!sessionDataCache[id]) {
        fetchSessionDataFull(id);
      }
    });
  }, [selectedSessionIds]);

  const toggleBenchmark = (code: string) => {
    if (selectedBenchmarks.includes(code)) {
      setSelectedBenchmarks(prev => prev.filter(c => c !== code));
    } else {
      setSelectedBenchmarks(prev => [...prev, code]);
    }
  };

  const handleViewSession = (id: string) => {
    selectSession(id);
    setActiveTab('dashboard');
  };

  const activeSessions = sessions.filter(s => s.status === 'running');
  const comparisonData = selectedSessionIds
    .filter(id => id !== primarySessionId && sessionDataCache[id])
    .map(id => ({
      id,
      name: sessions.find(s => s.id === id)?.strategy || id,
      data: sessionDataCache[id]?.equity || []
    }));

  // Ensure store hydration on mount
  useEffect(() => {
    try {
      useSessionStore.persist.rehydrate()
    } catch (e) {
      console.warn('Store rehydration failed:', e)
    }
  }, [])

  return (
    <div className="app-container">
      <Sidebar
        activeTab={activeTab}
        onTabChange={setActiveTab}
        activeSessions={activeSessions}
        onSessionSelect={selectSession}
      />

      <main className="main-content">
        <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '2rem' }}>
          <h1>{activeTab === 'lab' ? 'Strategy Lab' : activeTab === 'analysis' ? 'Analysis' : activeTab === 'automation' ? 'Automation' : 'Dashboard'}</h1>
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
            onSelectSession={selectSession}
            allSessions={sessions}
          />
        )}

        {activeTab === 'heatmap' && (
          <IndustryHeatmap />
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

        {activeTab === 'automation' && (
          <AutomationPanel
            strategies={strategies}
            onSelectSession={handleViewSession}
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
              onToggleSelection={toggleSession}
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
                onRestore={() => fetchSessionDataFull(primarySessionId)}
              />
            </div>
          </div>
        )}

        {activeTab === 'portfolio' && (
          <PortfolioManager />
        )}

        {activeTab === 'optimizer' && (
          <OptimizerPanel />
        )}

        {activeTab === 'attribution' && primarySessionId && (
          <AttributionPanel sessionId={primarySessionId} />
        )}

        {activeTab === 'logs' && (
          <StrategyLogViewer sessionId={primarySessionId} />
        )}
      </main>
    </div>
  );
};

export default App;
