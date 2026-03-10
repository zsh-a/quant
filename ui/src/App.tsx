import React, { Suspense, lazy, useState, useEffect, useRef, useCallback } from 'react';
import './App.css';
import {
  Trade,
  Position,
  EquityPoint,
  BenchmarkData,
  StrategyMeta,
} from './types';
import { useWebSocket } from './hooks/useWebSocket';
import {
  useSessions,
  useSelectedSessions,
  usePrimarySession,
  useSessionDataCache,
  useSessionActions,
  useSessionStore,
} from './store';

import Sidebar from './components/Sidebar';
import { API_BASE } from './utils/api';
import { AppShell } from './components/layout/AppShell';
import { PageHeader } from './components/layout/PageHeader';
import { StatusBadge } from './components/layout/StatusBadge';

const Comparison = lazy(() => import('./components/Comparison'));
const PortfolioManager = lazy(() => import('./components/PortfolioManager'));
const OptimizerPanel = lazy(() => import('./components/OptimizerPanel'));
const IndustryHeatmap = lazy(() => import('./components/IndustryHeatmap'));
const GlobalOverview = lazy(() => import('./components/GlobalOverview'));
const LabPanel = lazy(() => import('./components/LabPanel'));
const MarketAdminPanel = lazy(() => import('./components/MarketAdminPanel'));
const SessionDetail = lazy(() => import('./components/SessionDetail'));

const AVAILABLE_BENCHMARKS = [
  { code: 'sh.000300', name: 'HS300' },
  { code: 'sh.000905', name: 'ZZ500' },
  { code: 'sz.399006', name: 'ChiNext' },
];

function mergeEquity(prev: EquityPoint[], next: EquityPoint[]): EquityPoint[] {
  if (next.length === 0) return prev;
  const seen = new Set(prev.map((point) => point.timestamp));
  const added = next.filter((point) => !seen.has(point.timestamp));
  return added.length === 0 ? prev : [...prev, ...added];
}

function mergeTrades(prev: Trade[], next: Trade[]): Trade[] {
  if (next.length === 0) return prev;
  const key = (trade: Trade) => `${trade.timestamp}-${trade.symbol}-${trade.type}-${trade.quantity}`;
  const seen = new Set(prev.map(key));
  const added = next.filter((trade) => !seen.has(key(trade)));
  return added.length === 0 ? prev : [...prev, ...added];
}

const TITLES: Record<string, string> = {
  overview: '总览',
  lab: '策略实验室',
  session: '会话详情',
  comparison: '策略对比',
  heatmap: '行业热力图',
  portfolio: '组合管理',
  marketAdmin: '行情数据库',
  optimizer: '参数优化',
};

const TabFallback: React.FC = () => (
  <div className="glass flex min-h-[320px] items-center justify-center rounded-[28px] border border-border/70 px-6 py-10 text-sm text-muted-foreground">
    正在加载模块...
  </div>
);

const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'overview' | 'lab' | 'session' | 'comparison' | 'heatmap' | 'portfolio' | 'marketAdmin' | 'optimizer'>('overview');
  const [strategies, setStrategies] = useState<StrategyMeta[]>([]);
  const [error, setError] = useState<string | null>(null);

  const sessions = useSessions();
  const selectedSessionIds = useSelectedSessions();
  const primarySessionId = usePrimarySession();
  const sessionDataCache = useSessionDataCache();
  const { selectSession, toggleSession, updateSession, setSessions: setStoreSessions, addSessionData, removeSession } = useSessionActions();

  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Record<string, Position>>({});
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [benchmarksData, setBenchmarksData] = useState<Record<string, BenchmarkData[]>>({});

  const primarySession = sessions.find((session) => session.id === primarySessionId);
  const primarySessionRunId = primarySession?.run_id || null;
  const lastUpdatedRef = useRef<string | null>(null);
  const sessionDetailInFlightRef = useRef(false);

  const fetchStrategies = async () => {
    try {
      const resp = await fetch(`${API_BASE}/strategies`);
      const data = await resp.json();
      setStrategies(data);
    } catch (err) {
      console.error('Failed to fetch strategies', err);
    }
  };

  const fetchSessions = async () => {
    try {
      const resp = await fetch(`${API_BASE}/sessions`);
      const data = await resp.json();
      setStoreSessions(data);
    } catch (err) {
      console.error('Failed to fetch sessions', err);
    }
  };

  const fetchSessionDataFull = async (id: string) => {
    try {
      const session = sessions.find((item) => item.id === id);
      if (session && session.status === 'completed' && sessionDataCache[id]) {
        return;
      }

      const fetchPaged = async (path: string, limit = 2000) => {
        let offset = 0;
        let hasMore = true;
        const items: any[] = [];
        while (hasMore) {
          const resp = await fetch(`${API_BASE}${path}?limit=${limit}&offset=${offset}`);
          if (resp.status === 404) {
            removeSession(id);
            return [];
          }
          if (!resp.ok) {
            throw new Error(`Failed to fetch ${path}`);
          }
          const data = await resp.json();
          const pageItems = Array.isArray(data.items) ? data.items : [];
          items.push(...pageItems);
          hasMore = Boolean(data.has_more);
          if (pageItems.length === 0) {
            hasMore = false;
          } else {
            offset += pageItems.length;
          }
        }
        return items;
      };

      const [equity, trades] = await Promise.all([
        fetchPaged(`/sessions/${id}/equity`),
        fetchPaged(`/sessions/${id}/trades`),
      ]);

      const positions =
        equity.length > 0 ? equity[equity.length - 1]?.positions || {} : {};

      addSessionData(id, {
        equity,
        trades,
        positions,
      });
    } catch (err) {
      console.error('Error fetching full session data', err);
    }
  };

  const fetchSessionDetails = async (id: string) => {
    if (sessionDetailInFlightRef.current) {
      return;
    }
    try {
      sessionDetailInFlightRef.current = true;
      const since = lastUpdatedRef.current;
      const url = since
        ? `${API_BASE}/session/${id}/status?since=${encodeURIComponent(since)}`
        : `${API_BASE}/session/${id}/status`;
      const resp = await fetch(url);

      if (resp.status === 404) {
        removeSession(id);
        return;
      }

      const data = await resp.json();
      const equityList = data.equity_history || [];
      const tradeList = data.trades || [];

      updateSession(id, {
        status: data.status,
        progress: data.progress,
        end_date: data.end_date,
      });

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
      console.error('Fetch details error', err);
    } finally {
      sessionDetailInFlightRef.current = false;
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
        body: JSON.stringify(payload),
      });
      const data = await resp.json();

      await fetchSessions();
      selectSession(data.session_id);
      setActiveTab('session');

      if (useAsync && data.task_id) {
        pollTaskProgress(data.session_id, data.task_id);
      }
    } catch (err) {
      setError('Failed to start session');
    }
  };

  const pollTaskProgress = (sessionId: string, taskId: string) => {
    const poll = async () => {
      try {
        const resp = await fetch(`${API_BASE}/tasks/backtest/${taskId}`);
        const data = await resp.json();

        updateSession(sessionId, {
          progress: data.progress || 0,
          status: data.status === 'SUCCESS' ? 'completed' : data.status === 'FAILURE' ? 'failed' : 'running',
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
      console.error('Failed to stop', err);
    }
  };

  const deleteSession = async (id: string) => {
    const session = sessions.find((item) => item.id === id);
    if (!session) {
      return;
    }

    const confirmed = window.confirm(`删除会话 ${session.strategy} (${id.slice(0, 8)})？此操作会移除历史记录、交易、日志和检查点。`);
    if (!confirmed) {
      return;
    }

    try {
      const resp = await fetch(`${API_BASE}/session/${id}`, { method: 'DELETE' });
      if (!resp.ok) {
        const data = await resp.json().catch(() => null);
        throw new Error(data?.detail || '删除会话失败');
      }

      removeSession(id);
      if (primarySessionId === id) {
        setEquityHistory([]);
        setTrades([]);
        setPositions({});
        setBenchmarksData({});
        setActiveTab('lab');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : '删除会话失败';
      setError(message);
      console.error('Failed to delete session', err);
    }
  };

  const fetchBenchmarks = useCallback(async () => {
    if (!primarySessionId) {
      setBenchmarksData({});
      return;
    }

    const session = sessions.find((item) => item.id === primarySessionId);
    if (!session || !session.start_date) return;

    const newData: Record<string, BenchmarkData[]> = {};

    await Promise.all(selectedBenchmarks.map(async (benchmarkCode) => {
      try {
        let url = `${API_BASE}/market/benchmark?symbol=${benchmarkCode}&start_date=${session.start_date}`;
        if (session.end_date) url += `&end_date=${session.end_date}`;
        const resp = await fetch(url);
        if (resp.ok) {
          newData[benchmarkCode] = await resp.json();
        }
      } catch (err) {
        console.error(`Failed to fetch benchmark ${benchmarkCode}`, err);
      }
    }));

    setBenchmarksData(newData);
  }, [primarySessionId, selectedBenchmarks, sessions]);

  const handleOpenSession = (id: string) => {
    selectSession(id);
    setActiveTab('session');
  };

  const toggleBenchmark = (code: string) => {
    if (selectedBenchmarks.includes(code)) {
      setSelectedBenchmarks((prev) => prev.filter((item) => item !== code));
    } else {
      setSelectedBenchmarks((prev) => [...prev, code]);
    }
  };

  const { isConnected, usePolling } = useWebSocket({
    sessionId: primarySessionId || '',
    enabled: !!primarySessionId,
    onMessage: (message) => {
      switch (message.type) {
        case 'session_progress':
          updateSession(message.session_id, {
            progress: message.data.progress,
            status: message.data.status,
          });
          break;
        case 'equity_update': {
          const equity = message.data.equity;
          setEquityHistory((prev) => mergeEquity(prev, [equity]));
          lastUpdatedRef.current = equity.timestamp;
          break;
        }
        case 'equity_batch': {
          const updates = Array.isArray(message.data?.updates) ? message.data.updates : [];
          if (updates.length > 0) {
            setEquityHistory((prev) => mergeEquity(prev, updates));
            lastUpdatedRef.current = updates[updates.length - 1].timestamp;
          }
          break;
        }
        case 'trade_executed':
          setTrades((prev) => mergeTrades(prev, [message.data.trade]));
          break;
        case 'trades_batch': {
          const batchTrades = Array.isArray(message.data?.trades) ? message.data.trades : [];
          if (batchTrades.length > 0) {
            setTrades((prev) => mergeTrades(prev, batchTrades));
          }
          break;
        }
        case 'session_completed':
          updateSession(message.session_id, { status: 'completed', progress: 100 });
          if (message.session_id === primarySessionId) {
            fetchSessionDetails(message.session_id);
          }
          break;
        case 'session_failed':
          updateSession(message.session_id, { status: 'failed' });
          setError(message.data.error || 'Session failed');
          break;
        case 'session_stopped':
          updateSession(message.session_id, { status: 'stopped' });
          break;
        case 'error_occurred':
          setError(message.data.error);
          break;
      }
    },
    onConnect: () => console.log('[WebSocket] Connected'),
    onDisconnect: () => console.log('[WebSocket] Disconnected'),
    fallbackToPolling: true,
    pollingInterval: 2000,
  });

  useEffect(() => {
    const initData = async () => {
      try {
        await Promise.all([fetchStrategies(), fetchSessions()]);
      } catch (err) {
        console.error('Initial data fetch failed', err);
      }
    };

    initData();
    const listInterval = window.setInterval(fetchSessions, 15000);
    return () => clearInterval(listInterval);
  }, []);

  useEffect(() => {
    if (selectedBenchmarks.length > 0) {
      fetchBenchmarks();
    } else {
      setBenchmarksData({});
    }
  }, [selectedBenchmarks, fetchBenchmarks]);

  useEffect(() => {
    if (activeTab === 'session' && primarySessionId) {
      setEquityHistory([]);
      setTrades([]);
      setPositions({});
      lastUpdatedRef.current = null;
      fetchSessionDetails(primarySessionId);
    }

    let detailInterval: number | null = null;
    if (
      activeTab === 'session' &&
      primarySessionId
    ) {
      detailInterval = window.setInterval(() => {
        fetchSessionDetails(primarySessionId);
      }, primarySession?.status === 'running' ? 2500 : 8000);
    }

    return () => {
      if (detailInterval) clearInterval(detailInterval);
    };
  }, [activeTab, primarySessionId, primarySessionRunId, primarySession?.status]);

  useEffect(() => {
    selectedSessionIds.forEach((id) => {
      if (!sessionDataCache[id]) {
        fetchSessionDataFull(id);
      }
    });
  }, [selectedSessionIds]);

  useEffect(() => {
    try {
      useSessionStore.persist.rehydrate();
    } catch (e) {
      console.warn('Store rehydration failed:', e);
    }
  }, []);

  const activeSessions = sessions.filter((session) => session.status === 'running');
  const comparisonData = selectedSessionIds
    .filter((id) => id !== primarySessionId && sessionDataCache[id])
    .map((id) => ({
      id,
      name: sessions.find((session) => session.id === id)?.strategy || id,
      data: sessionDataCache[id]?.equity || [],
    }));

  return (
    <AppShell
      sidebar={
        <Sidebar
          activeTab={activeTab}
          onTabChange={(tab) => setActiveTab(tab as 'overview' | 'lab' | 'session' | 'comparison' | 'heatmap' | 'portfolio' | 'marketAdmin' | 'optimizer')}
          activeSessions={activeSessions}
          onSessionSelect={handleOpenSession}
          hasSelectedSession={!!primarySessionId}
        />
      }
      header={
        <div className="glass flex flex-col gap-4 px-5 py-4 lg:flex-row lg:items-end lg:justify-between">
          <PageHeader
            eyebrow="Modernized Operator View"
            title={TITLES[activeTab]}
            description={
              activeTab === 'overview'
                ? '统一查看策略运行、最近会话与系统状态。'
                : activeTab === 'lab'
                  ? '发起新任务、管理模拟流程，并整理实验结果。'
                  : activeTab === 'marketAdmin'
                    ? '查看行情数据库覆盖、批次历史和手动更新控制台。'
                  : activeTab === 'session'
                    ? '在同一工作区查看执行过程、风险分析、归因结果与日志。'
                    : '围绕同一套控制台视觉语言呈现分析与工具能力。'
            }
          />
          <div className="flex flex-wrap items-center gap-3">
            <div className="rounded-2xl border border-border/70 bg-secondary/45 px-4 py-3">
              <div className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">连接状态</div>
              <div className="mt-1 flex items-center gap-2 text-sm font-medium text-foreground">
                <StatusBadge value={isConnected || usePolling ? 'running' : 'failed'} />
                <span>{API_BASE.replace(/^https?:\/\//, '')}</span>
              </div>
            </div>
            <div className="rounded-2xl border border-border/70 bg-secondary/45 px-4 py-3">
              <div className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">已选会话</div>
              <div className="mt-1 text-lg font-semibold text-foreground">{selectedSessionIds.length}</div>
            </div>
          </div>
        </div>
      }
    >
      <Suspense fallback={<TabFallback />}>
        {activeTab === 'overview' && (
          <GlobalOverview
            sessions={sessions}
            activeSessions={activeSessions}
            primarySession={primarySession}
            onOpenSession={handleOpenSession}
            onOpenLab={() => setActiveTab('lab')}
          />
        )}

        {activeTab === 'lab' && (
          <LabPanel
            strategies={strategies}
            sessions={sessions}
            selectedSessionIds={selectedSessionIds}
            onStart={startSession}
            onToggleSelection={toggleSession}
            onViewSession={handleOpenSession}
            onStopSession={stopSession}
            onDeleteSession={deleteSession}
            error={error}
            onOpenMarketAdmin={() => setActiveTab('marketAdmin')}
          />
        )}

        {activeTab === 'marketAdmin' && <MarketAdminPanel />}

        {activeTab === 'session' && (
          <SessionDetail
            primarySession={primarySession}
            allSessions={sessions}
            equityHistory={equityHistory}
            trades={trades}
            positions={positions}
            comparisonData={comparisonData}
            benchmarksData={benchmarksData}
            selectedBenchmarks={selectedBenchmarks}
            onToggleBenchmark={toggleBenchmark}
            availableBenchmarks={AVAILABLE_BENCHMARKS}
            onSelectSession={handleOpenSession}
            onRestoreCheckpoint={() => primarySessionId && fetchSessionDataFull(primarySessionId)}
          />
        )}

        {activeTab === 'comparison' && (
          <Comparison
            selectedSessionIds={selectedSessionIds}
            sessionDataCache={sessionDataCache}
            allSessions={sessions}
            benchmarksData={benchmarksData}
            availableBenchmarks={AVAILABLE_BENCHMARKS}
          />
        )}

        {activeTab === 'heatmap' && <IndustryHeatmap />}
        {activeTab === 'portfolio' && <PortfolioManager />}
        {activeTab === 'optimizer' && <OptimizerPanel />}
      </Suspense>
    </AppShell>
  );
};

export default App;
