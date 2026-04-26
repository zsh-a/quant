import { Suspense, lazy, useEffect, useMemo } from 'react';
import { Routes, Route, Navigate, useNavigate, useLocation } from 'react-router-dom';
import './App.css';

import { useTheme } from './hooks/useTheme';
import { useStrategies } from './hooks/useStrategies';
import { useSessionData } from './hooks/useSessionData';
import { useBenchmarks } from './hooks/useBenchmarks';
import { useSessionCrud } from './hooks/useSessionCrud';
import { useSessionWebSocket } from './hooks/useSessionWebSocket';
import {
  useSessions,
  useSelectedSessions,
  usePrimarySession,
  useSessionDataCache,
  useSessionActions,
  useSessionStore,
  useActiveSessions,
} from './store';

import Sidebar from './components/Sidebar';
import { ErrorBoundary } from './components/ErrorBoundary';
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
const BrooksStudioPage = lazy(() => import('./features/studio/components/BrooksStudioPage'));
const StudioLanding = lazy(() => import('./features/studio/components/StudioLanding'));

const ROUTE_META: Record<string, { title: string; description: string }> = {
  '/': { title: 'Overview', description: 'Monitor strategy runs, recent sessions, and system status.' },
  '/lab': { title: 'Lab', description: 'Launch tasks, manage simulations, and research alpha factors.' },
  '/session': { title: 'Session', description: 'Inspect execution, risk analysis, attribution, and logs.' },
  '/compare': { title: 'Compare', description: 'Side-by-side strategy performance comparison.' },
  '/heatmap': { title: 'Heatmap', description: 'Industry sector heatmap visualization.' },
  '/portfolio': { title: 'Portfolio', description: 'Portfolio allocation and weighting management.' },
  '/market-admin': { title: 'Market Data', description: 'Database coverage, batch history, and update console.' },
  '/optimizer': { title: 'Optimizer', description: 'Parameter optimization workspace.' },
  '/studio': { title: 'Brooks Studio', description: 'Unified live + replay K-line studio with timeline scrubber.' },
};

function getRouteMeta(pathname: string) {
  if (pathname.startsWith('/session')) return ROUTE_META['/session'];
  if (pathname.startsWith('/lab')) return ROUTE_META['/lab'];
  if (pathname.startsWith('/studio')) return ROUTE_META['/studio'];
  return ROUTE_META[pathname] || ROUTE_META['/'];
}

const TabFallback = () => (
  <div className="glass flex min-h-[320px] items-center justify-center rounded-[28px] border border-border/70 px-6 py-10 text-sm text-muted-foreground">
    Loading module...
  </div>
);

const App = () => {
  const navigate = useNavigate();
  const location = useLocation();
  const { theme, toggle: toggleTheme } = useTheme();

  const sessions = useSessions();
  const selectedSessionIds = useSelectedSessions();
  const primarySessionId = usePrimarySession();
  const sessionDataCache = useSessionDataCache();
  const { selectSession, toggleSession } = useSessionActions();

  const { strategies } = useStrategies();
  const primarySession = sessions.find((s) => s.id === primarySessionId);

  const sessionData = useSessionData(location.pathname);
  const { selectedBenchmarks, benchmarksData, toggleBenchmark, availableBenchmarks } =
    useBenchmarks(primarySession);

  const handleOpenSession = (id: string) => {
    selectSession(id);
    navigate(`/session/${id}`);
  };

  const { startSession, stopSession, deleteSession, fetchSessions } = useSessionCrud({
    onSessionStarted: (id) => navigate(`/session/${id}`),
    onSessionDeleted: () => navigate('/lab'),
    fetchSessionDataFull: sessionData.fetchSessionDataFull,
  });

  const { isConnected, usePolling } = useSessionWebSocket({
    setEquityHistory: sessionData.setEquityHistory,
    setTrades: sessionData.setTrades,
    lastUpdatedRef: sessionData.lastUpdatedRef,
    fetchSessionDetails: sessionData.fetchSessionDetails,
  });

  // Periodic session list refresh
  useEffect(() => {
    fetchSessions();
    const interval = window.setInterval(fetchSessions, 15000);
    return () => clearInterval(interval);
  }, [fetchSessions]);

  // Rehydrate store
  useEffect(() => {
    try {
      useSessionStore.persist.rehydrate();
    } catch (e) {
      console.warn('Store rehydration failed:', e);
    }
  }, []);

  const activeSessions = useActiveSessions();
  const comparisonData = useMemo(
    () =>
      selectedSessionIds
        .filter((id) => id !== primarySessionId && sessionDataCache[id])
        .map((id) => ({
          id,
          name: sessions.find((s) => s.id === id)?.strategy || id,
          data: sessionDataCache[id]?.equity || [],
        })),
    [selectedSessionIds, primarySessionId, sessionDataCache, sessions]
  );

  const meta = useMemo(() => getRouteMeta(location.pathname), [location.pathname]);

  // Determine active nav key from pathname
  const activeNavKey = location.pathname.startsWith('/session')
    ? 'session'
    : location.pathname.startsWith('/lab')
      ? 'lab'
      : location.pathname === '/compare'
        ? 'comparison'
        : location.pathname === '/heatmap'
          ? 'heatmap'
          : location.pathname === '/portfolio'
            ? 'portfolio'
            : location.pathname === '/market-admin'
              ? 'marketAdmin'
              : location.pathname === '/optimizer'
                ? 'optimizer'
                : location.pathname.startsWith('/studio') || location.pathname.startsWith('/brooks-live')
                  ? 'trade'
                  : 'overview';

  return (
    <AppShell
      sidebar={
        <Sidebar
          activeTab={activeNavKey}
          onTabChange={(key) => {
            const routes: Record<string, string> = {
              overview: '/',
              lab: '/lab',
              session: primarySessionId ? `/session/${primarySessionId}` : '/lab',
              comparison: '/compare',
              heatmap: '/heatmap',
              portfolio: '/portfolio',
              marketAdmin: '/market-admin',
              optimizer: '/optimizer',
              trade: '/studio',
            };
            navigate(routes[key] || '/');
          }}
          activeSessions={activeSessions}
          onSessionSelect={handleOpenSession}
          hasSelectedSession={!!primarySessionId}
          theme={theme}
          onToggleTheme={toggleTheme}
        />
      }
      header={
        <div className="flex items-center justify-between gap-4 px-5 py-3">
          <PageHeader title={meta.title} description={meta.description} />
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-[12px] text-muted-foreground">
              <StatusBadge value={!primarySessionId ? 'idle' : isConnected || usePolling ? 'running' : 'failed'} />
              <span>{!primarySessionId ? '待机' : isConnected ? 'WS' : usePolling ? 'Poll' : '断开'}</span>
            </div>
            <div className="h-4 w-px bg-border" />
            <div className="text-[12px] text-muted-foreground">
              <span className="tabular-nums font-medium text-foreground">{selectedSessionIds.length}</span> selected
            </div>
          </div>
        </div>
      }
    >
      <ErrorBoundary>
      <Suspense fallback={<TabFallback />}>
        <Routes>
          <Route
            path="/"
            element={
              <GlobalOverview
                sessions={sessions}
                activeSessions={activeSessions}
                primarySession={primarySession}
                onOpenSession={handleOpenSession}
                onOpenLab={() => navigate('/lab')}
              />
            }
          />
          <Route
            path="/lab/*"
            element={
              <LabPanel
                strategies={strategies}
                sessions={sessions}
                selectedSessionIds={selectedSessionIds}
                onStart={startSession}
                onToggleSelection={toggleSession}
                onViewSession={handleOpenSession}
                onStopSession={stopSession}
                onDeleteSession={deleteSession}
                onOpenMarketAdmin={() => navigate('/market-admin')}
              />
            }
          />
          <Route path="/market-admin" element={<MarketAdminPanel />} />
          <Route
            path="/session/:id/*"
            element={
              <SessionDetail
                primarySession={primarySession}
                allSessions={sessions}
                equityHistory={sessionData.equityHistory}
                trades={sessionData.trades}
                positions={sessionData.positions}
                comparisonData={comparisonData}
                benchmarksData={benchmarksData}
                selectedBenchmarks={selectedBenchmarks}
                onToggleBenchmark={toggleBenchmark}
                availableBenchmarks={availableBenchmarks}
                onSelectSession={handleOpenSession}
                onRestoreCheckpoint={() => primarySessionId && sessionData.fetchSessionDataFull(primarySessionId)}
              />
            }
          />
          <Route
            path="/compare"
            element={
              <Comparison
                selectedSessionIds={selectedSessionIds}
                sessionDataCache={sessionDataCache}
                allSessions={sessions}
                benchmarksData={benchmarksData}
                availableBenchmarks={availableBenchmarks}
              />
            }
          />
          <Route path="/heatmap" element={<IndustryHeatmap />} />
          <Route path="/portfolio" element={<PortfolioManager />} />
          <Route path="/optimizer" element={<OptimizerPanel />} />
          <Route path="/studio" element={<StudioLanding />} />
          <Route path="/studio/:sessionId" element={<BrooksStudioPage />} />
          <Route path="/brooks-live" element={<Navigate to="/studio" replace />} />
          <Route path="/brooks-live/*" element={<Navigate to="/studio" replace />} />
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Suspense>
      </ErrorBoundary>
    </AppShell>
  );
};

export default App;
