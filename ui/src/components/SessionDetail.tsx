import React from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { FileDown } from 'lucide-react';
import type { BenchmarkData, EquityPoint, Position, SessionSummary, Trade } from '../types';
import Dashboard from './Dashboard';
import { RiskPanel } from './RiskPanel';
import { CheckpointList } from './CheckpointList';
import { AttributionPanel } from './AttributionPanel';
import { StrategyLogViewer } from './StrategyLogViewer';
import { LiveTradingPanel } from './LiveTradingPanel';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Button } from './ui/button';
import { EmptyState } from './layout/EmptyState';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';
import { Progress } from './ui/progress';
import { API_BASE, getToken } from '../utils/api';

interface SessionDetailProps {
  primarySession?: SessionSummary;
  allSessions: SessionSummary[];
  equityHistory: EquityPoint[];
  trades: Trade[];
  positions: Record<string, Position>;
  comparisonData: { id: string; name: string; data: EquityPoint[] }[];
  benchmarksData: Record<string, BenchmarkData[]>;
  selectedBenchmarks: string[];
  onToggleBenchmark: (code: string) => void;
  availableBenchmarks: { code: string; name: string }[];
  onSelectSession: (id: string) => void;
  onRestoreCheckpoint: () => void;
}

function getSubtab(pathname: string): 'overview' | 'trading' | 'risk' | 'analysis' | 'logs' {
  if (pathname.endsWith('/trading')) return 'trading';
  if (pathname.endsWith('/risk')) return 'risk';
  if (pathname.endsWith('/analysis')) return 'analysis';
  if (pathname.endsWith('/logs')) return 'logs';
  return 'overview';
}

const SessionDetail: React.FC<SessionDetailProps> = ({
  primarySession,
  allSessions,
  equityHistory,
  trades,
  positions,
  comparisonData,
  benchmarksData,
  selectedBenchmarks,
  onToggleBenchmark,
  availableBenchmarks,
  onSelectSession,
  onRestoreCheckpoint,
}) => {
  const location = useLocation();
  const navigate = useNavigate();
  const { id } = useParams<{ id: string }>();
  const subtab = getSubtab(location.pathname);

  if (!primarySession) {
    return (
      <EmptyState title="No Session Selected" description="Select a session from the overview or lab to view details." />
    );
  }

  const handleTabChange = (value: string) => {
    const base = `/session/${id}`;
    const path = value === 'overview' ? base : `${base}/${value}`;
    navigate(path, { replace: true });
  };

  return (
    <div className="space-y-6">
      <SectionCard
        title="Session Details"
        description={`${primarySession.strategy} · ${primarySession.symbol}`}
        action={
          <div className="min-w-[280px] space-y-2">
            <div className="flex items-center justify-between">
              <div className="text-xs font-medium text-muted-foreground">Switch Session</div>
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 text-xs"
                onClick={() => {
                  const token = getToken();
                  const qs = token ? `?token=${encodeURIComponent(token)}` : '';
                  window.open(`${API_BASE}/analysis/report/${primarySession.id}/pdf${qs}`, '_blank');
                }}
              >
                <FileDown size={13} />
                PDF Report
              </Button>
            </div>
            <select className="glass-input" value={primarySession.id} onChange={(e) => onSelectSession(e.target.value)}>
              {allSessions.map((session) => (
                <option key={session.id} value={session.id}>
                  {session.strategy} - {session.mode} ({session.id.slice(0, 6)}...)
                </option>
              ))}
            </select>
          </div>
        }
      >
        <div className="grid gap-4 lg:grid-cols-[1fr_auto] lg:items-end">
          <div className="space-y-3">
            <div className="flex flex-wrap gap-2">
              <StatusBadge value={primarySession.mode} />
              <StatusBadge value={primarySession.status} />
              <StatusBadge value={primarySession.source || 'manual'} />
            </div>
            <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2 xl:grid-cols-4">
              <div>Symbol: {primarySession.symbol}</div>
              <div>Source: {primarySession.source || 'manual'}</div>
              <div>Run: {primarySession.run_id ? `${primarySession.run_id.slice(0, 8)}...` : 'N/A'}</div>
              <div>Last processed: {primarySession.last_processed_at || 'N/A'}</div>
            </div>
          </div>
          <div className="min-w-[260px] space-y-2">
            <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-muted-foreground">
              <span>Progress</span>
              <span className="tabular-nums">{(primarySession.progress || 0).toFixed(0)}%</span>
            </div>
            <Progress value={primarySession.progress || 0} />
          </div>
        </div>
      </SectionCard>

      <Tabs value={subtab} onValueChange={handleTabChange}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          {(primarySession.mode === 'live' || primarySession.mode === 'paper') && (
            <TabsTrigger value="trading">Trading</TabsTrigger>
          )}
          <TabsTrigger value="risk">Risk</TabsTrigger>
          <TabsTrigger value="analysis">Analysis</TabsTrigger>
          <TabsTrigger value="logs">Logs</TabsTrigger>
        </TabsList>

        <TabsContent value="overview">
          <Dashboard
            primarySession={primarySession}
            equityHistory={equityHistory}
            trades={trades}
            positions={positions}
            comparisonData={comparisonData}
            benchmarksData={benchmarksData}
            selectedBenchmarks={selectedBenchmarks}
            onToggleBenchmark={onToggleBenchmark}
            availableBenchmarks={availableBenchmarks}
            onSelectSession={onSelectSession}
            allSessions={allSessions}
          />
        </TabsContent>

        {(primarySession.mode === 'live' || primarySession.mode === 'paper') && (
          <TabsContent value="trading">
            <LiveTradingPanel sessionId={primarySession.id} />
          </TabsContent>
        )}

        <TabsContent value="risk">
          <div className="grid gap-6">
            <RiskPanel sessionId={primarySession.id} />
            <CheckpointList sessionId={primarySession.id} onRestore={onRestoreCheckpoint} />
          </div>
        </TabsContent>

        <TabsContent value="analysis">
          <AttributionPanel sessionId={primarySession.id} />
        </TabsContent>

        <TabsContent value="logs">
          <StrategyLogViewer sessionId={primarySession.id} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default SessionDetail;
