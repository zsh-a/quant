import React, { useEffect, useState } from 'react';
import type { BenchmarkData, EquityPoint, Position, SessionSummary, Trade } from '../types';
import Dashboard from './Dashboard';
import SessionExecutionPanel from './SessionExecutionPanel';
import { RiskPanel } from './RiskPanel';
import { CheckpointList } from './CheckpointList';
import { AttributionPanel } from './AttributionPanel';
import { StrategyLogViewer } from './StrategyLogViewer';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { EmptyState } from './layout/EmptyState';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';
import { Progress } from './ui/progress';

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
  const [subtab, setSubtab] = useState<'overview' | 'execution' | 'risk' | 'analysis' | 'logs'>(() => {
    const stored = window.localStorage.getItem('quent.session.subtab');
    if (stored === 'execution' || stored === 'risk' || stored === 'analysis' || stored === 'logs') return stored;
    return 'overview';
  });

  useEffect(() => {
    window.localStorage.setItem('quent.session.subtab', subtab);
  }, [subtab]);

  if (!primarySession) {
    return (
      <EmptyState title="No Session Selected" description="从 Overview 或 Lab 选择一个 session 查看详情。" />
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="Session Detail"
        description={`${primarySession.strategy} · ${primarySession.symbol} · ${primarySession.source || 'manual'}`}
        action={
          <div className="min-w-[280px] space-y-2">
            <div className="tagline">切换 Session</div>
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
            </div>
            <div className="text-sm text-muted-foreground">
              状态：{primarySession.status} · 进度：{(primarySession.progress || 0).toFixed(0)}%
            </div>
          </div>
          <div className="min-w-[260px] space-y-2">
            <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-muted-foreground">
              <span>Execution Progress</span>
              <span>{(primarySession.progress || 0).toFixed(0)}%</span>
            </div>
            <Progress value={primarySession.progress || 0} />
          </div>
        </div>
      </SectionCard>

      <Tabs value={subtab} onValueChange={(value) => setSubtab(value as 'overview' | 'execution' | 'risk' | 'analysis' | 'logs')}>
        <TabsList>
          <TabsTrigger value="overview">Overview</TabsTrigger>
          <TabsTrigger value="execution">Execution</TabsTrigger>
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

        <TabsContent value="execution">
          <SessionExecutionPanel session={primarySession} trades={trades} />
        </TabsContent>

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
