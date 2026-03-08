import React, { useEffect, useState } from 'react';
import type { BenchmarkData, EquityPoint, Position, SessionSummary, Trade } from '../types';
import Dashboard from './Dashboard';
import { RiskPanel } from './RiskPanel';
import { CheckpointList } from './CheckpointList';
import { AttributionPanel } from './AttributionPanel';
import { StrategyLogViewer } from './StrategyLogViewer';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { EmptyState } from './layout/EmptyState';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';
import { Progress } from './ui/progress';
import { formatSourceLabel } from '../utils/display';

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
  const [subtab, setSubtab] = useState<'overview' | 'risk' | 'analysis' | 'logs'>(() => {
    const stored = window.localStorage.getItem('quent.session.subtab');
    if (stored === 'risk' || stored === 'analysis' || stored === 'logs') return stored;
    return 'overview';
  });

  useEffect(() => {
    window.localStorage.setItem('quent.session.subtab', subtab);
  }, [subtab]);

  if (!primarySession) {
    return (
      <EmptyState title="未选择会话" description="可从总览或实验室中选择一个会话进入详情视图。" />
    );
  }

  return (
    <div className="space-y-6">
      <SectionCard
        title="会话详情"
        description={`${primarySession.strategy} · ${primarySession.symbol} · ${formatSourceLabel(primarySession.source)}`}
        action={
          <div className="min-w-[280px] space-y-2">
            <div className="tagline">切换会话</div>
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
            <div className="text-sm text-muted-foreground">
              状态：{primarySession.status} · 进度：{(primarySession.progress || 0).toFixed(0)}%
            </div>
            <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2 xl:grid-cols-4">
              <div>标的：{primarySession.symbol}</div>
              <div>来源：{formatSourceLabel(primarySession.source)}</div>
              <div>Run：{primarySession.run_id ? `${primarySession.run_id.slice(0, 8)}...` : '未关联'}</div>
              <div>最近处理：{primarySession.last_processed_at || '暂无'}</div>
            </div>
          </div>
          <div className="min-w-[260px] space-y-2">
            <div className="flex items-center justify-between text-xs uppercase tracking-[0.22em] text-muted-foreground">
              <span>执行进度</span>
              <span>{(primarySession.progress || 0).toFixed(0)}%</span>
            </div>
            <Progress value={primarySession.progress || 0} />
          </div>
        </div>
      </SectionCard>

      <Tabs value={subtab} onValueChange={(value) => setSubtab(value as 'overview' | 'risk' | 'analysis' | 'logs')}>
        <TabsList>
          <TabsTrigger value="overview">总览</TabsTrigger>
          <TabsTrigger value="risk">风险</TabsTrigger>
          <TabsTrigger value="analysis">分析</TabsTrigger>
          <TabsTrigger value="logs">日志</TabsTrigger>
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
