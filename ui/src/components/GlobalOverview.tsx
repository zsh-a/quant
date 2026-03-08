import React from 'react';
import type { SessionSummary } from '../types';
import { Button } from './ui/button';
import { SectionCard } from './layout/SectionCard';
import { MetricCard } from './layout/MetricCard';
import { StatusBadge } from './layout/StatusBadge';
import { Progress } from './ui/progress';

interface GlobalOverviewProps {
  sessions: SessionSummary[];
  activeSessions: SessionSummary[];
  primarySession?: SessionSummary;
  onOpenSession: (sessionId: string) => void;
  onOpenLab: () => void;
}

const GlobalOverview: React.FC<GlobalOverviewProps> = ({
  sessions,
  activeSessions,
  primarySession,
  onOpenSession,
  onOpenLab,
}) => {
  const completedSessions = sessions.filter((session) => session.status === 'completed');
  const simulationSessions = sessions.filter((session) => session.source === 'automation');
  const recentSessions = sessions.slice(0, 8);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Total Sessions" value={sessions.length} hint="Historic and active runs" />
        <MetricCard label="Running Now" value={<span className="text-primary">{activeSessions.length}</span>} hint="Backtests or live tasks in progress" />
        <MetricCard label="Completed" value={completedSessions.length} hint="Finished with persisted results" />
        <MetricCard label="Simulation Jobs" value={simulationSessions.length} hint="Automation-backed sessions" />
      </div>

      <div className={`grid gap-6 ${primarySession ? 'xl:grid-cols-[minmax(0,1.35fr)_380px]' : ''}`}>
        <SectionCard
          title="Platform Overview"
          description="从这里进入 Lab、打开最近 session，或查看当前运行情况。"
          action={<Button onClick={onOpenLab}>打开 Lab</Button>}
        >
          <div className="grid gap-3">
            {recentSessions.map((session) => (
              <div
                key={session.id}
                className="rounded-2xl border border-border/70 bg-secondary/45 p-4 transition hover:border-primary/30 hover:bg-accent/45"
              >
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="flex flex-wrap items-center gap-2">
                      <div className="text-base font-semibold text-foreground">{session.strategy}</div>
                      <StatusBadge value={session.mode} />
                      <StatusBadge value={session.status} />
                    </div>
                    <div className="text-sm text-muted-foreground">{session.symbol} · {session.source || 'manual'}</div>
                    <div className="text-sm text-muted-foreground">{session.start_date} → {session.end_date || 'Ongoing'}</div>
                  </div>

                  <div className="rounded-2xl border border-border/60 bg-card/75 px-4 py-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <span className="text-sm text-muted-foreground">Progress</span>
                      <span className="text-sm font-semibold text-foreground">{session.progress?.toFixed?.(0) ?? session.progress}%</span>
                    </div>
                    <div className="mt-3">
                      <Progress value={session.progress || 0} />
                    </div>
                    <div className="mt-4">
                      <Button className="w-full" variant="outline" size="sm" onClick={() => onOpenSession(session.id)}>
                        查看详情
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {recentSessions.length === 0 && (
              <div className="rounded-2xl border border-dashed border-border/70 bg-card/40 p-5 text-sm text-muted-foreground">
                还没有 session，去 Lab 创建第一个运行任务。
              </div>
            )}
          </div>
        </SectionCard>

        {primarySession && (
          <SectionCard title="Current Session" description="Primary context pinned into the shell">
            <div className="space-y-3 rounded-2xl border border-border/60 bg-card/75 p-4">
              <div className="text-xl font-semibold text-foreground">{primarySession.strategy}</div>
              <div className="flex flex-wrap gap-2">
                <StatusBadge value={primarySession.mode} />
                <StatusBadge value={primarySession.status} />
              </div>
              <div className="text-sm text-muted-foreground">{primarySession.symbol} · {primarySession.source || 'manual'}</div>
              <div className="text-sm text-muted-foreground">进度：{(primarySession.progress || 0).toFixed(0)}%</div>
              <Progress value={primarySession.progress || 0} />
            </div>
            {primarySession.last_processed_at && (
              <div className="text-sm text-muted-foreground">最近处理到：{primarySession.last_processed_at}</div>
            )}
            <Button className="mt-2 w-full" onClick={() => onOpenSession(primarySession.id)}>
              打开 Session Detail
            </Button>
          </SectionCard>
        )}
      </div>
    </div>
  );
};

export default GlobalOverview;
