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
        <MetricCard label="会话总数" value={sessions.length} hint="包含历史记录与当前运行任务" />
        <MetricCard label="运行中" value={<span className="text-primary">{activeSessions.length}</span>} hint="正在执行的回测或实时任务" />
        <MetricCard label="已完成" value={completedSessions.length} hint="已持久化结果的完成会话" />
        <MetricCard label="模拟任务" value={simulationSessions.length} hint="由自动化流程触发的会话" />
      </div>

      <div className={`grid gap-6 ${primarySession ? 'xl:grid-cols-[minmax(0,1.35fr)_380px]' : ''}`}>
        <SectionCard
          title="平台总览"
          description="快速进入实验室、重新打开最近会话，或查看当前运行状态。"
          action={<Button onClick={onOpenLab}>进入实验室</Button>}
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
                      <span className="text-sm text-muted-foreground">进度</span>
                      <span className="text-sm font-semibold text-foreground">{session.progress?.toFixed?.(0) ?? session.progress}%</span>
                    </div>
                    <div className="mt-3">
                      <Progress value={session.progress || 0} />
                    </div>
                    <div className="mt-4">
                      <Button className="w-full" variant="outline" size="sm" onClick={() => onOpenSession(session.id)}>
                        查看会话
                      </Button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
            {recentSessions.length === 0 && (
              <div className="rounded-2xl border border-dashed border-border/70 bg-card/40 p-5 text-sm text-muted-foreground">
                还没有会话，前往实验室创建第一个运行任务。
              </div>
            )}
          </div>
        </SectionCard>

        {primarySession && (
          <SectionCard title="当前会话" description="当前聚焦的主会话会固定显示在右侧。">
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
              打开会话详情
            </Button>
          </SectionCard>
        )}
      </div>
    </div>
  );
};

export default GlobalOverview;
