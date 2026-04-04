import React from 'react';
import type { SessionSummary } from '../types';
import { Button } from './ui/button';
import { SectionCard } from './layout/SectionCard';
import { MetricCard } from './layout/MetricCard';
import { StatusBadge } from './layout/StatusBadge';
import { Progress } from './ui/progress';
import { formatSourceLabel } from '../utils/display';

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
  const completedSessions = sessions.filter((s) => s.status === 'completed');
  const simulationSessions = sessions.filter((s) => s.source === 'automation');
  const recentSessions = sessions.slice(0, 12);

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="会话总数" value={sessions.length} hint="包含历史记录与当前运行任务" />
        <MetricCard label="运行中" value={<span className="text-primary">{activeSessions.length}</span>} hint="正在执行的回测或实时任务" />
        <MetricCard label="已完成" value={completedSessions.length} hint="已持久化结果的完成会话" />
        <MetricCard label="模拟任务" value={simulationSessions.length} hint="由自动化流程触发的会话" />
      </div>

      {primarySession && (
        <div className="flex items-center gap-4 rounded-2xl border border-border/60 bg-card/75 px-5 py-3">
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">当前会话</span>
            <span className="font-semibold text-foreground">{primarySession.strategy}</span>
            <StatusBadge value={primarySession.mode} />
            <StatusBadge value={primarySession.status} />
          </div>
          <span className="text-sm text-muted-foreground">{primarySession.symbol}</span>
          <div className="flex items-center gap-2">
            <Progress value={primarySession.progress || 0} className="h-1.5 w-20" />
            <span className="text-xs tabular-nums text-muted-foreground">{(primarySession.progress || 0).toFixed(0)}%</span>
          </div>
          {primarySession.last_processed_at && (
            <span className="text-xs text-muted-foreground">处理到 {primarySession.last_processed_at}</span>
          )}
          <div className="ml-auto">
            <Button variant="outline" size="sm" onClick={() => onOpenSession(primarySession.id)}>打开详情</Button>
          </div>
        </div>
      )}

      <div>
        <SectionCard
          title="最近会话"
          description="快速进入实验室、重新打开最近会话，或查看当前运行状态。"
          action={<Button onClick={onOpenLab}>进入实验室</Button>}
        >
          <div className="overflow-x-auto rounded-xl border border-border/70">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-secondary/30 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-2.5">策略</th>
                  <th className="px-4 py-2.5">标的</th>
                  <th className="px-4 py-2.5">区间</th>
                  <th className="px-4 py-2.5">来源</th>
                  <th className="px-4 py-2.5">状态</th>
                  <th className="px-4 py-2.5 text-right">进度</th>
                  <th className="px-4 py-2.5" />
                </tr>
              </thead>
              <tbody>
                {recentSessions.map((session) => (
                  <tr
                    key={session.id}
                    className="border-b border-border/40 transition hover:bg-accent/40 cursor-pointer"
                    onClick={() => onOpenSession(session.id)}
                  >
                    <td className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-foreground">{session.strategy}</span>
                        <StatusBadge value={session.mode} />
                      </div>
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">{session.symbol}</td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">
                      {session.start_date} → {session.end_date || '...'}
                    </td>
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">{formatSourceLabel(session.source)}</td>
                    <td className="px-4 py-2.5">
                      <StatusBadge value={session.status} />
                    </td>
                    <td className="px-4 py-2.5">
                      <div className="flex items-center justify-end gap-2">
                        <Progress value={session.progress || 0} className="h-1.5 w-16" />
                        <span className="w-10 text-right text-xs tabular-nums text-muted-foreground">
                          {session.progress?.toFixed?.(0) ?? 0}%
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-2.5 text-right">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => { e.stopPropagation(); onOpenSession(session.id); }}
                      >
                        查看
                      </Button>
                    </td>
                  </tr>
                ))}
                {recentSessions.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">
                      还没有会话，前往实验室创建第一个运行任务。
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </SectionCard>
      </div>
    </div>
  );
};

export default GlobalOverview;
