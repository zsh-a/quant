import React, { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import type { SessionSummary } from '../types';
import { apiFetch } from '../utils/api';
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
  const completedSessions = sessions.filter((s) => s.status === 'completed');
  const simulationSessions = sessions.filter((s) => s.source === 'automation');
  const recentSessions = sessions.slice(0, 12);

  const [regime, setRegime] = useState<{ regime: string; confidence: number; annualized_vol: number } | null>(null);
  useEffect(() => {
    apiFetch('/market/regime?symbol=sh.000300')
      .then((r) => r.ok ? r.json() : null)
      .then((d) => d && setRegime(d))
      .catch(() => {});
  }, []);

  const regimeIcon = regime?.regime === 'bull'
    ? <TrendingUp size={15} className="text-emerald-500" />
    : regime?.regime === 'bear'
      ? <TrendingDown size={15} className="text-red-500" />
      : <Minus size={15} className="text-amber-500" />;
  const regimeLabel = regime?.regime === 'bull' ? 'Bull' : regime?.regime === 'bear' ? 'Bear' : 'Sideways';
  const regimeColor = regime?.regime === 'bull' ? 'text-emerald-500' : regime?.regime === 'bear' ? 'text-red-500' : 'text-amber-500';

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
        <MetricCard label="Total Sessions" value={sessions.length} hint="Including history and active tasks" />
        <MetricCard label="Running" value={<span className="text-primary">{activeSessions.length}</span>} hint="Currently executing backtests or live tasks" />
        <MetricCard label="Completed" value={completedSessions.length} hint="Sessions with persisted results" />
        <MetricCard label="Simulations" value={simulationSessions.length} hint="Sessions triggered by automation" />
        <MetricCard
          label="Market Regime"
          value={
            regime ? (
              <span className={`flex items-center gap-1.5 ${regimeColor}`}>
                {regimeIcon} {regimeLabel}
              </span>
            ) : '--'
          }
          hint={regime ? `Vol: ${(regime.annualized_vol * 100).toFixed(1)}% · Confidence: ${(regime.confidence * 100).toFixed(0)}%` : 'HS300 regime detection'}
        />
      </div>

      {primarySession && (
        <div className="flex items-center gap-4 rounded-2xl border border-border/60 bg-card/75 px-5 py-3">
          <div className="flex items-center gap-3">
            <span className="text-sm text-muted-foreground">Current</span>
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
            <span className="text-xs text-muted-foreground">processed to {primarySession.last_processed_at}</span>
          )}
          <div className="ml-auto">
            <Button variant="outline" size="sm" onClick={() => onOpenSession(primarySession.id)}>Open Details</Button>
          </div>
        </div>
      )}

      <div>
        <SectionCard
          title="Recent Sessions"
          description="Quick access to the lab, reopen recent sessions, or check run status."
          action={<Button onClick={onOpenLab}>Open Lab</Button>}
        >
          <div className="overflow-x-auto rounded-xl border border-border/70">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/60 bg-secondary/30 text-left text-xs font-medium uppercase tracking-wider text-muted-foreground">
                  <th className="px-4 py-2.5">Strategy</th>
                  <th className="px-4 py-2.5">Symbol</th>
                  <th className="px-4 py-2.5">Period</th>
                  <th className="px-4 py-2.5">Source</th>
                  <th className="px-4 py-2.5">Status</th>
                  <th className="px-4 py-2.5 text-right">Progress</th>
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
                    <td className="whitespace-nowrap px-4 py-2.5 text-muted-foreground">{session.source || 'manual'}</td>
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
                        View
                      </Button>
                    </td>
                  </tr>
                ))}
                {recentSessions.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-8 text-center text-muted-foreground">
                      No sessions yet. Go to the Lab to create your first task.
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
