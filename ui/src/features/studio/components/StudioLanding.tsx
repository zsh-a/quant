/**
 * Brooks Studio landing — picker / launcher rendered for `/studio` (no
 * session id). Lists every Brooks session (live + replay, running and
 * finished) read from the persistent session store, plus launchers for
 * starting a fresh live or replay session.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Play, Square } from 'lucide-react';
import { toast } from 'sonner';

import { SectionCard } from '../../../components/layout/SectionCard';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { apiFetch } from '../../../utils/api';
import ReplayLauncher from './ReplayLauncher';

interface SessionSummary {
  session_id: string;
  strategy_name: string;
  symbol: string;
  status: string;
  interval: string;
  created_at: string;
  params: Record<string, unknown>;
}

const RUNNING_STATUSES = new Set(['running', 'starting']);

function sessionKindOf(s: SessionSummary): 'live' | 'replay' {
  return (s.params?.session_kind as string | undefined) === 'replay' ? 'replay' : 'live';
}

function analystOf(s: SessionSummary): string {
  return String(s.params?.analyst ?? 'rule');
}

function statusBadgeClass(status: string): string {
  if (RUNNING_STATUSES.has(status)) {
    return 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30';
  }
  if (status === 'failed') {
    return 'bg-rose-500/15 text-rose-300 border border-rose-500/30';
  }
  return 'bg-muted/40 text-muted-foreground border border-border/60';
}

function kindBadgeClass(kind: 'live' | 'replay'): string {
  return kind === 'live'
    ? 'bg-sky-500/15 text-sky-300 border border-sky-500/30'
    : 'bg-violet-500/15 text-violet-300 border border-violet-500/30';
}

export default function StudioLanding() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [interval, setInterval] = useState('5m');
  const [analyst, setAnalyst] = useState('rule');
  const [starting, setStarting] = useState(false);

  const refreshSessions = useCallback(async () => {
    try {
      const resp = await apiFetch('/sessions');
      if (!resp.ok) return;
      const data = (await resp.json()) as SessionSummary[];
      const brooksOnly = (data ?? []).filter((s) => s.strategy_name === 'brooks');
      setSessions(brooksOnly.slice(0, 30));
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refreshSessions();
    const timer = window.setInterval(refreshSessions, 10_000);
    return () => clearInterval(timer);
  }, [refreshSessions]);

  const handleStart = useCallback(async () => {
    setStarting(true);
    try {
      const resp = await apiFetch('/brooks-live/start', {
        method: 'POST',
        body: JSON.stringify({ symbol, interval, analyst, mode: 'paper' }),
      });
      if (!resp.ok) {
        const err = await resp.text();
        throw new Error(err || 'start failed');
      }
      const data = await resp.json();
      toast.success(`Session started: ${data.session_id.slice(0, 8)}`);
      navigate(`/studio/${data.session_id}`);
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }, [symbol, interval, analyst, navigate]);

  const handleStop = useCallback(
    async (sessionId: string) => {
      try {
        const resp = await apiFetch(`/brooks-live/${sessionId}/stop`, { method: 'POST' });
        if (!resp.ok) {
          const err = await resp.text();
          throw new Error(err || 'stop failed');
        }
        toast.success(`Stop requested: ${sessionId.slice(0, 8)}`);
        // Give the worker a moment to flip the row to "stopped".
        setTimeout(refreshSessions, 1500);
      } catch (e) {
        toast.error(e instanceof Error ? e.message : String(e));
      }
    },
    [refreshSessions],
  );

  return (
    <div className="space-y-5 px-1 py-1" data-testid="studio-landing">
      <SectionCard
        title="Start a Brooks Studio session"
        description="Paper trading only. Live execution stays disabled in this build."
        action={
          <Button size="sm" onClick={handleStart} disabled={starting}>
            <Play size={14} className="mr-1" />
            {starting ? 'Starting…' : 'Start session'}
          </Button>
        }
      >
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-xs text-muted-foreground">
            Symbol
            <Input className="mt-1" value={symbol} onChange={(e) => setSymbol(e.target.value)} />
          </label>
          <label className="text-xs text-muted-foreground">
            Interval
            <Input className="mt-1" value={interval} onChange={(e) => setInterval(e.target.value)} />
          </label>
          <label className="text-xs text-muted-foreground">
            Analyst
            <Input className="mt-1" value={analyst} onChange={(e) => setAnalyst(e.target.value)} />
          </label>
        </div>
      </SectionCard>

      <ReplayLauncher />

      <SectionCard
        title="Sessions"
        description="Open a Brooks Studio session — live or replay. Newest first."
      >
        {sessions.length === 0 ? (
          <div className="rounded-md border border-dashed border-border/60 px-4 py-8 text-center text-xs text-muted-foreground">
            No sessions yet. Start one above to open Studio.
          </div>
        ) : (
          <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {sessions.map((s) => {
              const kind = sessionKindOf(s);
              const isRunning = RUNNING_STATUSES.has(s.status);
              return (
                <li
                  key={s.session_id}
                  className="group relative rounded-md border border-border/60 bg-card/60 transition-colors hover:border-primary/60 hover:bg-accent/40"
                >
                  <button
                    type="button"
                    onClick={() => navigate(`/studio/${s.session_id}`)}
                    className="flex w-full items-center justify-between gap-2 px-3 py-2 text-left text-xs"
                    data-testid={`studio-landing-session-${s.session_id}`}
                  >
                    <span className="flex min-w-0 flex-1 flex-col gap-1">
                      <span className="flex items-center gap-1.5">
                        <span className={`rounded px-1 py-0.5 text-[10px] font-medium uppercase ${kindBadgeClass(kind)}`}>
                          {kind}
                        </span>
                        <span className={`rounded px-1 py-0.5 text-[10px] font-medium uppercase ${statusBadgeClass(s.status)}`}>
                          {s.status}
                        </span>
                        <span className="truncate font-medium text-foreground tabular-nums">
                          {s.session_id.slice(0, 8)}
                        </span>
                      </span>
                      <span className="truncate text-muted-foreground">
                        {s.symbol} · {s.interval} · {analystOf(s)}
                      </span>
                    </span>
                    <ArrowRight className="size-3 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                  </button>
                  {isRunning && kind === 'live' && (
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        void handleStop(s.session_id);
                      }}
                      title="Stop live session"
                      aria-label={`Stop live session ${s.session_id}`}
                      className="absolute right-2 top-2 rounded p-1 text-muted-foreground opacity-0 transition-opacity hover:bg-rose-500/15 hover:text-rose-300 group-hover:opacity-100"
                      data-testid={`studio-landing-stop-${s.session_id}`}
                    >
                      <Square size={12} />
                    </button>
                  )}
                </li>
              );
            })}
          </ul>
        )}
      </SectionCard>
    </div>
  );
}
