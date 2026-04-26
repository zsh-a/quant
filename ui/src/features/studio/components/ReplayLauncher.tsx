/**
 * Brooks Studio replay launcher.
 *
 * Submits a (symbol, interval, start, end, analyst) tuple to
 * `POST /brooks-studio/replay`, polls the session status while the
 * Celery task walks the historical window, then navigates to the
 * resulting Studio session once the timeline is fully persisted.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { History } from 'lucide-react';
import { toast } from 'sonner';

import { SectionCard } from '../../../components/layout/SectionCard';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { Progress } from '../../../components/ui/progress';
import { apiFetch } from '../../../utils/api';
import { startReplay } from '../api';

type ReplayPhase = 'idle' | 'submitting' | 'running' | 'done' | 'failed';

interface ProgressInfo {
  status: string;
  progress: number;
  error?: string | null;
}

const POLL_INTERVAL_MS = 1500;

function isoDaysAgo(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() - days);
  return d.toISOString().slice(0, 16); // datetime-local format
}

function isoNow(): string {
  return new Date().toISOString().slice(0, 16);
}

export default function ReplayLauncher() {
  const navigate = useNavigate();
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [interval, setInterval] = useState('5m');
  const [analyst, setAnalyst] = useState('rule');
  const [start, setStart] = useState(isoDaysAgo(2));
  const [end, setEnd] = useState(isoNow());
  const [phase, setPhase] = useState<ReplayPhase>('idle');
  const [progressInfo, setProgressInfo] = useState<ProgressInfo>({ status: '', progress: 0 });
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const pollRef = useRef<number | null>(null);

  const stopPolling = useCallback(() => {
    if (pollRef.current !== null) {
      window.clearInterval(pollRef.current);
      pollRef.current = null;
    }
  }, []);

  useEffect(() => {
    return () => stopPolling();
  }, [stopPolling]);

  const pollStatus = useCallback(
    async (sessionId: string): Promise<void> => {
      try {
        const resp = await apiFetch(`/session/${sessionId}/status`);
        if (!resp.ok) return;
        const data = (await resp.json()) as {
          status?: string;
          progress?: number;
          error?: string | null;
        };
        const status = data.status ?? '';
        const progress = Number(data.progress ?? 0);
        setProgressInfo({ status, progress, error: data.error });

        if (status === 'stopped' || status === 'completed') {
          stopPolling();
          setPhase('done');
          toast.success(`Replay finished — ${sessionId.slice(0, 8)}`);
          navigate(`/studio/${sessionId}`);
        } else if (status === 'failed') {
          stopPolling();
          setPhase('failed');
          toast.error(`Replay failed: ${data.error || 'unknown error'}`);
        }
      } catch {
        /* transient — try again next tick */
      }
    },
    [navigate, stopPolling],
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      stopPolling();
      setPhase('submitting');
      setProgressInfo({ status: 'submitting', progress: 0 });
      try {
        const resp = await startReplay({
          symbol,
          interval,
          analyst,
          start: new Date(start).toISOString(),
          end: new Date(end).toISOString(),
        });
        setActiveSessionId(resp.session_id);
        setPhase('running');
        setProgressInfo({ status: 'starting', progress: 0 });
        pollRef.current = window.setInterval(() => {
          void pollStatus(resp.session_id);
        }, POLL_INTERVAL_MS);
      } catch (err) {
        setPhase('failed');
        const message = err instanceof Error ? err.message : String(err);
        toast.error(`Failed to start replay: ${message}`);
      }
    },
    [analyst, end, interval, pollStatus, start, stopPolling, symbol],
  );

  const isBusy = phase === 'submitting' || phase === 'running';

  return (
    <SectionCard
      title="Replay a historical window"
      description="Run the same Brooks pipeline against ClickHouse historical bars. Same Studio panels, no live tail."
      action={
        <Button size="sm" onClick={(e) => void handleSubmit(e as unknown as React.FormEvent)} disabled={isBusy}>
          <History size={14} className="mr-1" />
          {phase === 'submitting'
            ? 'Submitting…'
            : phase === 'running'
              ? `Running ${progressInfo.progress.toFixed(0)}%`
              : 'Run replay'}
        </Button>
      }
    >
      <form className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5" onSubmit={(e) => void handleSubmit(e)}>
        <label className="text-xs text-muted-foreground">
          Symbol
          <Input className="mt-1" value={symbol} onChange={(e) => setSymbol(e.target.value)} disabled={isBusy} />
        </label>
        <label className="text-xs text-muted-foreground">
          Interval
          <Input className="mt-1" value={interval} onChange={(e) => setInterval(e.target.value)} disabled={isBusy} />
        </label>
        <label className="text-xs text-muted-foreground">
          Analyst
          <Input className="mt-1" value={analyst} onChange={(e) => setAnalyst(e.target.value)} disabled={isBusy} />
        </label>
        <label className="text-xs text-muted-foreground">
          Start (UTC)
          <Input
            type="datetime-local"
            className="mt-1"
            value={start}
            onChange={(e) => setStart(e.target.value)}
            disabled={isBusy}
          />
        </label>
        <label className="text-xs text-muted-foreground">
          End (UTC)
          <Input
            type="datetime-local"
            className="mt-1"
            value={end}
            onChange={(e) => setEnd(e.target.value)}
            disabled={isBusy}
          />
        </label>
      </form>
      {phase === 'running' && (
        <div className="mt-4 space-y-2">
          <Progress value={progressInfo.progress} />
          <p className="text-xs text-muted-foreground">
            session {activeSessionId ? activeSessionId.slice(0, 8) : ''} · status {progressInfo.status} ·{' '}
            {progressInfo.progress.toFixed(1)}%
          </p>
        </div>
      )}
      {phase === 'failed' && progressInfo.error && (
        <p className="mt-3 text-xs text-destructive">replay failed: {progressInfo.error}</p>
      )}
    </SectionCard>
  );
}
