/**
 * Brooks Studio landing — picker / launcher rendered for `/studio` (no
 * session id). Lists active paper-trading sessions and lets the user start
 * a new one. Selecting a session navigates to `/studio/:sessionId`.
 *
 * Replaces the legacy BrooksLive launcher screen so users land here when
 * they hit the rewritten `/brooks-live` routes.
 */

import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, Play } from 'lucide-react';
import { toast } from 'sonner';

import { SectionCard } from '../../../components/layout/SectionCard';
import { Button } from '../../../components/ui/button';
import { Input } from '../../../components/ui/input';
import { apiFetch } from '../../../utils/api';

interface BrooksSessionSummary {
  session_id: string;
  analyst: string;
  config: Record<string, unknown>;
  started_at: string;
}

export default function StudioLanding() {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<BrooksSessionSummary[]>([]);
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [interval, setInterval] = useState('5m');
  const [analyst, setAnalyst] = useState('rule');
  const [starting, setStarting] = useState(false);

  const refreshActive = useCallback(async () => {
    try {
      const resp = await apiFetch('/brooks-live/sessions');
      if (!resp.ok) return;
      const data = await resp.json();
      setSessions(data.sessions || []);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refreshActive();
    const timer = window.setInterval(refreshActive, 10_000);
    return () => clearInterval(timer);
  }, [refreshActive]);

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
            <Input
              className="mt-1"
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
            />
          </label>
          <label className="text-xs text-muted-foreground">
            Analyst
            <Input className="mt-1" value={analyst} onChange={(e) => setAnalyst(e.target.value)} />
          </label>
        </div>
      </SectionCard>

      <SectionCard
        title="Active sessions"
        description="Select a running paper-trading session to inspect live + replay in Studio."
      >
        {sessions.length === 0 ? (
          <div className="rounded-md border border-dashed border-border/60 px-4 py-8 text-center text-xs text-muted-foreground">
            No active sessions. Start one above to open Studio.
          </div>
        ) : (
          <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
            {sessions.map((s) => (
              <li key={s.session_id}>
                <button
                  type="button"
                  onClick={() => navigate(`/studio/${s.session_id}`)}
                  className="group flex w-full items-center justify-between gap-2 rounded-md border border-border/60 bg-card/60 px-3 py-2 text-left text-xs transition-colors hover:border-primary/60 hover:bg-accent/40"
                  data-testid={`studio-landing-session-${s.session_id}`}
                >
                  <span className="flex min-w-0 flex-1 flex-col">
                    <span className="truncate font-medium text-foreground tabular-nums">
                      {s.session_id.slice(0, 8)}
                    </span>
                    <span className="truncate text-muted-foreground">
                      {s.analyst} · {String((s.config as { symbol?: string }).symbol ?? '?')}
                    </span>
                  </span>
                  <ArrowRight className="size-3 text-muted-foreground transition-transform group-hover:translate-x-0.5" />
                </button>
              </li>
            ))}
          </ul>
        )}
      </SectionCard>
    </div>
  );
}
