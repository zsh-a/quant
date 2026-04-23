/**
 * BrooksLive — top-level Phase 4.6 paper-trading panel.
 *
 * Drives a BrooksStrategy session in paper mode, shows realtime signals,
 * decisions, equity curve, and lets the user hot-swap analysts.
 */

import { useCallback, useEffect, useState } from 'react';
import { AlertCircle, Play, StopCircle } from 'lucide-react';
import { toast } from 'sonner';

import { SectionCard } from '../layout/SectionCard';
import { Button } from '../ui/button';
import { Input } from '../ui/input';
import { apiFetch } from '../../utils/api';
import { useBrooksLive } from '../../hooks/useBrooksLive';
import { AnalystSwitch } from './AnalystSwitch';
import { ChartPanel } from './ChartPanel';
import { DecisionLog } from './DecisionLog';
import { PnLCard } from './PnLCard';

interface BrooksSessionSummary {
  session_id: string;
  analyst: string;
  config: Record<string, unknown>;
  started_at: string;
}

export default function BrooksLive() {
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [symbol, setSymbol] = useState('BTC/USDT');
  const [interval, setInterval] = useState('5m');
  const [analyst, setAnalyst] = useState('rule');
  const [starting, setStarting] = useState(false);
  const [activeSessions, setActiveSessions] = useState<BrooksSessionSummary[]>([]);

  const { state, switchAnalyst, stopSession } = useBrooksLive(sessionId);

  const refreshActive = useCallback(async () => {
    try {
      const resp = await apiFetch('/brooks-live/sessions');
      if (!resp.ok) return;
      const data = await resp.json();
      setActiveSessions(data.sessions || []);
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    refreshActive();
    const timer = window.setInterval(refreshActive, 10_000);
    return () => clearInterval(timer);
  }, [refreshActive]);

  const start = useCallback(async () => {
    setStarting(true);
    try {
      const resp = await apiFetch('/brooks-live/start', {
        method: 'POST',
        body: JSON.stringify({
          symbol,
          interval,
          analyst,
          mode: 'paper',
        }),
      });
      if (!resp.ok) {
        const err = await resp.text();
        throw new Error(err || 'start failed');
      }
      const data = await resp.json();
      setSessionId(data.session_id);
      toast.success(`Session started: ${data.session_id.slice(0, 8)}`);
      refreshActive();
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    } finally {
      setStarting(false);
    }
  }, [symbol, interval, analyst, refreshActive]);

  const stop = useCallback(async () => {
    if (!sessionId) return;
    if (!confirm('紧急停止当前 BrooksLive 会话？未平仓位将被保留。')) return;
    try {
      await stopSession();
      toast.success('停止请求已发送');
    } catch (e) {
      toast.error(e instanceof Error ? e.message : String(e));
    }
  }, [sessionId, stopSession]);

  const handleSwitch = useCallback(
    async (name: string) => {
      if (!sessionId) {
        setAnalyst(name);
        return;
      }
      await switchAnalyst(name);
      setAnalyst(name);
      toast.success(`Analyst switched → ${name}`);
    },
    [sessionId, switchAnalyst],
  );

  return (
    <div className="space-y-5">
      {/* Session launcher */}
      <SectionCard
        title="BrooksLive paper trading"
        description="Phase 4.6 — 实时 BrooksStrategy + 模拟成交。实盘下单已禁用。"
        action={
          sessionId ? (
            <Button variant="danger" size="sm" onClick={stop}>
              <StopCircle size={14} className="mr-1" /> 紧急停止
            </Button>
          ) : (
            <Button size="sm" onClick={start} disabled={starting}>
              <Play size={14} className="mr-1" /> {starting ? '启动中…' : '启动 session'}
            </Button>
          )
        }
      >
        <div className="grid gap-3 sm:grid-cols-3">
          <label className="text-xs text-muted-foreground">
            Symbol
            <Input
              className="mt-1"
              value={symbol}
              onChange={(e) => setSymbol(e.target.value)}
              disabled={!!sessionId}
            />
          </label>
          <label className="text-xs text-muted-foreground">
            Interval
            <Input
              className="mt-1"
              value={interval}
              onChange={(e) => setInterval(e.target.value)}
              disabled={!!sessionId}
            />
          </label>
          <label className="text-xs text-muted-foreground">
            Analyst
            <Input
              className="mt-1"
              value={analyst}
              onChange={(e) => setAnalyst(e.target.value)}
              disabled={!!sessionId}
            />
          </label>
        </div>
        {state.error && (
          <div className="mt-3 flex items-center gap-2 rounded-md border border-red-500/40 bg-red-500/10 px-3 py-2 text-xs text-red-500">
            <AlertCircle size={14} /> {state.error}
          </div>
        )}
        {activeSessions.length > 0 && (
          <div className="mt-3 space-y-1 text-xs text-muted-foreground">
            <div>Active sessions ({activeSessions.length})</div>
            <div className="flex flex-wrap gap-2">
              {activeSessions.map((s) => (
                <button
                  key={s.session_id}
                  className={
                    'rounded-md border border-border/70 px-2 py-1 hover:bg-accent/40 ' +
                    (s.session_id === sessionId ? 'bg-accent/40 text-foreground' : '')
                  }
                  onClick={() => setSessionId(s.session_id)}
                >
                  {s.session_id.slice(0, 8)} · {s.analyst}
                </button>
              ))}
            </div>
          </div>
        )}
      </SectionCard>

      {sessionId ? (
        <>
          <PnLCard
            equityCurve={state.equityCurve}
            positions={state.positions}
            regime={state.lastRegime}
            analyst={state.analyst}
          />
          <div className="grid gap-4 xl:grid-cols-[2fr_1fr]">
            <ChartPanel bars={state.bars} symbol={state.symbol || symbol} />
            <AnalystSwitch current={state.analyst} onSwitch={handleSwitch} />
          </div>
          <DecisionLog signals={state.recentSignals} />
        </>
      ) : (
        <SectionCard title="No active session" description="配置参数并启动以订阅实时信号。">
          <div className="py-10 text-center text-sm text-muted-foreground">
            点击上方「启动 session」开始 paper trading。
          </div>
        </SectionCard>
      )}
    </div>
  );
}
