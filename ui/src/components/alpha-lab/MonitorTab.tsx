/**
 * Monitor tab — strategy state, LLM tracing, neural training.
 *
 * Consolidates all observability views into one place.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Activity, AlertTriangle, Clock, RefreshCw, Zap } from 'lucide-react'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { MetricCard } from '../layout/MetricCard'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { fmtDur, fmtTokens, TRACING_POLL_MS, API_BASE } from './shared'
import { formatPercent } from '../../utils/format'
import { alphaApi } from '../../utils/alphaApi'
import { StrategyManager } from './StrategyManager'
import type { AlphaLabTrainingSnapshot } from '../../types'

interface TracingSummary { llm_calls: number; llm_errors: number; total_tokens: number; total_cost_usd: number; avg_latency_ms: number }
interface TracingSpan { trace_id: string; span_id: string; operation: string; kind: string; status: string; duration_ms: number; attributes?: Record<string, any>; error?: string }

interface MonitorTabProps {
  onLoadFormula: (f: string) => void
}

export const MonitorTab: React.FC<MonitorTabProps> = ({ onLoadFormula }) => {
  const [subTab, setSubTab] = useState<'strategies' | 'tracing' | 'neural'>('strategies')
  const [tracingSummary, setTracingSummary] = useState<TracingSummary | null>(null)
  const [tracingSpans, setTracingSpans] = useState<TracingSpan[]>([])
  const [trainingHistory, setTrainingHistory] = useState<AlphaLabTrainingSnapshot[] | null>(null)
  const tracingRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null)

  const loadTracing = useCallback(async () => {
    try {
      const [s, sp] = await Promise.all([
        alphaApi.getTracingSummary() as unknown as Promise<TracingSummary>,
        alphaApi.getTracingSpans() as unknown as Promise<{ spans: TracingSpan[] }>,
      ])
      setTracingSummary(s); setTracingSpans(sp.spans ?? [])
    } catch { /* ignore */ }
  }, [])

  const loadTrainingHistory = useCallback(async () => {
    try {
      const r = await alphaApi.getNeuralHistory()
      setTrainingHistory(r.history.length > 0 ? r.history : null)
    } catch { setTrainingHistory(null) }
  }, [])

  // Poll tracing when active
  useEffect(() => {
    if (subTab === 'tracing') {
      void loadTracing()
      tracingRef.current = globalThis.setInterval(loadTracing, TRACING_POLL_MS)
    }
    return () => { if (tracingRef.current) { clearInterval(tracingRef.current); tracingRef.current = null } }
  }, [subTab, loadTracing])

  useEffect(() => {
    if (subTab === 'neural') void loadTrainingHistory()
  }, [subTab, loadTrainingHistory])

  return (
    <div className="space-y-4">
      <div className="flex gap-1 rounded-lg border border-border/60 bg-secondary/20 p-1 w-fit">
        {([['strategies', 'Strategies'], ['tracing', 'LLM Tracing'], ['neural', 'Neural Training']] as const).map(([key, label]) => (
          <button key={key} onClick={() => setSubTab(key)}
            className={`rounded-md px-4 py-1.5 text-xs font-medium transition ${subTab === key ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Strategies */}
      {subTab === 'strategies' && (
        <StrategyManager onLoadFormula={onLoadFormula} showOnly="strategies" />
      )}

      {/* LLM Tracing */}
      {subTab === 'tracing' && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-5">
            {[
              { icon: Zap, label: 'LLM Calls', value: tracingSummary?.llm_calls ?? '--' },
              { icon: Activity, label: 'Tokens', value: tracingSummary ? fmtTokens(tracingSummary.total_tokens) : '--' },
              { icon: Clock, label: 'Total Cost', value: tracingSummary ? `$${tracingSummary.total_cost_usd.toFixed(4)}` : '--' },
              { icon: Clock, label: 'Avg Latency', value: tracingSummary ? fmtDur(tracingSummary.avg_latency_ms) : '--' },
              { icon: AlertTriangle, label: 'Error Rate', value: tracingSummary && tracingSummary.llm_calls > 0 ? formatPercent(tracingSummary.llm_errors / tracingSummary.llm_calls, 1) : '--' },
            ].map(c => (
              <div key={c.label} className="rounded-xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"><c.icon className="size-3.5" />{c.label}</div>
                <div className="mt-2 text-xl font-semibold text-foreground">{c.value}</div>
              </div>
            ))}
          </div>
          <SectionCard title="Recent Spans" action={<Button variant="outline" size="sm" onClick={() => void loadTracing()}><RefreshCw className="size-4" /></Button>}>
            {tracingSpans.length > 0 ? (
              <div className="overflow-x-auto"><table className="w-full text-sm">
                <thead><tr className="border-b border-border/70">
                  {['Operation', 'Model', 'Tokens', 'Cost', 'Duration', 'Status'].map(h =>
                    <th key={h} className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground text-left">{h}</th>)}
                </tr></thead>
                <tbody>{tracingSpans.map(s => {
                  const model = s.attributes?.model ?? s.attributes?.['llm.model'] ?? '--'
                  const tokens = s.attributes?.total_tokens ?? s.attributes?.['llm.total_tokens']
                  const cost = s.attributes?.cost_usd ?? s.attributes?.['llm.cost_usd']
                  return (
                    <tr key={`${s.trace_id}-${s.span_id}`} className="border-b border-border/40 hover:bg-accent/30">
                      <td className="max-w-xs px-3 py-2"><div className="truncate text-xs font-medium">{s.operation}</div>{s.error && <div className="truncate text-[11px] text-rose-300">{s.error}</div>}</td>
                      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{String(model)}</td>
                      <td className="px-3 py-2 font-mono text-xs">{tokens != null ? fmtTokens(Number(tokens)) : '--'}</td>
                      <td className="px-3 py-2 font-mono text-xs">{cost != null ? `$${Number(cost).toFixed(4)}` : '--'}</td>
                      <td className="px-3 py-2 font-mono text-xs">{fmtDur(s.duration_ms)}</td>
                      <td className="px-3 py-2"><Badge variant={s.status === 'error' || s.status === 'ERROR' ? 'danger' : 'success'}>{s.status}</Badge></td>
                    </tr>
                  )
                })}</tbody>
              </table></div>
            ) : <EmptyState title="No spans" description="LLM call traces appear here automatically." />}
          </SectionCard>
        </div>
      )}

      {/* Neural Training */}
      {subTab === 'neural' && (
        <SectionCard title="Neural Training" description="Transformer + REINFORCE training progress"
          action={<Button variant="outline" size="sm" onClick={loadTrainingHistory}><RefreshCw className="size-3.5" />Refresh</Button>}>
          {!trainingHistory ? (
            <EmptyState title="No training history" description="Run a search with Neural strategy to see training progress." />
          ) : (
            <div className="space-y-6">
              <div className="rounded-xl border border-border/50 overflow-hidden">
                <img src={`${API_BASE}/alpha-lab/neural/plot?t=${Date.now()}`} alt="Training curves"
                  className="w-full" onError={e => { (e.target as HTMLImageElement).style.display = 'none' }} />
              </div>
              {trainingHistory.length > 0 && (() => {
                const last = trainingHistory[trainingHistory.length - 1]
                return (
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    <MetricCard label="Steps" value={last.step} />
                    <MetricCard label="Loss" value={last.loss.toFixed(4)} />
                    <MetricCard label="Avg Reward" value={last.avg_reward.toFixed(4)} />
                    <MetricCard label="Valid Ratio" value={`${(last.valid_ratio * 100).toFixed(1)}%`} />
                  </div>
                )
              })()}
              <div className="overflow-x-auto rounded-xl border border-border/50">
                <table className="w-full text-xs">
                  <thead><tr className="border-b border-border/50 text-muted-foreground">
                    <th className="px-3 py-2 text-left">Step</th>
                    <th className="px-3 py-2 text-right">Loss</th>
                    <th className="px-3 py-2 text-right">Avg Reward</th>
                    <th className="px-3 py-2 text-right">Best Reward</th>
                    <th className="px-3 py-2 text-right">Valid %</th>
                    <th className="px-3 py-2 text-right">Unique</th>
                    <th className="px-3 py-2 text-left">Best Formula</th>
                  </tr></thead>
                  <tbody>{trainingHistory.slice(-20).reverse().map(s => (
                    <tr key={s.step} className="border-b border-border/30 hover:bg-card/50">
                      <td className="px-3 py-1.5 font-mono">{s.step}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{s.loss.toFixed(4)}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{s.avg_reward.toFixed(4)}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{s.best_reward.toFixed(4)}</td>
                      <td className="px-3 py-1.5 text-right font-mono">{(s.valid_ratio * 100).toFixed(1)}%</td>
                      <td className="px-3 py-1.5 text-right">{s.unique}</td>
                      <td className="px-3 py-1.5 truncate max-w-[300px] font-mono text-muted-foreground">{s.best_formula}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            </div>
          )}
        </SectionCard>
      )}
    </div>
  )
}
