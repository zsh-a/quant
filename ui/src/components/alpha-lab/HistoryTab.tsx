/**
 * History tab — search runs + checkpoints.
 *
 * Run list shows summary stats at a glance (evaluations, best metrics, duration).
 * Expanded run shows dataset info, timing breakdown, top factors with split metrics.
 */
import React, { useCallback, useState } from 'react'
import {
  BarChart3, ChevronDown, ChevronRight, Clock,
  Loader2, RefreshCw, Trophy,
} from 'lucide-react'
import type { AlphaLabRunDetail, AlphaLabRunSummary } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { fmt, fmtDur, fmtTime } from './shared'
import { alphaApi } from '../../utils/alphaApi'
import { PipelineView } from './PipelineView'
import { StrategyManager } from './StrategyManager'

/* ── Run card (collapsed) ──────────────────────────────────────────── */

function RunSummaryRow({ run, expanded, onToggle }: {
  run: AlphaLabRunSummary; expanded: boolean; onToggle: () => void
}) {
  const ds = run.dataset
  const stats = run.search_stats
  const hasResults = (run.top_results ?? 0) > 0

  return (
    <button type="button" onClick={onToggle}
      className="flex w-full items-center gap-3 p-3 text-left hover:bg-accent/20 transition">
      <div className="shrink-0 text-muted-foreground">
        {expanded ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
      </div>

      {/* Run ID + time */}
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold truncate">{run.run_id}</div>
        <div className="mt-0.5 flex items-center gap-2 text-[11px] text-muted-foreground">
          <span>{fmtTime(run.saved_at)}</span>
          {ds?.symbols && <span>{(ds.symbols as string[]).length} symbols</span>}
          {ds?.interval && <span>{ds.interval as string}</span>}
          {ds?.shape && <span>{(ds.shape as [number, number])[0]} bars</span>}
        </div>
      </div>

      {/* Key metrics */}
      <div className="flex items-center gap-2 shrink-0">
        {run.timing_seconds != null && (
          <span className="flex items-center gap-1 text-[11px] text-muted-foreground">
            <Clock className="size-3" />{fmtDur(run.timing_seconds * 1000)}
          </span>
        )}
        {stats?.total_evaluations != null && (
          <Badge>{stats.total_evaluations} eval</Badge>
        )}
        <Badge variant={hasResults ? 'info' : 'default'}>{run.top_results ?? 0} top</Badge>
        {run.best_fitness != null && run.best_fitness > 0 && (
          <span className="text-[11px] font-mono text-emerald-400">fit {run.best_fitness.toFixed(3)}</span>
        )}
        {run.best_ic != null && run.best_ic !== 0 && (
          <span className="text-[11px] font-mono text-blue-400">IC {run.best_ic.toFixed(4)}</span>
        )}
      </div>
    </button>
  )
}

/* ── Run detail (expanded) ─────────────────────────────────────────── */

function RunDetailView({ detail, onLoadFormula }: {
  detail: AlphaLabRunDetail; onLoadFormula: (f: string) => void
}) {
  const [expandedFactor, setExpandedFactor] = useState<number | null>(null)
  const stats = detail.search_stats
  const timing = detail.timing
  const topResults = detail.top_results ?? []

  return (
    <div className="border-t border-border/40 p-4 space-y-4">
      {/* Stats + dataset row */}
      <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
        {stats?.total_evaluations != null && (
          <div className="rounded-lg border border-border/50 bg-card/60 px-3 py-2 text-center">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">Evaluations</div>
            <div className="mt-0.5 text-sm font-semibold">{stats.total_evaluations}</div>
          </div>
        )}
        {stats?.total_rejected != null && (
          <div className="rounded-lg border border-border/50 bg-card/60 px-3 py-2 text-center">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">Rejected</div>
            <div className="mt-0.5 text-sm font-semibold">{stats.total_rejected}</div>
          </div>
        )}
        {stats?.archive_size != null && (
          <div className="rounded-lg border border-border/50 bg-card/60 px-3 py-2 text-center">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">Archive</div>
            <div className="mt-0.5 text-sm font-semibold">{stats.archive_size}</div>
          </div>
        )}
        {timing?.overall_seconds != null && (
          <div className="rounded-lg border border-border/50 bg-card/60 px-3 py-2 text-center">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">Duration</div>
            <div className="mt-0.5 text-sm font-semibold">{fmtDur(timing.overall_seconds * 1000)}</div>
          </div>
        )}
        {detail.dataset && (
          <div className="rounded-lg border border-border/50 bg-card/60 px-3 py-2 text-center">
            <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">Dataset</div>
            <div className="mt-0.5 text-xs font-mono">
              {detail.dataset.shape?.[0]}x{detail.dataset.shape?.[1]} {detail.dataset.interval}
            </div>
          </div>
        )}
      </div>

      {/* Pipeline (if available) */}
      {detail.pipeline && detail.pipeline.rounds.length > 0 && (
        <details className="group rounded-lg border border-border/40 bg-card/30">
          <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-[11px] font-medium text-muted-foreground hover:text-foreground transition select-none">
            <BarChart3 className="size-3.5" />
            <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
            Pipeline ({detail.pipeline.rounds.length} rounds)
          </summary>
          <div className="border-t border-border/20 p-3">
            <PipelineView rounds={detail.pipeline.rounds} />
          </div>
        </details>
      )}

      {/* Top factors */}
      {topResults.length > 0 ? (
        <div className="space-y-2">
          <div className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
            <Trophy className="size-3.5 text-amber-400" />
            Top {topResults.length} Factors
          </div>
          {topResults.map((item, i) => {
            const m = item.metrics ?? {}
            const split = item.split_metrics
            const isExpanded = expandedFactor === i
            return (
              <div key={item.expr_hash ?? i} className="rounded-lg border border-border/50 bg-card/60 overflow-hidden">
                <div className="flex items-center gap-3 px-3 py-2 cursor-pointer hover:bg-accent/20 transition"
                  onClick={() => setExpandedFactor(isExpanded ? null : i)}>
                  <span className="text-muted-foreground shrink-0">
                    {isExpanded ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
                  </span>
                  <span className={`flex items-center justify-center size-5 rounded-full text-[10px] font-bold shrink-0 ${i < 3 ? 'bg-amber-500/20 text-amber-300' : 'bg-secondary text-muted-foreground'}`}>
                    {i + 1}
                  </span>
                  <div className="min-w-0 flex-1">
                    <div className="truncate font-mono text-[11px]">{item.formula}</div>
                  </div>
                  <div className="flex items-center gap-3 shrink-0 text-[11px] font-mono">
                    <span><span className="text-muted-foreground">fit</span> {fmt('sharpe', item.fitness)}</span>
                    <span><span className="text-muted-foreground">S</span> {fmt('sharpe', m.sharpe)}</span>
                    <span><span className="text-muted-foreground">IC</span> {fmt('rank_ic', m.rank_ic)}</span>
                    <span><span className="text-muted-foreground">T</span> {fmt('avg_turnover', m.avg_turnover)}</span>
                  </div>
                  <Button variant="ghost" size="sm" onClick={e => { e.stopPropagation(); onLoadFormula(item.formula) }}>Load</Button>
                </div>

                {isExpanded && (
                  <div className="border-t border-border/30 px-4 py-3 space-y-3">
                    {/* Full metrics */}
                    <div className="grid gap-2 grid-cols-4 md:grid-cols-8">
                      {(['sharpe', 'rank_ic', 'ic_ir', 'total_return', 'max_drawdown', 'avg_turnover', 'pnl_per_turnover', 'signal_coverage'] as const).map(k =>
                        m[k] != null ? (
                          <div key={k} className="text-center">
                            <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">{k.replace(/_/g, ' ')}</div>
                            <div className="mt-0.5 text-xs font-semibold font-mono">{fmt(k, m[k])}</div>
                          </div>
                        ) : null
                      )}
                    </div>

                    {/* Split metrics */}
                    {split && (split.train || split.valid || split.test) && (
                      <div className="overflow-x-auto">
                        <table className="w-full text-[11px]">
                          <thead>
                            <tr className="border-b border-border/30">
                              <th className="px-2 py-1 text-left text-muted-foreground font-semibold">Split</th>
                              <th className="px-2 py-1 text-right text-muted-foreground font-semibold">Sharpe</th>
                              <th className="px-2 py-1 text-right text-muted-foreground font-semibold">IC</th>
                              <th className="px-2 py-1 text-right text-muted-foreground font-semibold">Return</th>
                              <th className="px-2 py-1 text-right text-muted-foreground font-semibold">Max DD</th>
                              <th className="px-2 py-1 text-right text-muted-foreground font-semibold">Turnover</th>
                            </tr>
                          </thead>
                          <tbody>
                            {(['train', 'valid', 'test'] as const).map(s => {
                              const sm = split[s]
                              if (!sm) return null
                              return (
                                <tr key={s} className="border-b border-border/15">
                                  <td className="px-2 py-1"><Badge variant={s === 'test' ? 'info' : s === 'valid' ? 'success' : 'default'}>{s}</Badge></td>
                                  <td className="px-2 py-1 text-right font-mono">{fmt('sharpe', sm.sharpe)}</td>
                                  <td className="px-2 py-1 text-right font-mono">{fmt('rank_ic', sm.rank_ic)}</td>
                                  <td className="px-2 py-1 text-right font-mono">{fmt('total_return', sm.total_return)}</td>
                                  <td className="px-2 py-1 text-right font-mono">{fmt('max_drawdown', sm.max_drawdown)}</td>
                                  <td className="px-2 py-1 text-right font-mono">{fmt('avg_turnover', sm.avg_turnover)}</td>
                                </tr>
                              )
                            })}
                          </tbody>
                        </table>
                      </div>
                    )}

                    {/* Lineage */}
                    {item.lineage && Object.keys(item.lineage).length > 0 && (
                      <div className="flex flex-wrap gap-1.5">
                        {Object.entries(item.lineage).map(([k, v]) => v != null && (
                          <span key={k} className="rounded bg-secondary/60 px-1.5 py-0.5 text-[10px] text-muted-foreground">
                            {k}: <span className="font-mono">{typeof v === 'number' ? Number(v).toFixed(4) : String(v)}</span>
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <div className="text-xs text-muted-foreground py-2">No top factors recorded in this run.</div>
      )}
    </div>
  )
}

/* ── Main component ────────────────────────────────────────────────── */

interface HistoryTabProps {
  runs: AlphaLabRunSummary[]
  loading: boolean
  onLoadFormula: (f: string) => void
  onRefresh: () => void
  setErr: (e: string | null) => void
}

export const HistoryTab: React.FC<HistoryTabProps> = ({ runs, loading, onLoadFormula, onRefresh, setErr }) => {
  const [subTab, setSubTab] = useState<'runs' | 'checkpoints'>('runs')
  const [expandedRun, setExpandedRun] = useState<string | null>(null)
  const [runDetail, setRunDetail] = useState<AlphaLabRunDetail | null>(null)
  const [loadingRun, setLoadingRun] = useState(false)

  const handleToggleRun = useCallback(async (id: string) => {
    if (expandedRun === id) { setExpandedRun(null); setRunDetail(null); return }
    try {
      setLoadingRun(true); setExpandedRun(id)
      const r = await alphaApi.getRun(id); setRunDetail(r)
    } catch (e) { setErr(e instanceof Error ? e.message : 'Load failed'); setRunDetail(null) }
    finally { setLoadingRun(false) }
  }, [expandedRun, setErr])

  return (
    <div className="space-y-4">
      <div className="flex gap-1 rounded-lg border border-border/60 bg-secondary/20 p-1 w-fit">
        {([['runs', 'Runs'], ['checkpoints', 'Checkpoints']] as const).map(([key, label]) => (
          <button key={key} onClick={() => setSubTab(key)}
            className={`rounded-md px-4 py-1.5 text-xs font-medium transition ${subTab === key ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
            {label}
          </button>
        ))}
      </div>

      {subTab === 'runs' && (
        <SectionCard title="Search Runs" action={
          <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}>
            <RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        }>
          {runs.length ? (
            <div className="space-y-2">
              {runs.map(run => {
                const expanded = expandedRun === run.run_id
                return (
                  <div key={run.run_id} className="rounded-xl border border-border/60 bg-secondary/20 overflow-hidden">
                    <RunSummaryRow run={run} expanded={expanded} onToggle={() => void handleToggleRun(run.run_id)} />
                    {expanded && (
                      loadingRun
                        ? <div className="flex items-center gap-2 py-6 justify-center text-sm text-muted-foreground border-t border-border/30"><Loader2 className="size-4 animate-spin" />Loading...</div>
                        : runDetail
                          ? <RunDetailView detail={runDetail} onLoadFormula={onLoadFormula} />
                          : null
                    )}
                  </div>
                )
              })}
            </div>
          ) : <EmptyState title="No runs yet" description="Run a search to see results here." />}
        </SectionCard>
      )}

      {subTab === 'checkpoints' && (
        <StrategyManager onLoadFormula={onLoadFormula} showOnly="checkpoints" />
      )}
    </div>
  )
}
