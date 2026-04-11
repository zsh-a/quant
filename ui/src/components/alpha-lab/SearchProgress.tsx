/**
 * Search progress & results component.
 *
 * During search: SSE-connected live pipeline stage updates.
 * After completion: comprehensive results dashboard with:
 *   - Summary stats
 *   - Top factors table with train/valid/test split metrics
 *   - Pipeline overview
 */
import React, { useMemo, useState } from 'react'
import {
  AlertTriangle, BarChart3, CheckCircle2, ChevronDown, ChevronRight,
  Loader2, Trophy,
} from 'lucide-react'
import { useSearchSSE } from '../../hooks/useSearchSSE'
import type { AlphaLabSearchJob, AlphaLabZooEntry } from '../../types'
import { PipelineView } from './PipelineView'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { fmt, fmtDur } from './shared'

/* ── Types for split metrics ───────────────────────────────────────── */

interface SplitMetrics {
  train?: Record<string, number>
  valid?: Record<string, number>
  test?: Record<string, number>
}

type TopResult = AlphaLabZooEntry & { split_metrics?: SplitMetrics }

/* ── Summary stats cards ───────────────────────────────────────────── */

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="rounded-xl border border-border/60 bg-secondary/30 px-3 py-2.5 text-center">
      <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <div className="mt-1 text-lg font-semibold text-foreground">{value}</div>
      {sub && <div className="mt-0.5 text-[10px] text-muted-foreground">{sub}</div>}
    </div>
  )
}

/* ── Expandable factor row ─────────────────────────────────────────── */

const CORE_METRICS = ['sharpe', 'rank_ic', 'ic_ir', 'total_return', 'max_drawdown', 'avg_turnover', 'pnl_per_turnover', 'signal_coverage'] as const

function FactorRow({ item, rank, expanded, onToggle, onLoad }: {
  item: TopResult; rank: number; expanded: boolean; onToggle: () => void; onLoad: () => void
}) {
  const m = item.metrics ?? {}
  const split = item.split_metrics
  return (
    <div className="rounded-xl border border-border/50 bg-card/60 overflow-hidden">
      {/* Main row */}
      <div className="flex items-center gap-3 px-4 py-3 cursor-pointer hover:bg-accent/20 transition" onClick={onToggle}>
        <div className="flex items-center gap-2 shrink-0">
          <button type="button" className="text-muted-foreground">
            {expanded ? <ChevronDown className="size-4" /> : <ChevronRight className="size-4" />}
          </button>
          <span className={`flex items-center justify-center size-6 rounded-full text-[11px] font-bold ${rank <= 3 ? 'bg-amber-500/20 text-amber-300' : 'bg-secondary text-muted-foreground'}`}>
            {rank}
          </span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate font-mono text-xs text-foreground" title={item.formula}>{item.formula}</div>
        </div>
        <div className="flex items-center gap-3 shrink-0 text-xs font-mono">
          <span title="Fitness"><span className="text-muted-foreground mr-1">fit</span>{fmt('sharpe', item.fitness)}</span>
          <span title="Sharpe"><span className="text-muted-foreground mr-1">S</span>{fmt('sharpe', m.sharpe)}</span>
          <span title="Rank IC"><span className="text-muted-foreground mr-1">IC</span>{fmt('rank_ic', m.rank_ic)}</span>
          <span title="Turnover"><span className="text-muted-foreground mr-1">T</span>{fmt('avg_turnover', m.avg_turnover)}</span>
        </div>
        <Button variant="ghost" size="sm" onClick={e => { e.stopPropagation(); onLoad() }}>Load</Button>
      </div>

      {/* Expanded detail */}
      {expanded && (
        <div className="border-t border-border/30 px-4 py-3 space-y-3">
          {/* Full metrics grid */}
          <div className="grid gap-2 grid-cols-4 md:grid-cols-8">
            {CORE_METRICS.map(k => m[k] != null ? (
              <div key={k} className="text-center">
                <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">{k.replace(/_/g, ' ')}</div>
                <div className="mt-0.5 text-sm font-semibold font-mono">{fmt(k, m[k])}</div>
              </div>
            ) : null)}
          </div>

          {/* Train / Valid / Test split comparison */}
          {split && (split.train || split.valid || split.test) && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Cross-Validation Splits</div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border/40">
                      <th className="px-2 py-1.5 text-left text-[10px] font-semibold text-muted-foreground">Split</th>
                      <th className="px-2 py-1.5 text-right text-[10px] font-semibold text-muted-foreground">Sharpe</th>
                      <th className="px-2 py-1.5 text-right text-[10px] font-semibold text-muted-foreground">Rank IC</th>
                      <th className="px-2 py-1.5 text-right text-[10px] font-semibold text-muted-foreground">Return</th>
                      <th className="px-2 py-1.5 text-right text-[10px] font-semibold text-muted-foreground">Max DD</th>
                      <th className="px-2 py-1.5 text-right text-[10px] font-semibold text-muted-foreground">Turnover</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(['train', 'valid', 'test'] as const).map(s => {
                      const sm = split[s]
                      if (!sm) return null
                      return (
                        <tr key={s} className="border-b border-border/20">
                          <td className="px-2 py-1.5">
                            <Badge variant={s === 'test' ? 'info' : s === 'valid' ? 'success' : 'default'}>{s}</Badge>
                          </td>
                          <td className="px-2 py-1.5 text-right font-mono">{fmt('sharpe', sm.sharpe)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{fmt('rank_ic', sm.rank_ic)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{fmt('total_return', sm.total_return)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{fmt('max_drawdown', sm.max_drawdown)}</td>
                          <td className="px-2 py-1.5 text-right font-mono">{fmt('avg_turnover', sm.avg_turnover)}</td>
                        </tr>
                      )
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Lineage */}
          {item.lineage && (
            <div className="flex flex-wrap gap-2 text-[11px]">
              {Object.entries(item.lineage).map(([k, v]) => v != null && (
                <span key={k} className="rounded-md bg-secondary/60 px-2 py-0.5 text-muted-foreground">
                  {k}: <span className="font-mono text-foreground/80">{typeof v === 'number' ? Number(v).toFixed(4) : String(v)}</span>
                </span>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

/* ── Main component ────────────────────────────────────────────────── */

interface SearchProgressProps {
  searchJob: AlphaLabSearchJob
  onLoadFormula: (formula: string) => void
}

export const SearchProgress: React.FC<SearchProgressProps> = ({ searchJob, onLoadFormula }) => {
  const { rounds, currentStage } = useSearchSSE(
    searchJob.status === 'pending' || searchJob.status === 'running' ? searchJob.job_id : null,
  )
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null)

  const isActive = searchJob.status === 'pending' || searchJob.status === 'running'
  const isCompleted = searchJob.status === 'completed'
  const statusColor = isCompleted
    ? 'border-emerald-500/20 bg-emerald-500/10'
    : searchJob.status === 'failed'
    ? 'border-rose-500/20 bg-rose-500/10'
    : 'border-blue-500/20 bg-blue-500/10'

  const stats = searchJob.search_stats ?? {}
  const timing = searchJob.timing as Record<string, any> | undefined
  const topResults = (searchJob.top_results ?? []) as TopResult[]
  const pipeline = searchJob.pipeline
  const pipelineRounds = rounds.length > 0 ? rounds : pipeline?.rounds ?? []

  // Best factor
  const best = useMemo(() => {
    if (!topResults.length) return null
    return topResults.reduce((a, b) => ((a.fitness ?? 0) >= (b.fitness ?? 0) ? a : b))
  }, [topResults])

  return (
    <div className="space-y-4">
      {/* ── Status header ── */}
      <div className={`rounded-xl border p-4 space-y-4 ${statusColor}`}>
        <div className="flex items-center gap-3">
          {isCompleted ? <CheckCircle2 className="size-4 text-emerald-400" />
            : searchJob.status === 'failed' ? <AlertTriangle className="size-4 text-rose-400" />
            : <Loader2 className="size-4 animate-spin text-blue-400" />}
          <span className="text-sm font-semibold">
            {isCompleted ? 'Search completed' : searchJob.status === 'failed' ? 'Search failed' : 'Search running...'}
          </span>
          <Badge variant="info">{searchJob.job_id}</Badge>
          {searchJob.run_id && <Badge>{searchJob.run_id}</Badge>}
        </div>
        {searchJob.error && <p className="text-xs text-rose-300">{searchJob.error}</p>}

        {/* Live progress */}
        {isActive && currentStage && (
          <div className="flex items-center gap-2 text-xs text-blue-300">
            <Loader2 className="size-3 animate-spin" />
            R{currentStage.round} &middot; {currentStage.kind} ({currentStage.strategy})
          </div>
        )}

        {/* Summary stats (when completed) */}
        {isCompleted && (
          <div className="grid gap-3 grid-cols-2 md:grid-cols-5">
            <StatCard label="Evaluated" value={stats.total_evaluations ?? 0} sub={`${stats.total_rejected ?? 0} rejected`} />
            <StatCard label="Archive" value={stats.archive_size ?? topResults.length} />
            <StatCard label="Best Fitness" value={best ? fmt('sharpe', best.fitness) : '--'} />
            <StatCard label="Best IC" value={best ? fmt('rank_ic', best.metrics?.rank_ic) : '--'} />
            <StatCard label="Duration" value={timing?.overall_seconds ? fmtDur(timing.overall_seconds * 1000) : '--'} />
          </div>
        )}
      </div>

      {/* ── Pipeline visualization ── */}
      {pipelineRounds.length > 0 && (
        <details className="group rounded-xl border border-border/50 bg-card/40">
          <summary className="flex cursor-pointer items-center gap-2 px-4 py-3 text-xs font-medium text-muted-foreground hover:text-foreground transition select-none">
            <BarChart3 className="size-3.5" />
            <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
            Pipeline {isActive ? 'Progress' : 'Summary'}
            <span className="ml-2 text-[10px] opacity-70">{pipelineRounds.length} rounds</span>
          </summary>
          <div className="border-t border-border/30 p-4">
            <PipelineView rounds={pipelineRounds} compact={isActive} />
          </div>
        </details>
      )}

      {/* ── Top results (when completed) ── */}
      {isCompleted && topResults.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Trophy className="size-4 text-amber-400" />
            <span className="text-sm font-semibold">Top {topResults.length} Factors</span>
          </div>
          {topResults.map((item, i) => (
            <FactorRow
              key={item.expr_hash ?? i}
              item={item}
              rank={i + 1}
              expanded={expandedIdx === i}
              onToggle={() => setExpandedIdx(expandedIdx === i ? null : i)}
              onLoad={() => onLoadFormula(item.formula)}
            />
          ))}
        </div>
      )}

      {/* During search: show top results as they appear */}
      {isActive && topResults.length > 0 && (
        <div className="space-y-2">
          {topResults.slice(0, 3).map((item, i) => (
            <div key={item.expr_hash ?? i} className="flex items-center justify-between gap-3 rounded-lg bg-card/60 px-3 py-2">
              <div className="min-w-0">
                <div className="truncate font-mono text-xs text-foreground">{item.formula}</div>
                <div className="mt-0.5 text-xs text-muted-foreground">
                  fit {fmt('sharpe', item.fitness)} &middot; sharpe {fmt('sharpe', item.metrics?.sharpe)} &middot; IC {fmt('rank_ic', item.metrics?.rank_ic)}
                </div>
              </div>
              <Button variant="ghost" size="sm" onClick={() => onLoadFormula(item.formula)}>Load</Button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
