/**
 * Unified strategy state management tab.
 *
 * Three sections:
 *   1. Strategy State — registered strategies, stateful/stateless, stats
 *   2. Factor Catalog — queryable table of all generated factors
 *   3. Checkpoints — list of saved checkpoints with metadata
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Archive,
  Brain,
  Database,
  Filter,
  Loader2,
  RefreshCw,
  Save,
} from 'lucide-react'
import { alphaApi } from '../../utils/alphaApi'
import type {
  CheckpointEntry,
  FactorCatalogEntry,
  FactorCatalogResponse,
  StrategyInfo,
  StrategyStateResponse,
} from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { MetricCard } from '../layout/MetricCard'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { fmt, fmtDur, fmtTime } from './shared'

/* ── Strategy State Section ─────────────────────────────────────────── */

function StrategyStateSection({ data }: { data: StrategyStateResponse | null }) {
  if (!data) return <EmptyState title="Loading..." />

  return (
    <SectionCard title="Strategy State" description="Registered strategies and their current state">
      {/* Strategy cards */}
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {data.strategies.map(s => (
          <div key={s.name} className="rounded-xl border border-border/60 bg-secondary/30 p-4 space-y-2">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Brain className="size-4 text-muted-foreground" />
                <span className="text-sm font-semibold">{s.name}</span>
              </div>
              <Badge variant={s.stateful ? 'info' : 'default'}>
                {s.stateful ? 'stateful' : 'stateless'}
              </Badge>
            </div>
            {s.stats && (
              <div className="grid grid-cols-2 gap-2">
                {Object.entries(s.stats).filter(([k]) => k !== 'strategy').map(([k, v]) => (
                  <div key={k} className="text-xs">
                    <span className="text-muted-foreground">{k}: </span>
                    <span className="font-mono">{typeof v === 'number' ? (Number.isInteger(v) ? v : Number(v).toFixed(4)) : String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Strategy Memory */}
      {data.strategy_memory && (
        <div className="space-y-3 mt-4">
          <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">Strategy Memory (UCB1 Bandit)</div>
          {data.strategy_memory.themes && Object.keys(data.strategy_memory.themes).length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Theme Performance</div>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-border/50">
                      <th className="px-3 py-1.5 text-left">Theme</th>
                      <th className="px-3 py-1.5 text-right">Count</th>
                      <th className="px-3 py-1.5 text-right">Avg Fitness</th>
                      <th className="px-3 py-1.5 text-right">Success Rate</th>
                      <th className="px-3 py-1.5 text-right">Best Fitness</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(data.strategy_memory.themes)
                      .sort(([, a], [, b]) => b.avg_fitness - a.avg_fitness)
                      .slice(0, 10)
                      .map(([theme, stats]) => (
                        <tr key={theme} className="border-b border-border/30 hover:bg-accent/20">
                          <td className="px-3 py-1.5 font-medium">{theme}</td>
                          <td className="px-3 py-1.5 text-right font-mono">{stats.count}</td>
                          <td className="px-3 py-1.5 text-right font-mono">{stats.avg_fitness.toFixed(3)}</td>
                          <td className="px-3 py-1.5 text-right font-mono">{(stats.success_rate * 100).toFixed(0)}%</td>
                          <td className="px-3 py-1.5 text-right font-mono">{stats.best_fitness.toFixed(3)}</td>
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
          {data.strategy_memory.operators && Object.keys(data.strategy_memory.operators).length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Operator Performance</div>
              <div className="flex flex-wrap gap-2">
                {Object.entries(data.strategy_memory.operators)
                  .sort(([, a], [, b]) => b.avg_fitness - a.avg_fitness)
                  .slice(0, 12)
                  .map(([op, stats]) => (
                    <div key={op} className="rounded-lg border border-border/50 bg-card/60 px-2.5 py-1.5">
                      <span className="font-mono text-[11px]">{op}</span>
                      <span className="ml-2 text-[10px] text-muted-foreground">
                        n={stats.count} fit={stats.avg_fitness.toFixed(3)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>
      )}
    </SectionCard>
  )
}

/* ── Factor Catalog Section ─────────────────────────────────────────── */

function FactorCatalogSection({ onLoadFormula }: { onLoadFormula: (f: string) => void }) {
  const [data, setData] = useState<FactorCatalogResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [filterStrategy, setFilterStrategy] = useState<string>('')
  const [filterMinIc, setFilterMinIc] = useState<string>('')
  const [filterEvaluatedOnly, setFilterEvaluatedOnly] = useState(true)

  const loadCatalog = useCallback(async () => {
    try {
      setLoading(true)
      const r = await alphaApi.getFactorCatalog({
        strategy: filterStrategy || undefined,
        min_ic: filterMinIc ? parseFloat(filterMinIc) : undefined,
        evaluated_only: filterEvaluatedOnly,
        limit: 100,
      })
      setData(r)
    } catch { setData(null) }
    finally { setLoading(false) }
  }, [filterStrategy, filterMinIc, filterEvaluatedOnly])

  useEffect(() => { void loadCatalog() }, [loadCatalog])

  const strategies = useMemo(() => {
    if (!data?.stats) return []
    return Object.keys(data.stats)
  }, [data])

  return (
    <SectionCard
      title="Factor Catalog"
      description="All factors generated across strategies"
      action={<Button variant="outline" size="sm" onClick={() => void loadCatalog()} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>}
    >
      {/* Stats overview */}
      {data?.stats && Object.keys(data.stats).length > 0 && (
        <div className="grid gap-3 grid-cols-2 md:grid-cols-4">
          {Object.entries(data.stats).map(([strat, s]) => (
            <div key={strat} className="rounded-xl border border-border/60 bg-secondary/30 px-3 py-2.5">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{strat}</div>
              <div className="mt-1 text-base font-semibold">{s.total_evaluated} <span className="text-xs text-muted-foreground font-normal">/ {s.total_generated}</span></div>
              <div className="mt-0.5 text-[11px] text-muted-foreground">
                best {s.best_fitness.toFixed(3)} &middot; avg IC {s.avg_ic.toFixed(4)}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-wrap items-end gap-3">
        <div className="space-y-1">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Strategy</label>
          <select
            value={filterStrategy}
            onChange={e => setFilterStrategy(e.target.value)}
            className="h-9 rounded-lg border border-border bg-input px-2 text-xs text-foreground outline-none"
          >
            <option value="">All</option>
            {strategies.map(s => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
        <div className="space-y-1">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Min |IC|</label>
          <Input type="number" step="0.01" value={filterMinIc} onChange={e => setFilterMinIc(e.target.value)} placeholder="0.02" className="h-9 w-24" />
        </div>
        <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer">
          <input type="checkbox" checked={filterEvaluatedOnly} onChange={e => setFilterEvaluatedOnly(e.target.checked)} className="rounded" />
          Evaluated only
        </label>
        <Button variant="outline" size="sm" onClick={() => void loadCatalog()} disabled={loading}>
          <Filter className="size-3.5" />Apply
        </Button>
      </div>

      {/* Table */}
      {data && data.entries.length > 0 ? (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/70">
                {['Formula', 'Strategy', 'Round', 'IC', 'Sharpe', 'Turnover', 'Fitness', ''].map(h =>
                  <th key={h} className={`px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground ${h === 'Formula' ? 'text-left' : h === '' ? 'text-right' : 'text-right'}`}>{h}</th>
                )}
              </tr>
            </thead>
            <tbody>
              {data.entries.map(e => (
                <tr key={e.expr_hash} className="border-b border-border/40 hover:bg-accent/30 transition">
                  <td className="max-w-xs px-3 py-2"><div className="truncate font-mono text-xs" title={e.formula}>{e.formula}</div></td>
                  <td className="px-3 py-2"><Badge>{e.strategy}</Badge></td>
                  <td className="px-3 py-2 text-right font-mono text-xs">R{e.round_idx}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs">{fmt('rank_ic', e.rank_ic)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs">{fmt('sharpe', e.sharpe)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs">{fmt('avg_turnover', e.turnover)}</td>
                  <td className="px-3 py-2 text-right font-mono text-xs">{fmt('sharpe', e.fitness)}</td>
                  <td className="px-3 py-2 text-right">
                    <Button variant="ghost" size="sm" onClick={() => onLoadFormula(e.formula)}>Load</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : data ? (
        <EmptyState title="No factors found" description="Run a search to populate the factor catalog." />
      ) : null}
    </SectionCard>
  )
}

/* ── Checkpoints Section ────────────────────────────────────────────── */

function CheckpointsSection() {
  const [checkpoints, setCheckpoints] = useState<CheckpointEntry[]>([])
  const [loading, setLoading] = useState(false)

  const loadCheckpoints = useCallback(async () => {
    try {
      setLoading(true)
      const r = await alphaApi.listCheckpoints()
      setCheckpoints(r.checkpoints)
    } catch { setCheckpoints([]) }
    finally { setLoading(false) }
  }, [])

  useEffect(() => { void loadCheckpoints() }, [loadCheckpoints])

  return (
    <SectionCard
      title="Checkpoints"
      description="Saved search session states for warm-start resume"
      action={<Button variant="outline" size="sm" onClick={() => void loadCheckpoints()} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>}
    >
      {checkpoints.length > 0 ? (
        <div className="space-y-2">
          {checkpoints.map((ckpt, i) => {
            const ts = new Date(ckpt.timestamp * 1000)
            const strategiesList = Array.isArray(ckpt.strategies)
              ? ckpt.strategies.map(s => typeof s === 'string' ? s : s.name)
              : []
            return (
              <div key={`${ckpt.job_id}-${ckpt.round_idx}-${i}`} className="rounded-xl border border-border/60 bg-secondary/30 p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="flex items-center gap-2">
                    <Save className="size-4 text-muted-foreground" />
                    <span className="text-sm font-semibold">R{ckpt.round_idx}</span>
                    <Badge variant="info">{ckpt.job_id}</Badge>
                  </div>
                  <span className="text-xs text-muted-foreground">{fmtTime(ts.toISOString())}</span>
                </div>
                <div className="mt-2 flex flex-wrap gap-2">
                  {strategiesList.map(s => (
                    <Badge key={s}>{s}</Badge>
                  ))}
                  {ckpt.archive_count != null && (
                    <Badge variant="info">{ckpt.archive_count} archive</Badge>
                  )}
                  {ckpt.context_state?.total_evaluations != null && (
                    <span className="text-[11px] text-muted-foreground">
                      {ckpt.context_state.total_evaluations} evals
                    </span>
                  )}
                </div>
                {/* Strategy details for detailed checkpoints */}
                {Array.isArray(ckpt.strategies) && ckpt.strategies.length > 0 && typeof ckpt.strategies[0] === 'object' && (
                  <div className="mt-2 space-y-1">
                    {(ckpt.strategies as Array<{ name: string; format: string; metadata: Record<string, unknown> }>).map(s => (
                      <div key={s.name} className="flex items-center gap-2 text-[11px]">
                        <span className="font-mono text-muted-foreground">{s.name}</span>
                        <Badge>{s.format}</Badge>
                        {s.metadata?.global_step != null && (
                          <span className="text-muted-foreground">step={String(s.metadata.global_step)}</span>
                        )}
                        {s.metadata?.best_ic != null && (
                          <span className="text-muted-foreground">best_ic={Number(s.metadata.best_ic).toFixed(4)}</span>
                        )}
                        {s.metadata?.zoo_size != null && (
                          <span className="text-muted-foreground">zoo={String(s.metadata.zoo_size)}</span>
                        )}
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      ) : (
        <EmptyState title="No checkpoints" description="Checkpoints are created automatically during search at round boundaries." />
      )}
    </SectionCard>
  )
}

/* ── Main Component ─────────────────────────────────────────────────── */

interface StrategyManagerProps {
  onLoadFormula: (f: string) => void
  /** Render only a specific section (for embedding in other tabs) */
  showOnly?: 'strategies' | 'catalog' | 'checkpoints'
}

export const StrategyManager: React.FC<StrategyManagerProps> = ({ onLoadFormula, showOnly }) => {
  const [stateData, setStateData] = useState<StrategyStateResponse | null>(null)
  const [loading, setLoading] = useState(false)

  const needsState = !showOnly || showOnly === 'strategies'

  const loadState = useCallback(async () => {
    if (!needsState) return
    try {
      setLoading(true)
      const r = await alphaApi.getStrategyState()
      setStateData(r)
    } catch { /* ignore */ }
    finally { setLoading(false) }
  }, [needsState])

  useEffect(() => { void loadState() }, [loadState])

  // Render single section if showOnly is set
  if (showOnly === 'catalog') return <FactorCatalogSection onLoadFormula={onLoadFormula} />
  if (showOnly === 'checkpoints') return <CheckpointsSection />
  if (showOnly === 'strategies') {
    return (
      <div className="space-y-6">
        <div className="grid gap-4 md:grid-cols-3">
          <MetricCard label="Strategies" value={stateData?.strategies.length ?? 0}
            hint={stateData ? `${stateData.strategies.filter(s => s.stateful).length} stateful` : 'loading'}
            trend={<span className="inline-flex items-center gap-1"><Brain className="size-4" /></span>} />
          <MetricCard label="Themes Tracked"
            value={stateData?.strategy_memory?.themes ? Object.keys(stateData.strategy_memory.themes).length : 0}
            hint="UCB1 bandit selection"
            trend={<span className="inline-flex items-center gap-1"><Database className="size-4" /></span>} />
          <MetricCard label="Operator Patterns"
            value={stateData?.strategy_memory?.operators ? Object.keys(stateData.strategy_memory.operators).length : 0}
            hint="Tracked operator patterns"
            trend={<span className="inline-flex items-center gap-1"><Archive className="size-4" /></span>} />
        </div>
        <StrategyStateSection data={stateData} />
      </div>
    )
  }

  // Full view (all sections)
  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-3">
        <MetricCard label="Strategies" value={stateData?.strategies.length ?? 0}
          hint={stateData ? `${stateData.strategies.filter(s => s.stateful).length} stateful` : 'loading'}
          trend={<span className="inline-flex items-center gap-1"><Brain className="size-4" /></span>} />
        <MetricCard label="Themes Tracked"
          value={stateData?.strategy_memory?.themes ? Object.keys(stateData.strategy_memory.themes).length : 0}
          hint="UCB1 bandit selection"
          trend={<span className="inline-flex items-center gap-1"><Database className="size-4" /></span>} />
        <MetricCard label="Operator Patterns"
          value={stateData?.strategy_memory?.operators ? Object.keys(stateData.strategy_memory.operators).length : 0}
          hint="Tracked operator patterns"
          trend={<span className="inline-flex items-center gap-1"><Archive className="size-4" /></span>} />
      </div>
      <StrategyStateSection data={stateData} />
      <FactorCatalogSection onLoadFormula={onLoadFormula} />
      <CheckpointsSection />
    </div>
  )
}
