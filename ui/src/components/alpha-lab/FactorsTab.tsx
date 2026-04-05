/**
 * Factors tab — Zoo + Factor Catalog + Combine.
 *
 * Consolidates all factor browsing/management into one place.
 */
import React, { useCallback, useMemo, useState } from 'react'
import { LibraryBig, Loader2, RefreshCw, Workflow } from 'lucide-react'
import type { AlphaLabCombineResult, AlphaLabWorkspace as WorkspacePayload, AlphaLabZooEntry } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { MiniChart, MetricGrid } from './MiniChart'
import { fmt, fmtTime, toISO } from './shared'
import { alphaApi } from '../../utils/alphaApi'
import { StrategyManager } from './StrategyManager'

interface FactorsTabProps {
  ws: WorkspacePayload | null
  interval: string
  symbols: string
  startTime: string
  endTime: string
  symList: () => string[]
  loading: boolean
  onLoadFormula: (f: string) => void
  onRefresh: () => void
  setErr: (e: string | null) => void
}

export const FactorsTab: React.FC<FactorsTabProps> = ({
  ws, interval, symbols, startTime, endTime, symList,
  loading, onLoadFormula, onRefresh, setErr,
}) => {
  const [subTab, setSubTab] = useState<'zoo' | 'catalog' | 'combine'>('zoo')
  const [zooSort, setZooSort] = useState<'fitness' | 'saved_at'>('fitness')
  const [combining, setCombining] = useState(false)
  const [combineResult, setCombineResult] = useState<AlphaLabCombineResult | null>(null)
  const [combineMethod, setCombineMethod] = useState('ic_weighted')

  const sortedZoo = useMemo(() => {
    const e = [...(ws?.zoo ?? [])]
    return zooSort === 'fitness'
      ? e.sort((a, b) => (b.fitness ?? -Infinity) - (a.fitness ?? -Infinity))
      : e.sort((a, b) => (b.saved_at ? new Date(b.saved_at).getTime() : 0) - (a.saved_at ? new Date(a.saved_at).getTime() : 0))
  }, [ws?.zoo, zooSort])

  const handleCombine = useCallback(async () => {
    try {
      setCombining(true); setCombineResult(null); setErr(null)
      const r = await alphaApi.combineZoo({
        symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime),
        interval, method: combineMethod, summary_only: true,
      })
      setCombineResult(r)
    } catch (e) { setErr(e instanceof Error ? e.message : 'Combine failed') }
    finally { setCombining(false) }
  }, [symList, startTime, endTime, interval, combineMethod, setErr])

  return (
    <div className="space-y-4">
      {/* Sub-navigation */}
      <div className="flex gap-1 rounded-lg border border-border/60 bg-secondary/20 p-1 w-fit">
        {([['zoo', 'Zoo'], ['catalog', 'Catalog'], ['combine', 'Combine']] as const).map(([key, label]) => (
          <button key={key} onClick={() => setSubTab(key)}
            className={`rounded-md px-4 py-1.5 text-xs font-medium transition ${subTab === key ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'}`}>
            {label}
          </button>
        ))}
      </div>

      {/* Zoo */}
      {subTab === 'zoo' && (
        <SectionCard title="Factor Zoo" action={
          <div className="flex items-center gap-2">
            <select value={zooSort} onChange={e => setZooSort(e.target.value as typeof zooSort)}
              className="h-8 rounded-lg border border-border bg-input px-2 text-xs text-foreground outline-none">
              <option value="fitness">By Fitness</option><option value="saved_at">By Time</option>
            </select>
            <Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>
          </div>
        }>
          {sortedZoo.length > 0 ? (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead><tr className="border-b border-border/70">
                  {['Formula', 'Fitness', 'Sharpe', 'Rank IC', 'Turnover', 'Source', 'Saved', ''].map(h =>
                    <th key={h} className={`px-3 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground ${h === 'Formula' ? 'text-left' : h === '' ? 'text-right' : 'text-right'}`}>{h}</th>)}
                </tr></thead>
                <tbody>{sortedZoo.map(e => (
                  <tr key={e.expr_hash ?? e.formula} className="border-b border-border/40 hover:bg-accent/30 transition">
                    <td className="max-w-xs px-3 py-2.5"><div className="truncate font-mono text-xs" title={e.formula}>{e.formula}</div></td>
                    <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('sharpe', e.fitness)}</td>
                    <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('sharpe', e.metrics?.sharpe)}</td>
                    <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('rank_ic', e.metrics?.rank_ic)}</td>
                    <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('avg_turnover', e.metrics?.avg_turnover)}</td>
                    <td className="px-3 py-2.5">{e.source ? <Badge>{e.source}</Badge> : '--'}</td>
                    <td className="px-3 py-2.5 text-xs text-muted-foreground whitespace-nowrap">{fmtTime(e.saved_at)}</td>
                    <td className="px-3 py-2.5 text-right"><Button variant="ghost" size="sm" onClick={() => onLoadFormula(e.formula)}>Load</Button></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          ) : <EmptyState title="Zoo is empty" description="Run a search to populate the factor zoo." />}
        </SectionCard>
      )}

      {/* Catalog (delegated to StrategyManager's FactorCatalogSection) */}
      {subTab === 'catalog' && (
        <StrategyManager onLoadFormula={onLoadFormula} showOnly="catalog" />
      )}

      {/* Combine */}
      {subTab === 'combine' && (
        <>
          <SectionCard title="Factor Combination" description="Blend diverse factors from zoo into a composite signal.">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Method</label>
                <select value={combineMethod} onChange={e => setCombineMethod(e.target.value)}
                  className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
                  <option value="equal">Equal Weight</option><option value="ic_weighted">IC Weighted</option><option value="ridge">Ridge Regression</option>
                </select></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Interval</label><Input value={interval} disabled /></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Symbols</label><Input value={symbols} disabled /></div>
            </div>
            <Button onClick={() => void handleCombine()} disabled={combining || (ws?.zoo.length ?? 0) === 0}>
              {combining ? <Loader2 className="animate-spin" /> : <Workflow />}{combining ? 'Combining...' : 'Run Combination'}</Button>
          </SectionCard>
          {combineResult && (
            <SectionCard title="Combination Result">
              <div className="space-y-4">
                {combineResult.metrics && <MetricGrid metrics={combineResult.metrics} keys={['sharpe', 'rank_ic', 'total_return', 'max_drawdown', 'avg_turnover']} />}
                {combineResult.combination?.selected_factors && (
                  <div className="space-y-1">
                    <div className="flex gap-2 mb-2"><Badge variant="info">{combineResult.combination.method}</Badge><Badge>{combineResult.combination.factor_count} factors</Badge></div>
                    {combineResult.combination.selected_factors.map(f => (
                      <div key={f.formula} className="flex items-center justify-between rounded-lg border border-border/50 bg-card/70 px-3 py-2">
                        <span className="truncate font-mono text-xs">{f.formula}</span>
                        <span className="shrink-0 text-xs text-muted-foreground">IC {fmt('rank_ic', f.rank_ic)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <MiniChart data={combineResult.equity_series ?? []} label="Combined Equity" height={200} />
              </div>
            </SectionCard>
          )}
        </>
      )}
    </div>
  )
}
