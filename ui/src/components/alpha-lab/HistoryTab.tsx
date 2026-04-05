/**
 * History tab — search runs + checkpoints.
 */
import React, { useCallback, useState } from 'react'
import { ChevronDown, ChevronRight, Loader2, RefreshCw } from 'lucide-react'
import type { AlphaLabRunDetail, AlphaLabRunSummary } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { fmt, fmtTime } from './shared'
import { alphaApi } from '../../utils/alphaApi'
import { StrategyManager } from './StrategyManager'

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
        <SectionCard title="Search Runs" action={<Button variant="outline" size="sm" onClick={onRefresh} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>}>
          {runs.length ? (
            <div className="space-y-2">{runs.map(run => {
              const expanded = expandedRun === run.run_id
              return (
                <div key={run.run_id} className="rounded-xl border border-border/70 bg-secondary/30">
                  <button type="button" onClick={() => void handleToggleRun(run.run_id)}
                    className="flex w-full items-center justify-between gap-4 p-3 text-left hover:bg-accent/30 transition">
                    <div className="flex items-center gap-2">
                      {expanded ? <ChevronDown className="size-4 text-muted-foreground" /> : <ChevronRight className="size-4 text-muted-foreground" />}
                      <span className="text-sm font-semibold truncate">{run.run_id}</span>
                    </div>
                    <div className="flex items-center gap-2 shrink-0">
                      <Badge variant="info">{run.top_results ?? 0} top</Badge>
                      <span className="text-xs text-muted-foreground">{fmtTime(run.saved_at)}</span>
                    </div>
                  </button>
                  {expanded && (
                    <div className="border-t border-border/40 p-3">
                      {loadingRun ? <div className="flex items-center gap-2 py-4 justify-center text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />Loading...</div>
                        : runDetail ? (
                          <div className="space-y-2">
                            {runDetail.search_stats && <div className="flex gap-2 flex-wrap">
                              {Object.entries(runDetail.search_stats).map(([k, v]) => <Badge key={k}>{k}: {v}</Badge>)}
                            </div>}
                            {(runDetail.top_results ?? []).slice(0, 8).map(item => (
                              <div key={item.expr_hash ?? item.formula} className="flex items-center justify-between gap-3 rounded-lg border border-border/50 bg-card/70 px-3 py-2">
                                <div className="min-w-0 flex-1">
                                  <div className="truncate font-mono text-[11px]">{item.formula}</div>
                                  <div className="mt-0.5 text-xs text-muted-foreground">fit {fmt('sharpe', item.fitness)} &middot; sharpe {fmt('sharpe', item.metrics?.sharpe)} &middot; IC {fmt('rank_ic', item.metrics?.rank_ic)}</div>
                                </div>
                                <Button variant="ghost" size="sm" onClick={() => onLoadFormula(item.formula)}>Load</Button>
                              </div>
                            ))}
                          </div>
                        ) : null}
                    </div>
                  )}
                </div>
              )
            })}</div>
          ) : <EmptyState title="No runs yet" description="Run a search to see results here." />}
        </SectionCard>
      )}

      {subTab === 'checkpoints' && (
        <StrategyManager onLoadFormula={onLoadFormula} showOnly="checkpoints" />
      )}
    </div>
  )
}
