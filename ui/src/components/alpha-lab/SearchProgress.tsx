/**
 * SSE-connected search progress component.
 *
 * Replaces the polling-only search status block with real-time
 * pipeline stage updates via Server-Sent Events.
 */
import React from 'react'
import { CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react'
import { useSearchSSE } from '../../hooks/useSearchSSE'
import type { AlphaLabSearchJob } from '../../types'
import { PipelineView } from './PipelineView'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { fmt } from './shared'

interface SearchProgressProps {
  searchJob: AlphaLabSearchJob
  onLoadFormula: (formula: string) => void
}

export const SearchProgress: React.FC<SearchProgressProps> = ({ searchJob, onLoadFormula }) => {
  const { rounds, currentStage, isComplete } = useSearchSSE(
    searchJob.status === 'pending' || searchJob.status === 'running' ? searchJob.job_id : null,
  )

  const isActive = searchJob.status === 'pending' || searchJob.status === 'running'
  const statusColor = searchJob.status === 'completed'
    ? 'border-emerald-500/20 bg-emerald-500/10'
    : searchJob.status === 'failed'
    ? 'border-rose-500/20 bg-rose-500/10'
    : 'border-blue-500/20 bg-blue-500/10'

  return (
    <div className={`rounded-xl border p-4 space-y-4 ${statusColor}`}>
      {/* Header */}
      <div className="flex items-center gap-3">
        {searchJob.status === 'completed' ? (
          <CheckCircle2 className="size-4 text-emerald-400" />
        ) : searchJob.status === 'failed' ? (
          <AlertTriangle className="size-4 text-rose-400" />
        ) : (
          <Loader2 className="size-4 animate-spin text-blue-400" />
        )}
        <span className="text-sm font-semibold">
          {searchJob.status === 'completed' ? 'Search completed' : searchJob.status === 'failed' ? 'Search failed' : 'Search running...'}
        </span>
        <Badge variant="info">{searchJob.job_id}</Badge>
        {searchJob.run_id && <Badge>{searchJob.run_id}</Badge>}
      </div>

      {searchJob.error && <p className="text-xs text-rose-300">{searchJob.error}</p>}

      {/* Live progress indicator */}
      {isActive && currentStage && (
        <div className="flex items-center gap-2 text-xs text-blue-300">
          <Loader2 className="size-3 animate-spin" />
          <span>
            R{currentStage.round} &middot; {currentStage.kind} ({currentStage.strategy})
          </span>
        </div>
      )}

      {/* Pipeline visualization (from SSE) */}
      {rounds.length > 0 && (
        <div className="rounded-lg border border-border/30 bg-card/40 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Pipeline Progress</div>
          <PipelineView rounds={rounds} compact />
        </div>
      )}

      {/* Pipeline visualization (from completed job) */}
      {searchJob.status === 'completed' && searchJob.pipeline && rounds.length === 0 && (
        <div className="rounded-lg border border-border/30 bg-card/40 p-3">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-2">Pipeline Summary</div>
          <PipelineView rounds={searchJob.pipeline.rounds} />
        </div>
      )}

      {/* Top results */}
      {searchJob.top_results?.slice(0, 5).map((item, i) => (
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
  )
}
