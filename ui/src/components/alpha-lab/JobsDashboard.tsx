/**
 * Jobs dashboard — compact summary of all in-flight & recent search jobs.
 *
 * Gives the user an at-a-glance view (counts, latest stage, ETA, cancel)
 * without scrolling through every expanded JobCard. Lives above the
 * SearchTab's detail stack.
 */
import React, { useMemo } from 'react'
import { Loader2, StopCircle, X } from 'lucide-react'
import type { AlphaLabSearchJob } from '../../types'
import type { SearchJobsState } from '../../hooks/useSearchJobs'
import { useAlphaEventsChannel } from '../../hooks/useAlphaEventsChannel'
import { Badge } from '../ui/badge'
import { SectionCard } from '../layout/SectionCard'

function statusBadge(status: AlphaLabSearchJob['status']): React.ReactNode {
  switch (status) {
    case 'pending':   return <Badge variant="info">pending</Badge>
    case 'running':   return <Badge variant="info">running</Badge>
    case 'cancelling': return <Badge variant="warning">cancelling</Badge>
    case 'cancelled': return <Badge variant="warning">cancelled</Badge>
    case 'completed': return <Badge variant="success">completed</Badge>
    case 'failed':    return <Badge variant="danger">failed</Badge>
    default:          return <Badge>{status}</Badge>
  }
}

function fmtSec(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return '—'
  if (s < 90) return `${Math.round(s)}s`
  if (s < 5400) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

interface Props {
  searchJobs: SearchJobsState
}

export const JobsDashboard: React.FC<Props> = ({ searchJobs }) => {
  const active = searchJobs.jobs.filter(j =>
    j.status === 'pending' || j.status === 'running' || j.status === 'cancelling'
  )
  const completed = searchJobs.jobs.filter(j => j.status === 'completed').length
  const failed = searchJobs.jobs.filter(j => j.status === 'failed').length
  const cancelled = searchJobs.jobs.filter(j => j.status === 'cancelled').length

  const activeIds = useMemo(() => active.map(j => j.job_id), [active])
  const ticks = useAlphaEventsChannel(activeIds)

  if (searchJobs.jobs.length === 0) return null

  return (
    <SectionCard
      title="Jobs Dashboard"
      description={`${active.length} active · ${completed} completed · ${failed} failed · ${cancelled} cancelled`}
    >
      {active.length === 0 ? (
        <div className="text-xs text-muted-foreground">No active searches right now.</div>
      ) : (
        <div className="space-y-2">
          {active.map(job => {
            const tick = ticks[job.job_id]
            const round = (tick?.latestRound?.round_idx ?? tick?.latestRound?.round ?? null) as number | null
            const stageKind = tick?.latestStage?.kind as string | undefined
            const stageStrategy = tick?.latestStage?.strategy as string | undefined
            const createdAt = job.created_at ? new Date(job.created_at).getTime() : null
            const elapsedSec = createdAt ? (Date.now() - createdAt) / 1000 : null
            return (
              <div
                key={job.job_id}
                className="flex items-center justify-between gap-3 rounded-lg border border-border/40 bg-secondary/20 px-3 py-2"
              >
                <div className="flex min-w-0 items-center gap-2">
                  {(job.status === 'running' || job.status === 'pending' || job.status === 'cancelling') && (
                    <Loader2 className="size-3 animate-spin text-blue-400 shrink-0" />
                  )}
                  <span className="font-mono text-xs shrink-0">{job.job_id}</span>
                  {statusBadge(job.status)}
                  {job.strategy && <span className="text-xs text-muted-foreground truncate">{job.strategy}</span>}
                </div>
                <div className="flex items-center gap-3 text-[11px] text-muted-foreground shrink-0">
                  {round != null && <span>R{round}</span>}
                  {stageKind && (
                    <span>
                      {stageKind}
                      {stageStrategy ? `·${stageStrategy}` : ''}
                    </span>
                  )}
                  <span>elapsed {fmtSec(elapsedSec)}</span>
                  {job.status !== 'cancelling' && (
                    <button
                      onClick={() => void searchJobs.cancel(job.job_id)}
                      className="text-red-400 hover:text-red-300 flex items-center gap-1"
                      title="Cancel this search"
                    >
                      <StopCircle className="size-3.5" />
                    </button>
                  )}
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Recent terminal jobs row */}
      {searchJobs.jobs.some(j => j.status !== 'running' && j.status !== 'pending') && (
        <div className="mt-3 flex flex-wrap gap-2 pt-3 border-t border-border/30">
          {searchJobs.jobs
            .filter(j => j.status !== 'running' && j.status !== 'pending' && j.status !== 'cancelling')
            .slice(0, 10)
            .map(job => (
              <div
                key={`done-${job.job_id}`}
                className="flex items-center gap-1.5 rounded-md border border-border/40 bg-card/60 px-2 py-1 text-[11px]"
              >
                {statusBadge(job.status)}
                <span className="font-mono text-muted-foreground">{job.job_id}</span>
                {job.top_count != null && <span className="text-muted-foreground">·{job.top_count} top</span>}
                <button
                  onClick={() => searchJobs.hide(job.job_id)}
                  className="text-muted-foreground/60 hover:text-foreground transition"
                  title="Hide"
                >
                  <X className="size-3" />
                </button>
              </div>
            ))}
        </div>
      )}
    </SectionCard>
  )
}
