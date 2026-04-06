/**
 * Multi-job search manager hook.
 *
 * - On mount: fetches GET /search-jobs to recover running/completed jobs
 *   (survives page refresh as long as the backend is alive).
 * - Tracks multiple concurrent jobs in a Map<job_id, AlphaLabSearchJob>.
 * - Polls active (pending/running) jobs every POLL_MS.
 * - Exposes submit(), dismiss(), and the ordered job list.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { AlphaLabSearchJob } from '../types'
import { alphaApi } from '../utils/alphaApi'

const POLL_MS = 3_000

function isActive(j: AlphaLabSearchJob) {
  return j.status === 'pending' || j.status === 'running'
}

export interface SearchJobsState {
  /** All tracked jobs, newest first. */
  jobs: AlphaLabSearchJob[]
  /** Whether any job is currently active. */
  hasActive: boolean
  /** Submit a new search job and start tracking it. */
  submit: (params: Record<string, unknown>) => Promise<string>
  /** Stop tracking a job (does NOT cancel the backend task). */
  dismiss: (jobId: string) => void
  /** Re-fetch a single job (e.g. after SSE completes). */
  refresh: (jobId: string) => Promise<void>
}

export function useSearchJobs(onJobComplete?: () => void): SearchJobsState {
  const [jobMap, setJobMap] = useState<Map<string, AlphaLabSearchJob>>(new Map())
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const completedRef = useRef<Set<string>>(new Set())

  // Derive sorted list
  const jobs = Array.from(jobMap.values()).sort((a, b) =>
    (b.created_at ?? '').localeCompare(a.created_at ?? ''),
  )
  const hasActive = jobs.some(isActive)

  // --- Initial load: recover jobs from backend ---
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const { jobs: remote } = await alphaApi.listSearchJobs()
        if (cancelled) return
        setJobMap(prev => {
          const next = new Map(prev)
          for (const j of remote) {
            // Keep local state if we already have a richer version
            if (!next.has(j.job_id)) next.set(j.job_id, j)
          }
          return next
        })
      } catch { /* backend unreachable — start fresh */ }
    })()
    return () => { cancelled = true }
  }, [])

  // --- Poll active jobs ---
  useEffect(() => {
    if (pollRef.current) clearInterval(pollRef.current)

    const activeIds = jobs.filter(isActive).map(j => j.job_id)
    if (activeIds.length === 0) return

    const poll = async () => {
      for (const id of activeIds) {
        try {
          const updated = await alphaApi.getSearchJob(id)
          setJobMap(prev => {
            const next = new Map(prev)
            next.set(id, updated)
            return next
          })
          // Fire callback on transition to terminal state
          if (!isActive(updated) && !completedRef.current.has(id)) {
            completedRef.current.add(id)
            onJobComplete?.()
          }
        } catch { /* ignore single-job poll failure */ }
      }
    }

    void poll()
    pollRef.current = setInterval(poll, POLL_MS)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
  // Re-run when active job set changes
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobs.filter(isActive).map(j => j.job_id).join(',')])

  // --- Submit ---
  const submit = useCallback(async (params: Record<string, unknown>): Promise<string> => {
    const r = await alphaApi.submitSearch(params)
    const job: AlphaLabSearchJob = {
      job_id: r.job_id,
      status: 'pending',
      created_at: new Date().toISOString(),
      strategy: (params.strategy as string) ?? 'evolution',
    }
    setJobMap(prev => new Map(prev).set(r.job_id, job))
    return r.job_id
  }, [])

  // --- Dismiss ---
  const dismiss = useCallback((jobId: string) => {
    setJobMap(prev => {
      const next = new Map(prev)
      next.delete(jobId)
      return next
    })
  }, [])

  // --- Refresh single job ---
  const refresh = useCallback(async (jobId: string) => {
    try {
      const updated = await alphaApi.getSearchJob(jobId)
      setJobMap(prev => new Map(prev).set(jobId, updated))
    } catch { /* ignore */ }
  }, [])

  return { jobs, hasActive, submit, dismiss, refresh }
}
