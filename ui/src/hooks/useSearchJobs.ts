/**
 * Multi-job search manager hook.
 *
 * - On mount: fetches GET /search-jobs to recover running/completed jobs
 *   (survives page refresh as long as the backend is alive).
 * - Tracks multiple concurrent jobs in a Map<job_id, AlphaLabSearchJob>.
 * - Polls active (pending/running) jobs every POLL_MS.
 * - Exposes submit(), cancel(), hide(), retry(), and the ordered job list.
 * - Persists submit params for failed jobs to localStorage (24h TTL) so
 *   users can retry without re-entering the form after a reload.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import type { AlphaLabSearchJob } from '../types'
import { alphaApi } from '../utils/alphaApi'

const POLL_MS = 3_000
const FAILED_PARAMS_KEY = 'alphaLab.failedJobParams.v1'
const FAILED_PARAMS_TTL_MS = 24 * 60 * 60 * 1000

type FailedParamStore = Record<string, { params: Record<string, unknown>; savedAt: number }>

function isActive(j: AlphaLabSearchJob) {
  return j.status === 'pending' || j.status === 'running' || j.status === 'cancelling'
}

function readFailedParams(): FailedParamStore {
  try {
    const raw = localStorage.getItem(FAILED_PARAMS_KEY)
    if (!raw) return {}
    const data = JSON.parse(raw) as FailedParamStore
    const now = Date.now()
    const fresh: FailedParamStore = {}
    for (const [k, v] of Object.entries(data)) {
      if (v && typeof v.savedAt === 'number' && now - v.savedAt < FAILED_PARAMS_TTL_MS) {
        fresh[k] = v
      }
    }
    return fresh
  } catch {
    return {}
  }
}

function writeFailedParams(store: FailedParamStore): void {
  try {
    localStorage.setItem(FAILED_PARAMS_KEY, JSON.stringify(store))
  } catch { /* quota exceeded — ignore */ }
}

export interface SearchJobsState {
  /** All tracked jobs, newest first. */
  jobs: AlphaLabSearchJob[]
  /** Whether any job is currently active. */
  hasActive: boolean
  /** Submit a new search job and start tracking it. Returns job_id. */
  submit: (params: Record<string, unknown>) => Promise<string>
  /** Request backend cancellation of a running job (graceful). */
  cancel: (jobId: string) => Promise<void>
  /** Remove a job from local state — does NOT cancel the backend task. */
  hide: (jobId: string) => void
  /** Re-submit using the stored params from a failed job. Returns new job_id. */
  retry: (jobId: string) => Promise<string | null>
  /** Re-fetch a single job (e.g. after SSE completes). */
  refresh: (jobId: string) => Promise<void>
}

export function useSearchJobs(onJobComplete?: () => void): SearchJobsState {
  const [jobMap, setJobMap] = useState<Map<string, AlphaLabSearchJob>>(new Map())
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const completedRef = useRef<Set<string>>(new Set())
  const paramsRef = useRef<Map<string, Record<string, unknown>>>(new Map())

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
          if (!isActive(updated) && !completedRef.current.has(id)) {
            completedRef.current.add(id)
            // Persist failed params for retry
            if (updated.status === 'failed') {
              const params = paramsRef.current.get(id)
              if (params) {
                const store = readFailedParams()
                store[id] = { params, savedAt: Date.now() }
                writeFailedParams(store)
              }
            }
            onJobComplete?.()
          }
        } catch { /* ignore single-job poll failure */ }
      }
    }

    void poll()
    pollRef.current = setInterval(poll, POLL_MS)
    return () => { if (pollRef.current) clearInterval(pollRef.current) }
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
      ...(r.request_id ? { request_id: r.request_id as string } : {}),
    } as AlphaLabSearchJob
    paramsRef.current.set(r.job_id, params)
    setJobMap(prev => new Map(prev).set(r.job_id, job))
    return r.job_id
  }, [])

  // --- Cancel (graceful backend stop) ---
  const cancel = useCallback(async (jobId: string): Promise<void> => {
    try {
      const resp = await alphaApi.cancelSearchJob(jobId)
      setJobMap(prev => {
        const next = new Map(prev)
        const existing = next.get(jobId)
        if (existing) {
          next.set(jobId, { ...existing, status: (resp.status as AlphaLabSearchJob['status']) ?? 'cancelling' })
        }
        return next
      })
    } catch {
      // Surface failure by bumping status to 'failed' so user sees the Retry button.
      setJobMap(prev => {
        const next = new Map(prev)
        const existing = next.get(jobId)
        if (existing) next.set(jobId, { ...existing, status: 'failed', error: 'cancel_failed' })
        return next
      })
    }
  }, [])

  // --- Hide (local only) ---
  const hide = useCallback((jobId: string) => {
    setJobMap(prev => {
      const next = new Map(prev)
      next.delete(jobId)
      return next
    })
    paramsRef.current.delete(jobId)
    const store = readFailedParams()
    if (store[jobId]) {
      delete store[jobId]
      writeFailedParams(store)
    }
  }, [])

  // --- Retry (re-submit with stored params) ---
  const retry = useCallback(async (jobId: string): Promise<string | null> => {
    let params = paramsRef.current.get(jobId)
    if (!params) {
      const store = readFailedParams()
      params = store[jobId]?.params
    }
    if (!params) return null
    const newId = await submit(params)
    return newId
  }, [submit])

  // --- Refresh single job ---
  const refresh = useCallback(async (jobId: string) => {
    try {
      const updated = await alphaApi.getSearchJob(jobId)
      setJobMap(prev => new Map(prev).set(jobId, updated))
    } catch { /* ignore */ }
  }, [])

  return { jobs, hasActive, submit, cancel, hide, retry, refresh }
}
