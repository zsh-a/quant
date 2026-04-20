/**
 * Alpha-Lab event channel: one connection per active job, aggregated into
 * a single state tree by job_id.
 *
 * The backend exposes SSE at ``/alpha-lab/search-jobs/{id}/events``; this
 * hook manages N concurrent EventSource instances so the UI layer can
 * treat the whole "fleet of running jobs" as a single observable.
 */
import { useEffect, useRef, useState } from 'react'
import { API_BASE } from '../utils/api'

export interface AlphaJobEvent {
  type: 'stage' | 'round' | 'complete' | 'status'
  data: Record<string, unknown>
}

export interface AlphaJobTick {
  jobId: string
  lastEvent?: AlphaJobEvent
  eventsByType: Record<string, number>
  latestStage?: Record<string, unknown>
  latestRound?: Record<string, unknown>
  terminal: boolean
}

export function useAlphaEventsChannel(jobIds: string[]) {
  const [ticks, setTicks] = useState<Record<string, AlphaJobTick>>({})
  const sourcesRef = useRef<Map<string, EventSource>>(new Map())

  useEffect(() => {
    const ids = new Set(jobIds)

    // Close connections for jobs no longer tracked
    for (const [id, es] of sourcesRef.current.entries()) {
      if (!ids.has(id)) {
        es.close()
        sourcesRef.current.delete(id)
      }
    }

    // Open connections for newly tracked jobs
    for (const id of ids) {
      if (sourcesRef.current.has(id)) continue
      let es: EventSource
      try {
        es = new EventSource(`${API_BASE}/alpha-lab/search-jobs/${id}/events`)
      } catch {
        continue
      }
      sourcesRef.current.set(id, es)

      es.onmessage = (e) => {
        try {
          const ev = JSON.parse(e.data) as AlphaJobEvent
          setTicks(prev => {
            const tick: AlphaJobTick = prev[id] ?? { jobId: id, eventsByType: {}, terminal: false }
            const next: AlphaJobTick = {
              ...tick,
              lastEvent: ev,
              eventsByType: { ...tick.eventsByType, [ev.type]: (tick.eventsByType[ev.type] ?? 0) + 1 },
            }
            if (ev.type === 'stage') next.latestStage = ev.data
            if (ev.type === 'round') next.latestRound = ev.data
            if (ev.type === 'complete') next.terminal = true
            return { ...prev, [id]: next }
          })
          if (ev.type === 'complete') {
            es.close()
            sourcesRef.current.delete(id)
          }
        } catch { /* malformed payload — ignore */ }
      }

      es.onerror = () => {
        es.close()
        sourcesRef.current.delete(id)
      }
    }

    return () => {
      // On unmount: close everything
      for (const es of sourcesRef.current.values()) es.close()
      sourcesRef.current.clear()
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobIds.join(',')])

  return ticks
}
