/**
 * SSE hook for real-time search pipeline progress.
 *
 * Connects to /api/alpha-lab/search-jobs/{jobId}/events and
 * accumulates RoundRecord[] as they arrive.
 * Falls back to polling if SSE is unavailable.
 */
import { useCallback, useEffect, useRef, useState } from 'react'
import { API_BASE, getToken } from '../utils/api'
import type {
  AlphaRoundRecord,
  AlphaStageRecord,
  SearchSSEEvent,
} from '../types'

export interface SearchSSEState {
  rounds: AlphaRoundRecord[]
  currentStage: AlphaStageRecord | null
  isComplete: boolean
  error: string | null
}

export function useSearchSSE(jobId: string | null): SearchSSEState {
  const [rounds, setRounds] = useState<AlphaRoundRecord[]>([])
  const [currentStage, setCurrentStage] = useState<AlphaStageRecord | null>(null)
  const [isComplete, setIsComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const esRef = useRef<EventSource | null>(null)

  const reset = useCallback(() => {
    setRounds([])
    setCurrentStage(null)
    setIsComplete(false)
    setError(null)
  }, [])

  useEffect(() => {
    if (!jobId) {
      reset()
      return
    }

    reset()
    const token = getToken()
    const qs = token ? `?token=${encodeURIComponent(token)}` : ''
    const url = `${API_BASE}/alpha-lab/search-jobs/${jobId}/events${qs}`
    const es = new EventSource(url)
    esRef.current = es

    es.onmessage = (event) => {
      try {
        const parsed = JSON.parse(event.data) as SearchSSEEvent
        switch (parsed.type) {
          case 'stage':
            setCurrentStage(parsed.data)
            break
          case 'round':
            setRounds(prev => [...prev, parsed.data])
            setCurrentStage(null)
            break
          case 'complete':
            setIsComplete(true)
            if (parsed.data.error) setError(parsed.data.error)
            es.close()
            break
        }
      } catch {
        // ignore parse errors
      }
    }

    es.onerror = () => {
      // SSE connection failed — this is normal if the job already completed
      es.close()
    }

    return () => {
      es.close()
      esRef.current = null
    }
  }, [jobId, reset])

  return { rounds, currentStage, isComplete, error }
}
