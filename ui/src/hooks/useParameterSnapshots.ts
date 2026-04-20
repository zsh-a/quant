/**
 * Cross-tab parameter snapshot store.
 *
 * Persists a named bag of Alpha Lab workspace params (market, symbols,
 * interval, time range, universe, excludeST) to localStorage so users can
 * round-trip between configurations without re-entering every field.
 */
import { useCallback, useEffect, useState } from 'react'

const STORAGE_KEY = 'alphaLab.paramSnapshots.v1'

export interface ParamSnapshotValue {
  market: string
  interval: string
  symbols: string
  universe: string | null
  excludeST: boolean
  startTime: string
  endTime: string
  formula?: string
}

export interface ParamSnapshot {
  name: string
  value: ParamSnapshotValue
  savedAt: number
}

function read(): ParamSnapshot[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return []
    const parsed = JSON.parse(raw) as ParamSnapshot[]
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

function write(list: ParamSnapshot[]): void {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
  } catch { /* quota → ignore */ }
}

export function useParameterSnapshots() {
  const [snapshots, setSnapshots] = useState<ParamSnapshot[]>(() => read())

  useEffect(() => {
    // Sync between tabs
    const onStorage = (e: StorageEvent) => {
      if (e.key === STORAGE_KEY) setSnapshots(read())
    }
    window.addEventListener('storage', onStorage)
    return () => window.removeEventListener('storage', onStorage)
  }, [])

  const save = useCallback((name: string, value: ParamSnapshotValue) => {
    const cleaned = name.trim()
    if (!cleaned) return
    setSnapshots(prev => {
      const filtered = prev.filter(s => s.name !== cleaned)
      const next = [{ name: cleaned, value, savedAt: Date.now() }, ...filtered].slice(0, 20)
      write(next)
      return next
    })
  }, [])

  const remove = useCallback((name: string) => {
    setSnapshots(prev => {
      const next = prev.filter(s => s.name !== name)
      write(next)
      return next
    })
  }, [])

  return { snapshots, save, remove }
}
