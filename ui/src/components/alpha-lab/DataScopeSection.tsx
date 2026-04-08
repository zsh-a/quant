/**
 * Shared data scope selector — interval, time range, symbols/universe.
 * Compact layout used by ResearchTab and SearchTab.
 */
import React, { useCallback, useMemo } from 'react'
import { ChevronRight } from 'lucide-react'
import type { AlphaLabWorkspace as WorkspacePayload } from '../../types'
import { Badge } from '../ui/badge'
import { Input } from '../ui/input'

interface SymbolPreset {
  key: string
  label: string
  brief: string
  symbols?: string[]
  universe?: string | null
}

export interface DataScopeProps {
  interval: string
  setInterval: (v: string) => void
  symbols: string
  setSymbols: (v: string) => void
  startTime: string
  setStartTime: (v: string) => void
  endTime: string
  setEndTime: (v: string) => void
  intervals: string[]
  symList: () => string[]
  market: string
  universe: string | null
  setUniverse: (v: string | null) => void
  excludeST: boolean
  setExcludeST: (v: boolean) => void
  ws: WorkspacePayload | null
}

const Label: React.FC<{ children: React.ReactNode }> = ({ children }) => (
  <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{children}</label>
)

export const DataScopeSection: React.FC<DataScopeProps> = ({
  interval, setInterval, symbols, setSymbols,
  startTime, setStartTime, endTime, setEndTime,
  intervals, symList, market, universe, setUniverse,
  excludeST, setExcludeST, ws,
}) => {
  const isAShare = market === 'a_share'

  const marketPresets = (ws?.defaults as any)?.market_presets ?? {}
  const symbolPresets: SymbolPreset[] = isAShare
    ? (marketPresets[market]?.symbol_presets ?? [])
    : ((ws?.defaults as any)?.symbol_presets ?? [])

  const activePreset = useMemo(() => {
    if (isAShare) return symbolPresets.find(p => p.universe === universe)?.key ?? 'custom'
    const current = symList().sort().join(',')
    return symbolPresets.find(p => p.symbols && [...p.symbols].sort().join(',') === current)?.key ?? 'custom'
  }, [symbols, universe, symbolPresets, symList, isAShare])

  const handlePresetChange = useCallback((key: string) => {
    const preset = symbolPresets.find(p => p.key === key)
    if (!preset) return
    if (isAShare) {
      setUniverse(preset.universe ?? null)
      if (!preset.universe) setSymbols('')
    } else {
      if (preset.symbols) setSymbols(preset.symbols.join(','))
    }
  }, [symbolPresets, setSymbols, setUniverse, isAShare])

  return (
    <div className="space-y-3">
      {/* Row 1: Interval + time range (stacked in narrow contexts) */}
      <div className="space-y-1">
        <Label>Interval</Label>
        <select value={interval} onChange={e => setInterval(e.target.value)}
          className="h-8 w-full rounded-lg border border-border bg-input px-2 text-xs text-foreground outline-none">
          {intervals.map(i => <option key={i} value={i}>{i}</option>)}
        </select>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="space-y-1">
          <Label>Start</Label>
          <Input className="h-8 text-xs" type={isAShare ? 'date' : 'datetime-local'}
            value={isAShare ? startTime.slice(0, 10) : startTime}
            onChange={e => setStartTime(isAShare ? e.target.value + 'T00:00' : e.target.value)} />
        </div>
        <div className="space-y-1">
          <Label>End</Label>
          <Input className="h-8 text-xs" type={isAShare ? 'date' : 'datetime-local'}
            value={isAShare ? endTime.slice(0, 10) : endTime}
            onChange={e => setEndTime(isAShare ? e.target.value + 'T23:59' : e.target.value)} />
        </div>
      </div>

      {/* Row 2: Presets */}
      {symbolPresets.length > 0 && (
        <div className="space-y-1">
          <Label>{isAShare ? 'Universe' : 'Symbols'}</Label>
          <div className="flex flex-wrap gap-1">
            {symbolPresets.map(p => (
              <button key={p.key} onClick={() => handlePresetChange(p.key)} title={p.brief}
                className={`rounded-md px-2 py-0.5 text-[10px] font-medium transition ${
                  activePreset === p.key
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-secondary/50 text-muted-foreground hover:text-foreground'
                }`}>
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Row 3: Details (A-share options or symbol input) */}
      {isAShare ? (
        <div className="flex items-center gap-2 flex-wrap">
          {universe && <Badge variant="info" className="text-[10px]">指数 {universe}</Badge>}
          <label className="flex items-center gap-1 text-[10px] text-muted-foreground cursor-pointer select-none">
            <input type="checkbox" checked={excludeST} onChange={e => setExcludeST(e.target.checked)}
              className="rounded border-border size-3" />
            排除 ST
          </label>
          {activePreset === 'custom' && (
            <Input value={symbols} onChange={e => setSymbols(e.target.value)}
              placeholder="sh.600519,sh.601318"
              className="flex-1 font-mono text-[10px] h-7" />
          )}
        </div>
      ) : (
        <>
          <div className="flex items-center gap-1.5">
            <Input value={symbols} onChange={e => setSymbols(e.target.value)}
              placeholder="BTCUSDT,ETHUSDT"
              className="flex-1 font-mono text-[10px] h-7" />
            <Badge variant="secondary" className="text-[9px] shrink-0">{symList().length}</Badge>
          </div>
          {symList().length > 8 && (
            <details className="group">
              <summary className="cursor-pointer text-[10px] text-muted-foreground hover:text-foreground transition select-none">
                <ChevronRight className="inline size-3 transition-transform group-open:rotate-90" />
                {' '}{symList().length} symbols
              </summary>
              <div className="mt-1 flex flex-wrap gap-0.5">
                {symList().map(s => (
                  <span key={s} className="rounded bg-secondary/40 px-1 py-px text-[9px] font-mono text-muted-foreground">{s}</span>
                ))}
              </div>
            </details>
          )}
        </>
      )}
    </div>
  )
}
