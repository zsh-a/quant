/**
 * Shared data scope selector — interval, time range, symbols/universe.
 * Used by ResearchTab and SearchTab.
 */
import React, { useCallback, useMemo } from 'react'
import { ChevronRight, Info } from 'lucide-react'
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

export const DataScopeSection: React.FC<DataScopeProps> = ({
  interval, setInterval, symbols, setSymbols,
  startTime, setStartTime, endTime, setEndTime,
  intervals, symList, market, universe, setUniverse,
  excludeST, setExcludeST, ws,
}) => {
  const isAShare = market === 'a_share'

  // Symbol presets: crypto uses top-level symbol_presets, A-share uses market_presets
  const marketPresets = (ws?.defaults as any)?.market_presets ?? {}
  const symbolPresets: SymbolPreset[] = isAShare
    ? (marketPresets[market]?.symbol_presets ?? [])
    : ((ws?.defaults as any)?.symbol_presets ?? [])

  const activePreset = useMemo(() => {
    if (isAShare) {
      return symbolPresets.find(p => p.universe === universe)?.key ?? 'custom'
    }
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
    <div className="space-y-4">
      {/* Interval + time range */}
      <div className="grid gap-4 md:grid-cols-3">
        <div className="space-y-1.5">
          <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
            title="K 线周期">Interval<Info className="size-3 opacity-40" /></label>
          <select value={interval} onChange={e => setInterval(e.target.value)}
            className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
            {intervals.map(i => <option key={i} value={i}>{i}</option>)}
          </select>
        </div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
            title="起始时间">Start<Info className="size-3 opacity-40" /></label>
          <Input type={isAShare ? 'date' : 'datetime-local'} value={isAShare ? startTime.slice(0, 10) : startTime}
            onChange={e => setStartTime(isAShare ? e.target.value + 'T00:00' : e.target.value)} />
        </div>
        <div className="space-y-1.5">
          <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
            title="结束时间">End<Info className="size-3 opacity-40" /></label>
          <Input type={isAShare ? 'date' : 'datetime-local'} value={isAShare ? endTime.slice(0, 10) : endTime}
            onChange={e => setEndTime(isAShare ? e.target.value + 'T23:59' : e.target.value)} />
        </div>
      </div>

      {/* Symbol / Universe presets */}
      <div className="space-y-2">
        <div className="flex items-center gap-3">
          <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
            title={isAShare ? '成分股' : '交易对'}>{isAShare ? 'Universe' : 'Symbols'}<Info className="size-3 opacity-40" /></label>
          {symbolPresets.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {symbolPresets.map(p => (
                <button key={p.key} onClick={() => handlePresetChange(p.key)}
                  title={p.brief}
                  className={`rounded-full px-2.5 py-1 text-[11px] font-medium transition-colors ${
                    activePreset === p.key
                      ? 'bg-primary text-primary-foreground'
                      : 'bg-secondary/60 text-muted-foreground hover:bg-secondary hover:text-foreground'
                  }`}>
                  {p.label}
                </button>
              ))}
            </div>
          )}
        </div>

        {isAShare ? (
          <div className="flex items-center gap-3">
            {universe && (
              <Badge variant="info" className="whitespace-nowrap text-xs">
                指数 {universe}
              </Badge>
            )}
            <label className="flex items-center gap-1.5 text-xs text-muted-foreground cursor-pointer select-none">
              <input type="checkbox" checked={excludeST} onChange={e => setExcludeST(e.target.checked)}
                className="rounded border-border" />
              排除 ST
            </label>
            {activePreset === 'custom' && (
              <Input value={symbols} onChange={e => setSymbols(e.target.value)}
                placeholder="sh.600519,sh.601318"
                className="flex-1 font-mono text-xs" />
            )}
          </div>
        ) : (
          <>
            <div className="flex items-center gap-2">
              <Input value={symbols} onChange={e => setSymbols(e.target.value)}
                placeholder="BTCUSDT,ETHUSDT,SOLUSDT"
                className="flex-1 font-mono text-xs" />
              <Badge variant="secondary" className="whitespace-nowrap text-[10px]">
                {symList().length} symbols
              </Badge>
            </div>
            {symList().length > 5 && (
              <details className="group">
                <summary className="cursor-pointer text-[10px] text-muted-foreground hover:text-foreground transition select-none">
                  <ChevronRight className="inline size-3 transition-transform group-open:rotate-90" />
                  {' '}Show all {symList().length} symbols
                </summary>
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {symList().map(s => (
                    <span key={s} className="rounded bg-secondary/50 px-1.5 py-0.5 text-[10px] font-mono text-muted-foreground">
                      {s}
                    </span>
                  ))}
                </div>
              </details>
            )}
          </>
        )}
      </div>
    </div>
  )
}
