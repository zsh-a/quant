/**
 * Alpha Lab Workspace — thin shell with 5 tabs organized by user intent.
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, Cpu, LibraryBig, Workflow, Zap } from 'lucide-react'

import type { AlphaLabWorkspace as WorkspacePayload } from '../types'
import { useSearchJobs } from '../hooks/useSearchJobs'
import { Badge } from './ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { ResearchTab } from './alpha-lab/ResearchTab'
import { SearchTab } from './alpha-lab/SearchTab'
import { FactorsTab } from './alpha-lab/FactorsTab'
import { HistoryTab } from './alpha-lab/HistoryTab'
import { MonitorTab } from './alpha-lab/MonitorTab'
import { dtDate, DEFAULT_LOOKBACK_YEARS } from './alpha-lab/shared'
import { alphaApi } from '../utils/alphaApi'

type Tab = 'research' | 'search' | 'factors' | 'history' | 'monitor'

const MARKET_LABELS: Record<string, string> = { crypto: 'Crypto', a_share: 'A 股' }

interface AlphaLabWorkspaceProps {
  onViewSession?: (sessionId: string) => void
}

export const AlphaLabWorkspace: React.FC<AlphaLabWorkspaceProps> = ({ onViewSession }) => {
  const [ws, setWs] = useState<WorkspacePayload | null>(null)
  const [tab, setTab] = useState<Tab>('research')
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [booted, setBooted] = useState(false)

  // Shared data params
  const [market, setMarket] = useState('crypto')
  const [formula, setFormula] = useState('')
  const [interval, setInterval] = useState('5m')
  const [symbols, setSymbols] = useState('BTCUSDT,ETHUSDT,SOLUSDT')
  const [universe, setUniverse] = useState<string | null>(null)
  const [excludeST, setExcludeST] = useState(false)
  const [startTime, setStartTime] = useState(() => dtDate(new Date(Date.now() - DEFAULT_LOOKBACK_YEARS * 365 * 86400_000)) + 'T00:00')
  const [endTime, setEndTime] = useState(() => dtDate(new Date()) + 'T23:59')

  const availableMarkets: string[] = (ws?.defaults as any)?.available_markets ?? ['crypto']
  const marketPresets = useMemo(() => (ws?.defaults as any)?.market_presets ?? {}, [ws])
  const currentMarketPreset = useMemo(() => marketPresets[market] ?? {}, [marketPresets, market])
  const intervals = useMemo(() => currentMarketPreset.intervals ?? ws?.defaults.intervals ?? ['5m', '15m', '1h', '4h'], [currentMarketPreset, ws])
  const samples = useMemo(() => currentMarketPreset.sample_formulas ?? ws?.defaults.sample_formulas ?? [], [currentMarketPreset, ws])
  const symList = useCallback(() => symbols.split(',').map(s => s.trim()).filter(Boolean), [symbols])

  const handleMarketChange = useCallback((m: string) => {
    setMarket(m)
    const preset = marketPresets[m] ?? {}
    const newIntervals = preset.intervals ?? []
    setInterval(newIntervals[0] ?? '1d')
    setFormula(preset.sample_formulas?.[0] ?? 'cs_rank(ts_mean(close, 5) - close)')
    const lookback = dtDate(new Date(Date.now() - DEFAULT_LOOKBACK_YEARS * 365 * 86400_000)) + 'T00:00'
    if (m === 'a_share') {
      setSymbols(''); setUniverse('000300'); setExcludeST(true)
      setStartTime(lookback)
    } else {
      const cm = ws?.defaults.crypto_market as Record<string, any>
      setSymbols(Array.isArray(cm?.default_symbols) ? cm.default_symbols.join(',') : 'BTCUSDT,ETHUSDT,SOLUSDT')
      setUniverse(null); setExcludeST(false)
      setStartTime(lookback)
    }
  }, [marketPresets, ws])

  const loadWorkspace = useCallback(async () => {
    try {
      setLoading(true); setErr(null)
      const data = await alphaApi.getWorkspace()
      setWs(data)
      if (!booted) {
        const cm = data.defaults.crypto_market as Record<string, any>
        const mp = (data.defaults as any)?.market_presets?.crypto ?? {}
        const defaultIntervals = mp.intervals ?? data.defaults.intervals ?? ['5m']
        setInterval(defaultIntervals.includes('5m') ? '5m' : defaultIntervals[0] ?? '5m')
        setSymbols(Array.isArray(cm?.default_symbols) && cm.default_symbols.length > 0 ? cm.default_symbols.join(',') : 'BTCUSDT,ETHUSDT,SOLUSDT')
        setFormula(data.zoo[0]?.formula ?? data.defaults.sample_formulas?.[0] ?? 'cs_rank(ts_mean(close, 5) - close)')
        setBooted(true)
      }
    } catch (e) { setErr(e instanceof Error ? e.message : 'Failed to load workspace') }
    finally { setLoading(false) }
  }, [booted])

  useEffect(() => { void loadWorkspace() }, [loadWorkspace])

  const searchJobs = useSearchJobs(() => void loadWorkspace())
  const handleLoadFormula = useCallback((f: string) => { setFormula(f); setTab('research') }, [])

  const engineLabel = ws?.engine
    ? `${ws.engine.backend}${ws.engine.triton ? '+Triton' : ''}`
    : '...'
  const activeCount = searchJobs.jobs.filter(j => j.status === 'pending' || j.status === 'running').length

  return (
    <div className="space-y-3">
      {/* Compact header: market selector + stats inline */}
      <div className="flex items-center gap-4 flex-wrap">
        {/* Market toggle */}
        {availableMarkets.length > 1 && (
          <div className="flex gap-1 rounded-lg border border-border/60 bg-secondary/20 p-0.5">
            {availableMarkets.map(m => (
              <button key={m} onClick={() => handleMarketChange(m)}
                className={`rounded-md px-3 py-1 text-xs font-medium transition ${
                  market === m ? 'bg-card text-foreground shadow-sm' : 'text-muted-foreground hover:text-foreground'
                }`}>
                {MARKET_LABELS[m] ?? m}
              </button>
            ))}
          </div>
        )}

        {/* Inline stats */}
        <div className="flex items-center gap-3 text-xs text-muted-foreground">
          <span className="inline-flex items-center gap-1">
            <Cpu className="size-3" />
            {ws?.engine?.triton ? <Badge variant="success" className="text-[9px] px-1 py-0">Triton</Badge>
              : ws?.engine?.backend === 'torch' ? <Badge variant="info" className="text-[9px] px-1 py-0">CUDA</Badge>
              : <Badge className="text-[9px] px-1 py-0">CPU</Badge>}
            <span className="font-mono">{engineLabel}</span>
          </span>
          <span className="inline-flex items-center gap-1">
            <LibraryBig className="size-3" />{ws?.zoo.length ?? 0} factors
          </span>
          <span className="inline-flex items-center gap-1">
            <Workflow className="size-3" />{ws?.runs.length ?? 0} runs
          </span>
        </div>
      </div>

      {err && <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-3 py-2 text-xs text-rose-200">{err}</div>}

      <Tabs value={tab} onValueChange={v => setTab(v as Tab)}>
        <TabsList>
          <TabsTrigger value="research"><Activity className="mr-1 size-3" />Research</TabsTrigger>
          <TabsTrigger value="search">
            <Zap className="mr-1 size-3" />Search
            {activeCount > 0 && <Badge variant="info" className="ml-1 text-[9px] px-1 py-0">{activeCount}</Badge>}
          </TabsTrigger>
          <TabsTrigger value="factors"><LibraryBig className="mr-1 size-3" />Factors</TabsTrigger>
          <TabsTrigger value="history"><Workflow className="mr-1 size-3" />History</TabsTrigger>
          <TabsTrigger value="monitor"><Cpu className="mr-1 size-3" />Monitor</TabsTrigger>
        </TabsList>

        <TabsContent value="research">
          <ResearchTab
            formula={formula} setFormula={setFormula}
            interval={interval} setInterval={setInterval}
            symbols={symbols} setSymbols={setSymbols}
            startTime={startTime} setStartTime={setStartTime}
            endTime={endTime} setEndTime={setEndTime}
            intervals={intervals} samples={samples} symList={symList}
            market={market} universe={universe} setUniverse={setUniverse}
            excludeST={excludeST} setExcludeST={setExcludeST}
            ws={ws}
            onSaved={() => { void loadWorkspace(); setTab('factors') }}
            setErr={setErr}
          />
        </TabsContent>

        <TabsContent value="search">
          <SearchTab
            formula={formula}
            interval={interval} setInterval={setInterval}
            symbols={symbols} setSymbols={setSymbols}
            startTime={startTime} setStartTime={setStartTime}
            endTime={endTime} setEndTime={setEndTime}
            intervals={intervals} symList={symList}
            market={market} universe={universe} setUniverse={setUniverse}
            excludeST={excludeST} setExcludeST={setExcludeST}
            ws={ws} onLoadFormula={handleLoadFormula}
            searchJobs={searchJobs}
            setErr={setErr}
          />
        </TabsContent>

        <TabsContent value="factors">
          <FactorsTab
            ws={ws} interval={interval} symbols={symbols}
            startTime={startTime} endTime={endTime} symList={symList}
            market={market} universe={universe} excludeST={excludeST}
            loading={loading} onLoadFormula={handleLoadFormula}
            onRefresh={() => void loadWorkspace()} setErr={setErr}
            onViewSession={onViewSession}
          />
        </TabsContent>

        <TabsContent value="history">
          <HistoryTab
            runs={ws?.runs ?? []} loading={loading}
            onLoadFormula={handleLoadFormula}
            onRefresh={() => void loadWorkspace()} setErr={setErr}
          />
        </TabsContent>

        <TabsContent value="monitor">
          <MonitorTab onLoadFormula={handleLoadFormula} />
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default AlphaLabWorkspace
