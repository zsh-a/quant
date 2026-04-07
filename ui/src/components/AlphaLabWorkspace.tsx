/**
 * Alpha Lab Workspace — thin shell with 5 tabs organized by user intent.
 *
 * Tabs:
 *   Research  — formula editing, validation, single-formula analysis
 *   Search    — automated factor discovery (config, execution, results)
 *   Factors   — factor zoo, catalog, combination
 *   History   — search runs, checkpoints
 *   Monitor   — strategy state, LLM tracing, neural training
 */
import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { Activity, Cpu, Database, LibraryBig, Workflow, Zap } from 'lucide-react'

import type { AlphaLabWorkspace as WorkspacePayload } from '../types'
import { useSearchJobs } from '../hooks/useSearchJobs'
import { MetricCard } from './layout/MetricCard'
import { Badge } from './ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { ResearchTab } from './alpha-lab/ResearchTab'
import { SearchTab } from './alpha-lab/SearchTab'
import { FactorsTab } from './alpha-lab/FactorsTab'
import { HistoryTab } from './alpha-lab/HistoryTab'
import { MonitorTab } from './alpha-lab/MonitorTab'
import { dtLocal } from './alpha-lab/shared'
import { alphaApi } from '../utils/alphaApi'

type Tab = 'research' | 'search' | 'factors' | 'history' | 'monitor'

export const AlphaLabWorkspace: React.FC = () => {
  const [ws, setWs] = useState<WorkspacePayload | null>(null)
  const [tab, setTab] = useState<Tab>('research')
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [booted, setBooted] = useState(false)

  // Shared data params (used across Research, Search, Factors)
  const [market, setMarket] = useState('crypto')
  const [formula, setFormula] = useState('')
  const [interval, setInterval] = useState('5m')
  const [symbols, setSymbols] = useState('BTCUSDT,ETHUSDT,SOLUSDT')
  const [universe, setUniverse] = useState<string | null>(null)
  const [excludeST, setExcludeST] = useState(false)
  const [startTime, setStartTime] = useState(() => dtLocal(new Date(Date.now() - 7 * 86400_000)))
  const [endTime, setEndTime] = useState(() => dtLocal(new Date()))

  const availableMarkets: string[] = (ws?.defaults as any)?.available_markets ?? ['crypto']
  const marketPresets = useMemo(() => (ws?.defaults as any)?.market_presets ?? {}, [ws])
  const currentMarketPreset = useMemo(() => marketPresets[market] ?? {}, [marketPresets, market])
  const intervals = useMemo(() => currentMarketPreset.intervals ?? ws?.defaults.intervals ?? ['5m', '15m', '1h', '4h'], [currentMarketPreset, ws])
  const samples = useMemo(() => currentMarketPreset.sample_formulas ?? ws?.defaults.sample_formulas ?? [], [currentMarketPreset, ws])
  const symList = useCallback(() => symbols.split(',').map(s => s.trim()).filter(Boolean), [symbols])

  // When market changes, reset interval/symbols/universe to defaults
  const handleMarketChange = useCallback((m: string) => {
    setMarket(m)
    const preset = marketPresets[m] ?? {}
    const newIntervals = preset.intervals ?? []
    setInterval(newIntervals[0] ?? '1d')
    setFormula(preset.sample_formulas?.[0] ?? 'cs_rank(ts_mean(close, 5) - close)')
    if (m === 'a_share') {
      setSymbols('')
      setUniverse('000300')
      setExcludeST(true)
      setStartTime(dtLocal(new Date(Date.now() - 365 * 86400_000)))
    } else {
      const cm = ws?.defaults.crypto_market as Record<string, any>
      setSymbols(Array.isArray(cm?.default_symbols) ? cm.default_symbols.join(',') : 'BTCUSDT,ETHUSDT,SOLUSDT')
      setUniverse(null)
      setExcludeST(false)
      setStartTime(dtLocal(new Date(Date.now() - 7 * 86400_000)))
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

  // Multi-job manager — survives page refresh via backend recovery
  const searchJobs = useSearchJobs(() => void loadWorkspace())

  const handleLoadFormula = useCallback((f: string) => {
    setFormula(f); setTab('research')
  }, [])

  const engineLabel = ws?.engine ? `${ws.engine.backend}${ws.engine.triton ? ' · Triton' : ''} · ${ws.engine.device}` : 'loading'
  const activeCount = searchJobs.jobs.filter(j => j.status === 'pending' || j.status === 'running').length

  return (
    <div className="space-y-6">
      {/* Market selector + overview cards */}
      <div className="flex items-center gap-3 mb-2">
        {availableMarkets.length > 1 && availableMarkets.map(m => (
          <button key={m} onClick={() => handleMarketChange(m)}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition-colors ${
              market === m
                ? 'bg-primary text-primary-foreground shadow-sm'
                : 'bg-secondary/60 text-muted-foreground hover:bg-secondary hover:text-foreground'
            }`}>
            {m === 'crypto' ? 'Crypto' : m === 'a_share' ? 'A 股' : m}
          </button>
        ))}
      </div>
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Engine" value={ws?.operators.length ?? 0} hint={engineLabel}
          trend={<span className="inline-flex items-center gap-2"><Cpu className="size-4" />
            {ws?.engine?.triton ? <Badge variant="success">Triton GPU</Badge> : ws?.engine?.backend === 'torch' ? <Badge variant="info">CUDA</Badge> : <Badge>CPU</Badge>}
          </span>} />
        <MetricCard label="Factor Zoo" value={ws?.zoo.length ?? 0} hint="Persisted factors from search"
          trend={<span className="inline-flex items-center gap-2"><LibraryBig className="size-4" /></span>} />
        <MetricCard label="Search Runs" value={ws?.runs.length ?? 0} hint="Historical search results"
          trend={<span className="inline-flex items-center gap-2"><Workflow className="size-4" /></span>} />
        <MetricCard label="Strategies" value={ws?.strategy_modes?.length ?? 0} hint="Available search strategies"
          trend={<span className="inline-flex items-center gap-2"><Database className="size-4" /></span>} />
      </div>

      {err && <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{err}</div>}

      <Tabs value={tab} onValueChange={v => setTab(v as Tab)}>
        <TabsList>
          <TabsTrigger value="research"><Activity className="mr-1.5 size-3.5" />Research</TabsTrigger>
          <TabsTrigger value="search">
            <Zap className="mr-1.5 size-3.5" />Search
            {activeCount > 0 && <Badge variant="info" className="ml-1.5 text-[10px] px-1.5 py-0">{activeCount}</Badge>}
          </TabsTrigger>
          <TabsTrigger value="factors"><LibraryBig className="mr-1.5 size-3.5" />Factors</TabsTrigger>
          <TabsTrigger value="history"><Workflow className="mr-1.5 size-3.5" />History</TabsTrigger>
          <TabsTrigger value="monitor"><Cpu className="mr-1.5 size-3.5" />Monitor</TabsTrigger>
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
