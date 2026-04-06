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
  const [formula, setFormula] = useState('')
  const [interval, setInterval] = useState('5m')
  const [symbols, setSymbols] = useState('BTCUSDT,ETHUSDT,SOLUSDT')
  const [startTime, setStartTime] = useState(() => dtLocal(new Date(Date.now() - 7 * 86400_000)))
  const [endTime, setEndTime] = useState(() => dtLocal(new Date()))

  const intervals = useMemo(() => ws?.defaults.intervals ?? ['5m', '15m', '1h', '4h'], [ws])
  const samples = useMemo(() => ws?.defaults.sample_formulas ?? [], [ws])
  const symList = useCallback(() => symbols.split(',').map(s => s.trim()).filter(Boolean), [symbols])

  const loadWorkspace = useCallback(async () => {
    try {
      setLoading(true); setErr(null)
      const data = await alphaApi.getWorkspace()
      setWs(data)
      if (!booted) {
        const cm = data.defaults.crypto_market as Record<string, any>
        setInterval(data.defaults.intervals?.includes('5m') ? '5m' : data.defaults.intervals?.[0] ?? '5m')
        setSymbols(Array.isArray(cm?.default_symbols) && cm.default_symbols.length > 0 ? cm.default_symbols.join(',') : 'BTCUSDT,ETHUSDT,SOLUSDT')
        setFormula(data.zoo[0]?.formula ?? data.defaults.sample_formulas?.[0] ?? 'cs_rank(ts_mean(close, 5) - close)')
        setBooted(true)
      }
    } catch (e) { setErr(e instanceof Error ? e.message : 'Failed to load workspace') }
    finally { setLoading(false) }
  }, [booted])

  useEffect(() => { void loadWorkspace() }, [loadWorkspace])

  const handleLoadFormula = useCallback((f: string) => {
    setFormula(f); setTab('research')
  }, [])

  const engineLabel = ws?.engine ? `${ws.engine.backend}${ws.engine.triton ? ' · Triton' : ''} · ${ws.engine.device}` : 'loading'

  return (
    <div className="space-y-6">
      {/* Overview cards */}
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
          <TabsTrigger value="search"><Zap className="mr-1.5 size-3.5" />Search</TabsTrigger>
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
            ws={ws} onLoadFormula={handleLoadFormula}
            onSearchComplete={() => void loadWorkspace()}
            setErr={setErr}
          />
        </TabsContent>

        <TabsContent value="factors">
          <FactorsTab
            ws={ws} interval={interval} symbols={symbols}
            startTime={startTime} endTime={endTime} symList={symList}
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
