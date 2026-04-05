import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Cpu,
  Eye,
  LibraryBig,
  Loader2,
  RefreshCw,
  Save,
  SearchCode,
  Workflow,
  Zap,
} from 'lucide-react'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'

import type {
  AlphaLabCombineResult,
  AlphaLabEvaluationSummary,
  AlphaLabRunDetail,
  AlphaLabSearchJob,
  AlphaLabSeriesPoint,
  AlphaLabValidationReport,
  AlphaLabWorkspace as WorkspacePayload,
  AlphaLabZooEntry,
} from '../types'
import { API_BASE } from '../utils/api'
import { formatPercent, formatPrice } from '../utils/format'
import { EmptyState } from './layout/EmptyState'
import { MetricCard } from './layout/MetricCard'
import { SectionCard } from './layout/SectionCard'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

/* ── constants ────────────────────────────────────────────────────────── */

const METRIC_KEYS = ['sharpe', 'rank_ic', 'ic_ir', 'total_return', 'max_drawdown', 'avg_turnover', 'signal_coverage', 'pnl_per_turnover'] as const
const IC_DETAIL_KEYS = ['rank_ic_1d', 'rank_ic_5d', 'rank_ic_10d', 'ic_decay', 'ic_std', 'turnover_proxy'] as const
const PCT_METRICS = new Set(['total_return', 'max_drawdown', 'volatility', 'signal_coverage', 'turnover_proxy'])
const LABELS: Record<string, string> = {
  sharpe: 'Sharpe', rank_ic: 'Rank IC', ic_ir: 'IC IR', ic_std: 'IC Std', ic_decay: 'IC Decay',
  pnl_per_turnover: 'PnL/Turnover', total_return: 'Total Return', max_drawdown: 'Max DD',
  avg_turnover: 'Avg Turnover', signal_coverage: 'Coverage', turnover_proxy: 'Turnover Proxy',
  rank_ic_1d: 'IC 1D', rank_ic_5d: 'IC 5D', rank_ic_10d: 'IC 10D', pnl_efficiency_score: 'PnL Eff',
}
const SEARCH_POLL_MS = 3000
const TRACING_POLL_MS = 10_000

/* ── helpers ──────────────────────────────────────────────────────────── */

type Tab = 'workbench' | 'zoo' | 'runs' | 'combine' | 'tracing'

async function api<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, init)
  const body = await r.json().catch(() => null)
  if (!r.ok) throw new Error(body?.detail ?? 'request failed')
  return body as T
}

const pad = (n: number) => String(n).padStart(2, '0')
const dtLocal = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
const toISO = (v: string) => { const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toISOString() }

function fmt(key: string, v?: number | null) {
  const n = Number(v)
  if (!Number.isFinite(n)) return '--'
  if (PCT_METRICS.has(key)) return formatPercent(n, 2)
  return Math.abs(n) >= 10 ? formatPrice(n, 2) : formatPrice(n, 4)
}

function fmtTime(v?: string | null) {
  if (!v) return '--'
  const d = new Date(v)
  return Number.isNaN(d.getTime()) ? v : d.toLocaleString('zh-CN', { hour12: false })
}

function fmtDur(ms: number) {
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

function fmtTokens(n: number) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

/* ── tiny chart wrapper ───────────────────────────────────────────────── */

function MiniChart({ data, label, color = 'hsl(var(--primary))', height = 180, pct }: {
  data: AlphaLabSeriesPoint[]; label: string; color?: string; height?: number; pct?: boolean
}) {
  if (!data?.length) return null
  return (
    <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
      <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">{label}</div>
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
          <XAxis dataKey="i" tick={false} axisLine={false} />
          <YAxis domain={['auto', 'auto']} width={48} tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
            tickFormatter={pct ? (v: number) => `${(v * 100).toFixed(0)}%` : (v: number) => v.toFixed(2)} />
          <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 12, fontSize: 12 }}
            formatter={(v: number) => [pct ? `${(v * 100).toFixed(2)}%` : v.toFixed(4), label]} labelFormatter={(i: number) => `Bar ${i}`} />
          <Area type="monotone" dataKey="v" stroke={color} fill={color} fillOpacity={0.1} strokeWidth={1.5} dot={false} />
          {!pct && <Line type="monotone" dataKey="v" stroke={color} strokeWidth={1.5} dot={false} />}
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

/* ── metric grid ──────────────────────────────────────────────────────── */

function MetricGrid({ metrics, keys }: { metrics: Record<string, number>; keys: readonly string[] }) {
  return (
    <div className="grid gap-3 grid-cols-2 md:grid-cols-4 xl:grid-cols-8">
      {keys.map(k => metrics[k] != null ? (
        <div key={k} className="rounded-xl border border-border/60 bg-secondary/30 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{LABELS[k] ?? k}</div>
          <div className="mt-1 text-base font-semibold text-foreground">{fmt(k, metrics[k])}</div>
        </div>
      ) : null)}
    </div>
  )
}

/* ── tracing types ────────────────────────────────────────────────────── */

interface TracingSummary { llm_calls: number; llm_errors: number; total_tokens: number; total_cost_usd: number; avg_latency_ms: number }
interface TracingSpan { trace_id: string; span_id: string; operation: string; kind: string; status: string; duration_ms: number; attributes?: Record<string, any>; error?: string }

/* ======================================================================== */
/*  Component                                                               */
/* ======================================================================== */

export const AlphaLabWorkspace: React.FC = () => {
  // ── workspace ──
  const [ws, setWs] = useState<WorkspacePayload | null>(null)
  const [tab, setTab] = useState<Tab>('workbench')
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [booted, setBooted] = useState(false)

  // ── workbench ──
  const [formula, setFormula] = useState('')
  const [interval, setInterval] = useState('5m')
  const [symbols, setSymbols] = useState('BTCUSDT,ETHUSDT,SOLUSDT')
  const [startTime, setStartTime] = useState(() => dtLocal(new Date(Date.now() - 7 * 86400_000)))
  const [endTime, setEndTime] = useState(() => dtLocal(new Date()))
  const [validation, setValidation] = useState<AlphaLabValidationReport | null>(null)
  const [analysis, setAnalysis] = useState<AlphaLabEvaluationSummary | null>(null)
  const [validating, setValidating] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [saving, setSaving] = useState(false)

  // ── search ──
  const [searchJob, setSearchJob] = useState<AlphaLabSearchJob | null>(null)
  const [searchSeeds, setSearchSeeds] = useState('')
  const [popSize, setPopSize] = useState(6)
  const [offspring, setOffspring] = useState(3)
  const [gens, setGens] = useState(3)
  const [topK, setTopK] = useState(5)
  const [nSplits, setNSplits] = useState(5)
  const searchPollRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null)

  // ���─ runs ──
  const [expandedRun, setExpandedRun] = useState<string | null>(null)
  const [runDetail, setRunDetail] = useState<AlphaLabRunDetail | null>(null)
  const [loadingRun, setLoadingRun] = useState(false)
  const [zooSort, setZooSort] = useState<'fitness' | 'saved_at'>('fitness')

  // ── combine ──
  const [combining, setCombining] = useState(false)
  const [combineResult, setCombineResult] = useState<AlphaLabCombineResult | null>(null)
  const [combineMethod, setCombineMethod] = useState('ic_weighted')

  // ── tracing ──
  const [tracingSummary, setTracingSummary] = useState<TracingSummary | null>(null)
  const [tracingSpans, setTracingSpans] = useState<TracingSpan[]>([])
  const tracingRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null)

  // ── derived ──
  const intervals = useMemo(() => ws?.defaults.intervals ?? ['5m', '15m', '1h', '4h'], [ws])
  const samples = useMemo(() => ws?.defaults.sample_formulas ?? [], [ws])
  const sortedZoo = useMemo(() => {
    const e = [...(ws?.zoo ?? [])]
    return zooSort === 'fitness'
      ? e.sort((a, b) => (b.fitness ?? -Infinity) - (a.fitness ?? -Infinity))
      : e.sort((a, b) => (b.saved_at ? new Date(b.saved_at).getTime() : 0) - (a.saved_at ? new Date(a.saved_at).getTime() : 0))
  }, [ws?.zoo, zooSort])

  const symList = useCallback(() => symbols.split(',').map(s => s.trim()).filter(Boolean), [symbols])

  // ── API handlers ──

  const loadWorkspace = useCallback(async () => {
    try {
      setLoading(true); setErr(null)
      const data = await api<WorkspacePayload>('/alpha-lab/workspace')
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

  const handleValidate = useCallback(async () => {
    try { setValidating(true); setErr(null)
      const r = await api<AlphaLabValidationReport>('/alpha-lab/validate', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ formula }) })
      setValidation(r)
    } catch (e) { setValidation(null); setErr(e instanceof Error ? e.message : 'Validation failed') }
    finally { setValidating(false) }
  }, [formula])

  const handleAnalyze = useCallback(async () => {
    try { setAnalyzing(true); setErr(null)
      const r = await api<AlphaLabEvaluationSummary>('/alpha-lab/evaluate-db', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ formula, interval, symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime), summary_only: true }),
      })
      setAnalysis(r)
      if (r.normalized_formula) setValidation({ ok: true, normalized_formula: r.normalized_formula, errors: [], warnings: [] })
    } catch (e) { setAnalysis(null); setErr(e instanceof Error ? e.message : 'Analysis failed') }
    finally { setAnalyzing(false) }
  }, [formula, interval, symbols, startTime, endTime, symList])

  const handleSave = useCallback(async () => {
    try { setSaving(true); setErr(null)
      await api('/alpha-lab/zoo', { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ formula, fitness: analysis?.metrics?.sharpe, metrics: analysis?.metrics ?? {}, lineage: {}, source: 'frontend' }) })
      await loadWorkspace(); setTab('zoo')
    } catch (e) { setErr(e instanceof Error ? e.message : 'Save failed') }
    finally { setSaving(false) }
  }, [formula, analysis, loadWorkspace])

  const handleSearch = useCallback(async () => {
    try { setErr(null)
      const seeds = searchSeeds.split('\n').map(s => s.trim()).filter(Boolean)
      if (formula.trim() && !seeds.includes(formula.trim())) seeds.unshift(formula.trim())
      const r = await api<{ job_id: string; status: string }>('/alpha-lab/search-db', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime), interval, seeds,
          population_size: popSize, offspring_count: offspring, generations: gens, top_k: topK, n_splits: nSplits, persist: true }),
      })
      setSearchJob({ job_id: r.job_id, status: 'pending' })
    } catch (e) { setErr(e instanceof Error ? e.message : 'Search submit failed') }
  }, [formula, interval, symbols, startTime, endTime, searchSeeds, popSize, offspring, gens, topK, nSplits, symList])

  // Poll search job status
  useEffect(() => {
    if (!searchJob || searchJob.status === 'completed' || searchJob.status === 'failed') {
      if (searchPollRef.current) { clearInterval(searchPollRef.current); searchPollRef.current = null }
      return
    }
    const poll = async () => {
      try {
        const r = await api<AlphaLabSearchJob>(`/alpha-lab/search-jobs/${searchJob.job_id}`)
        setSearchJob(r)
        if (r.status === 'completed' || r.status === 'failed') {
          if (r.status === 'completed') void loadWorkspace()
        }
      } catch { /* ignore poll errors */ }
    }
    void poll()
    searchPollRef.current = globalThis.setInterval(poll, SEARCH_POLL_MS)
    return () => { if (searchPollRef.current) { clearInterval(searchPollRef.current); searchPollRef.current = null } }
  }, [searchJob?.job_id, searchJob?.status, loadWorkspace])

  const handleLoadFormula = useCallback((f: string) => { setFormula(f); setValidation(null); setAnalysis(null); setTab('workbench') }, [])

  const handleToggleRun = useCallback(async (id: string) => {
    if (expandedRun === id) { setExpandedRun(null); setRunDetail(null); return }
    try { setLoadingRun(true); setExpandedRun(id)
      const r = await api<AlphaLabRunDetail>(`/alpha-lab/runs/${id}`); setRunDetail(r)
    } catch (e) { setErr(e instanceof Error ? e.message : 'Load failed'); setRunDetail(null) }
    finally { setLoadingRun(false) }
  }, [expandedRun])

  const handleCombine = useCallback(async () => {
    try { setCombining(true); setCombineResult(null); setErr(null)
      const r = await api<AlphaLabCombineResult>('/alpha-lab/combine-zoo', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime), interval, method: combineMethod, summary_only: true }),
      })
      setCombineResult(r)
    } catch (e) { setErr(e instanceof Error ? e.message : 'Combine failed') }
    finally { setCombining(false) }
  }, [symbols, startTime, endTime, interval, combineMethod, symList])

  const loadTracing = useCallback(async () => {
    try {
      const [s, sp] = await Promise.all([api<TracingSummary>('/alpha-lab/tracing/summary'), api<{ spans: TracingSpan[] }>('/alpha-lab/tracing/spans')])
      setTracingSummary(s); setTracingSpans(sp.spans ?? [])
    } catch { /* ignore */ }
  }, [])

  useEffect(() => { void loadWorkspace() }, [loadWorkspace])
  useEffect(() => {
    if (tab === 'tracing') { void loadTracing(); tracingRef.current = globalThis.setInterval(loadTracing, TRACING_POLL_MS) }
    return () => { if (tracingRef.current) { clearInterval(tracingRef.current); tracingRef.current = null } }
  }, [tab, loadTracing])

  const isSearchActive = searchJob?.status === 'pending' || searchJob?.status === 'running'
  const engineLabel = ws?.engine ? `${ws.engine.backend}${ws.engine.triton ? ' · Triton' : ''} · ${ws.engine.device}` : 'loading'

  /* ──────────────────────────────────────────────────────────────────── */

  return (
    <div className="space-y-6">
      {/* overview cards */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard label="Engine" value={ws?.operators.length ?? 0}
          hint={engineLabel}
          trend={<span className="inline-flex items-center gap-2"><Cpu className="size-4" />
            {ws?.engine?.triton ? <Badge variant="success">Triton GPU</Badge> : ws?.engine?.backend === 'torch' ? <Badge variant="info">CUDA</Badge> : <Badge>CPU</Badge>}
          </span>} />
        <MetricCard label="Factor Zoo" value={ws?.zoo.length ?? 0}
          hint="Persisted factors from search"
          trend={<span className="inline-flex items-center gap-2"><LibraryBig className="size-4" /></span>} />
        <MetricCard label="Search Runs" value={ws?.runs.length ?? 0}
          hint="Historical GA search results"
          trend={<span className="inline-flex items-center gap-2"><Workflow className="size-4" /></span>} />
        <MetricCard label="LLM Calls" value={tracingSummary?.llm_calls ?? 0}
          hint={tracingSummary ? `${fmtTokens(tracingSummary.total_tokens)} tokens · $${tracingSummary.total_cost_usd.toFixed(4)}` : 'Switch to Tracing tab'}
          trend={<span className="inline-flex items-center gap-2"><Zap className="size-4" />{tracingSummary ? `avg ${fmtDur(tracingSummary.avg_latency_ms)}` : ''}</span>} />
      </div>

      {/* error banner */}
      {err && <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">{err}</div>}

      <Tabs value={tab} onValueChange={v => setTab(v as Tab)}>
        <TabsList>
          <TabsTrigger value="workbench"><Activity className="mr-1.5 size-3.5" />Workbench</TabsTrigger>
          <TabsTrigger value="zoo"><LibraryBig className="mr-1.5 size-3.5" />Zoo</TabsTrigger>
          <TabsTrigger value="runs"><Workflow className="mr-1.5 size-3.5" />Runs</TabsTrigger>
          <TabsTrigger value="combine"><SearchCode className="mr-1.5 size-3.5" />Combine</TabsTrigger>
          <TabsTrigger value="tracing"><Eye className="mr-1.5 size-3.5" />Tracing</TabsTrigger>
        </TabsList>

        {/* ──── TAB: Workbench ──── */}
        <TabsContent value="workbench" className="space-y-6">
          <SectionCard title="Formula" action={<Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>}>
            <textarea value={formula} onChange={e => setFormula(e.target.value)} placeholder="cs_rank(ts_mean(close, 5) - close)"
              className="min-h-28 w-full rounded-xl border border-border bg-input px-4 py-3 text-sm font-mono text-foreground outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30" />
            {samples.length > 0 && <div className="flex flex-wrap gap-2">{samples.map(s => (
              <button key={s} type="button" onClick={() => handleLoadFormula(s)}
                className="rounded-full border border-border/80 bg-secondary/60 px-3 py-1 text-xs font-mono text-muted-foreground transition hover:text-foreground">{s}</button>
            ))}</div>}
            <div className="grid gap-4 md:grid-cols-4">
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Interval</label>
                <select value={interval} onChange={e => setInterval(e.target.value)}
                  className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">{intervals.map(i => <option key={i} value={i}>{i}</option>)}</select>
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Start</label>
                <Input type="datetime-local" value={startTime} onChange={e => setStartTime(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">End</label>
                <Input type="datetime-local" value={endTime} onChange={e => setEndTime(e.target.value)} />
              </div>
              <div className="space-y-1.5">
                <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Symbols</label>
                <Input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="BTCUSDT,ETHUSDT" />
              </div>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="outline" onClick={() => void handleValidate()} disabled={validating || !formula.trim()}>
                {validating ? <Loader2 className="animate-spin" /> : <CheckCircle2 />}Validate</Button>
              <Button onClick={() => void handleAnalyze()} disabled={analyzing || !formula.trim()}>
                {analyzing ? <Loader2 className="animate-spin" /> : <Activity />}Analyze</Button>
              <Button variant="secondary" onClick={() => void handleSave()} disabled={saving || !formula.trim()}>
                {saving ? <Loader2 className="animate-spin" /> : <Save />}Save to Zoo</Button>
            </div>
            {validation && (
              <div className={`rounded-xl border px-4 py-3 text-sm ${validation.ok ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/20 bg-amber-500/10 text-amber-300'}`}>
                <div className="flex items-center gap-2 font-semibold">{validation.ok ? <CheckCircle2 className="size-4" /> : <AlertTriangle className="size-4" />}{validation.ok ? 'Valid' : 'Invalid'}</div>
                {validation.normalized_formula && <p className="mt-1 break-all font-mono text-xs opacity-80">{validation.normalized_formula}</p>}
                {validation.errors?.map(e => <p key={e} className="mt-1 text-xs text-rose-300">{e}</p>)}
              </div>
            )}
          </SectionCard>

          {/* Search panel */}
          <SectionCard title="GA Search" description="Run evolutionary search with LLM-driven breeding. Submits as background job.">
            <div className="grid gap-4 grid-cols-2 md:grid-cols-5">
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Pop Size</label>
                <Input type="number" min={2} max={32} value={popSize} onChange={e => setPopSize(+e.target.value || 6)} /></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Offspring</label>
                <Input type="number" min={1} max={16} value={offspring} onChange={e => setOffspring(+e.target.value || 3)} /></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Generations</label>
                <Input type="number" min={1} max={20} value={gens} onChange={e => setGens(+e.target.value || 3)} /></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Top-K</label>
                <Input type="number" min={1} max={20} value={topK} onChange={e => setTopK(+e.target.value || 5)} /></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">CPCV Folds</label>
                <Input type="number" min={2} max={10} value={nSplits} onChange={e => setNSplits(+e.target.value || 5)} /></div>
            </div>
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Extra Seeds (one per line)</label>
              <textarea value={searchSeeds} onChange={e => setSearchSeeds(e.target.value)} rows={2}
                className="w-full rounded-xl border border-border bg-input px-4 py-2 text-xs font-mono text-foreground outline-none" />
            </div>
            <div className="flex items-center gap-3">
              <Button onClick={() => void handleSearch()} disabled={isSearchActive || !formula.trim()}>
                {isSearchActive ? <Loader2 className="animate-spin" /> : <Zap />}{isSearchActive ? 'Running...' : 'Start Search'}</Button>
            </div>

            {/* Search job status */}
            {searchJob && (
              <div className={`rounded-xl border p-4 space-y-3 ${searchJob.status === 'completed' ? 'border-emerald-500/20 bg-emerald-500/10' : searchJob.status === 'failed' ? 'border-rose-500/20 bg-rose-500/10' : 'border-blue-500/20 bg-blue-500/10'}`}>
                <div className="flex items-center gap-3">
                  {searchJob.status === 'completed' ? <CheckCircle2 className="size-4 text-emerald-400" /> : searchJob.status === 'failed' ? <AlertTriangle className="size-4 text-rose-400" /> : <Loader2 className="size-4 animate-spin text-blue-400" />}
                  <span className="text-sm font-semibold">{searchJob.status === 'completed' ? 'Search completed' : searchJob.status === 'failed' ? 'Search failed' : 'Search running...'}</span>
                  <Badge variant="info">{searchJob.job_id}</Badge>
                  {searchJob.run_id && <Badge>{searchJob.run_id}</Badge>}
                </div>
                {searchJob.error && <p className="text-xs text-rose-300">{searchJob.error}</p>}
                {searchJob.top_results?.slice(0, 5).map((item, i) => (
                  <div key={item.expr_hash ?? i} className="flex items-center justify-between gap-3 rounded-lg bg-card/60 px-3 py-2">
                    <div className="min-w-0">
                      <div className="truncate font-mono text-xs text-foreground">{item.formula}</div>
                      <div className="mt-0.5 text-xs text-muted-foreground">
                        fit {fmt('sharpe', item.fitness)} · sharpe {fmt('sharpe', item.metrics?.sharpe)} · IC {fmt('rank_ic', item.metrics?.rank_ic)}
                      </div>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(item.formula)}>Load</Button>
                  </div>
                ))}
              </div>
            )}
          </SectionCard>

          {/* Analysis results */}
          {analysis ? (
            <SectionCard title="Analysis">
              <div className="space-y-4">
                <MetricGrid metrics={analysis.metrics} keys={METRIC_KEYS} />
                {(analysis.metrics.rank_ic_1d != null || analysis.metrics.ic_decay != null) && (
                  <MetricGrid metrics={analysis.metrics} keys={IC_DETAIL_KEYS} />
                )}
                <div className="flex flex-wrap gap-2">
                  {analysis.backend && <Badge>{analysis.backend}</Badge>}
                  {analysis.device && <Badge>{analysis.device}</Badge>}
                  {analysis.dataset?.shape && <Badge>{analysis.dataset.shape[0]} × {analysis.dataset.shape[1]}</Badge>}
                  {analysis.expr_hash && <Badge variant="info" className="font-mono text-[10px]">{analysis.expr_hash.slice(0, 12)}</Badge>}
                </div>
                <div className="grid gap-4 lg:grid-cols-2">
                  <MiniChart data={analysis.equity_series ?? []} label="Equity Curve" height={200} />
                  <MiniChart data={analysis.drawdown_series ?? []} label="Drawdown" color="#f43f5e" height={200} pct />
                </div>
                <MiniChart data={analysis.turnover_series ?? []} label="Turnover" color="#a78bfa" height={140} />
              </div>
            </SectionCard>
          ) : (
            <EmptyState title="No analysis yet" description="Enter a formula and click Analyze to see results."
              action={<Button onClick={() => void handleAnalyze()} disabled={analyzing || !formula.trim()}><Activity className="size-4" />Analyze</Button>} />
          )}
        </TabsContent>

        {/* ──── TAB: Zoo ──── */}
        <TabsContent value="zoo">
          <SectionCard title="Factor Zoo" action={
            <div className="flex items-center gap-2">
              <select value={zooSort} onChange={e => setZooSort(e.target.value as typeof zooSort)}
                className="h-8 rounded-lg border border-border bg-input px-2 text-xs text-foreground outline-none">
                <option value="fitness">By Fitness</option><option value="saved_at">By Time</option>
              </select>
              <Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>
            </div>
          }>
            {sortedZoo.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="border-b border-border/70">
                    {['Formula', 'Fitness', 'Sharpe', 'Rank IC', 'Turnover', 'Source', 'Saved', ''].map(h =>
                      <th key={h} className={`px-3 py-2.5 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground ${h === 'Formula' ? 'text-left' : h === '' ? 'text-right' : 'text-right'}`}>{h}</th>)}
                  </tr></thead>
                  <tbody>{sortedZoo.map(e => (
                    <tr key={e.expr_hash ?? e.formula} className="border-b border-border/40 hover:bg-accent/30 transition">
                      <td className="max-w-xs px-3 py-2.5"><div className="truncate font-mono text-xs" title={e.formula}>{e.formula}</div></td>
                      <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('sharpe', e.fitness)}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('sharpe', e.metrics?.sharpe)}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('rank_ic', e.metrics?.rank_ic)}</td>
                      <td className="px-3 py-2.5 text-right font-mono text-xs">{fmt('avg_turnover', e.metrics?.avg_turnover)}</td>
                      <td className="px-3 py-2.5">{e.source ? <Badge>{e.source}</Badge> : '--'}</td>
                      <td className="px-3 py-2.5 text-xs text-muted-foreground whitespace-nowrap">{fmtTime(e.saved_at)}</td>
                      <td className="px-3 py-2.5 text-right"><Button variant="ghost" size="sm" onClick={() => handleLoadFormula(e.formula)}>Load</Button></td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
            ) : <EmptyState title="Zoo is empty" description="Run a search to populate the factor zoo." />}
          </SectionCard>
        </TabsContent>

        {/* ──── TAB: Runs ──── */}
        <TabsContent value="runs">
          <SectionCard title="Search Runs" action={<Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loading}><RefreshCw className={`size-4 ${loading ? 'animate-spin' : ''}`} /></Button>}>
            {ws?.runs.length ? (
              <div className="space-y-2">{ws.runs.map(run => {
                const expanded = expandedRun === run.run_id
                return (
                  <div key={run.run_id} className="rounded-xl border border-border/70 bg-secondary/30">
                    <button type="button" onClick={() => void handleToggleRun(run.run_id)}
                      className="flex w-full items-center justify-between gap-4 p-3 text-left hover:bg-accent/30 transition">
                      <div className="flex items-center gap-2">
                        {expanded ? <ChevronDown className="size-4 text-muted-foreground" /> : <ChevronRight className="size-4 text-muted-foreground" />}
                        <span className="text-sm font-semibold truncate">{run.run_id}</span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0">
                        <Badge variant="info">{run.top_results ?? 0} top</Badge>
                        <span className="text-xs text-muted-foreground">{fmtTime(run.saved_at)}</span>
                      </div>
                    </button>
                    {expanded && (
                      <div className="border-t border-border/40 p-3">
                        {loadingRun ? <div className="flex items-center gap-2 py-4 justify-center text-sm text-muted-foreground"><Loader2 className="size-4 animate-spin" />Loading...</div>
                          : runDetail ? (
                            <div className="space-y-2">
                              {runDetail.search_stats && <div className="flex gap-2 flex-wrap">
                                {Object.entries(runDetail.search_stats).map(([k, v]) => <Badge key={k}>{k}: {v}</Badge>)}
                              </div>}
                              {(runDetail.top_results ?? []).slice(0, 8).map(item => (
                                <div key={item.expr_hash ?? item.formula} className="flex items-center justify-between gap-3 rounded-lg border border-border/50 bg-card/70 px-3 py-2">
                                  <div className="min-w-0 flex-1">
                                    <div className="truncate font-mono text-[11px]">{item.formula}</div>
                                    <div className="mt-0.5 text-xs text-muted-foreground">fit {fmt('sharpe', item.fitness)} · sharpe {fmt('sharpe', item.metrics?.sharpe)} · IC {fmt('rank_ic', item.metrics?.rank_ic)}</div>
                                  </div>
                                  <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(item.formula)}>Load</Button>
                                </div>
                              ))}
                            </div>
                          ) : null}
                      </div>
                    )}
                  </div>
                )
              })}</div>
            ) : <EmptyState title="No runs yet" description="Run a GA search to see results here." />}
          </SectionCard>
        </TabsContent>

        {/* ──── TAB: Combine ──── */}
        <TabsContent value="combine" className="space-y-6">
          <SectionCard title="Factor Combination" description="Select diverse factors from zoo and blend into a composite signal.">
            <div className="grid gap-4 md:grid-cols-3">
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Method</label>
                <select value={combineMethod} onChange={e => setCombineMethod(e.target.value)}
                  className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
                  <option value="equal">Equal Weight</option><option value="ic_weighted">IC Weighted</option><option value="ridge">Ridge Regression</option>
                </select></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Interval</label><Input value={interval} disabled /></div>
              <div className="space-y-1.5"><label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Symbols</label><Input value={symbols} disabled /></div>
            </div>
            <Button onClick={() => void handleCombine()} disabled={combining || (ws?.zoo.length ?? 0) === 0}>
              {combining ? <Loader2 className="animate-spin" /> : <Workflow />}{combining ? 'Combining...' : 'Run Combination'}</Button>
          </SectionCard>
          {combineResult && (
            <SectionCard title="Combination Result">
              <div className="space-y-4">
                {combineResult.metrics && <MetricGrid metrics={combineResult.metrics} keys={['sharpe', 'rank_ic', 'total_return', 'max_drawdown', 'avg_turnover']} />}
                {combineResult.combination?.selected_factors && (
                  <div className="space-y-1">
                    <div className="flex gap-2 mb-2"><Badge variant="info">{combineResult.combination.method}</Badge><Badge>{combineResult.combination.factor_count} factors</Badge></div>
                    {combineResult.combination.selected_factors.map(f => (
                      <div key={f.formula} className="flex items-center justify-between rounded-lg border border-border/50 bg-card/70 px-3 py-2">
                        <span className="truncate font-mono text-xs">{f.formula}</span>
                        <span className="shrink-0 text-xs text-muted-foreground">IC {fmt('rank_ic', f.rank_ic)}</span>
                      </div>
                    ))}
                  </div>
                )}
                <MiniChart data={combineResult.equity_series ?? []} label="Combined Equity" height={200} />
              </div>
            </SectionCard>
          )}
        </TabsContent>

        {/* ──── TAB: Tracing ──── */}
        <TabsContent value="tracing" className="space-y-6">
          <div className="grid gap-4 md:grid-cols-5">
            {[
              { icon: Zap, label: 'LLM Calls', value: tracingSummary?.llm_calls ?? '--' },
              { icon: Activity, label: 'Tokens', value: tracingSummary ? fmtTokens(tracingSummary.total_tokens) : '--' },
              { icon: Clock, label: 'Total Cost', value: tracingSummary ? `$${tracingSummary.total_cost_usd.toFixed(4)}` : '--' },
              { icon: Clock, label: 'Avg Latency', value: tracingSummary ? fmtDur(tracingSummary.avg_latency_ms) : '--' },
              { icon: AlertTriangle, label: 'Error Rate', value: tracingSummary && tracingSummary.llm_calls > 0 ? formatPercent(tracingSummary.llm_errors / tracingSummary.llm_calls, 1) : '--' },
            ].map(c => (
              <div key={c.label} className="rounded-xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"><c.icon className="size-3.5" />{c.label}</div>
                <div className="mt-2 text-xl font-semibold text-foreground">{c.value}</div>
              </div>
            ))}
          </div>
          <SectionCard title="Recent Spans" action={<Button variant="outline" size="sm" onClick={() => void loadTracing()}><RefreshCw className="size-4" /></Button>}>
            {tracingSpans.length > 0 ? (
              <div className="overflow-x-auto"><table className="w-full text-sm">
                <thead><tr className="border-b border-border/70">
                  {['Operation', 'Model', 'Tokens', 'Cost', 'Duration', 'Status'].map(h =>
                    <th key={h} className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground text-left">{h}</th>)}
                </tr></thead>
                <tbody>{tracingSpans.map(s => {
                  const model = s.attributes?.model ?? s.attributes?.['llm.model'] ?? '--'
                  const tokens = s.attributes?.total_tokens ?? s.attributes?.['llm.total_tokens']
                  const cost = s.attributes?.cost_usd ?? s.attributes?.['llm.cost_usd']
                  return (
                    <tr key={`${s.trace_id}-${s.span_id}`} className="border-b border-border/40 hover:bg-accent/30">
                      <td className="max-w-xs px-3 py-2"><div className="truncate text-xs font-medium">{s.operation}</div>{s.error && <div className="truncate text-[11px] text-rose-300">{s.error}</div>}</td>
                      <td className="px-3 py-2 font-mono text-xs text-muted-foreground">{String(model)}</td>
                      <td className="px-3 py-2 font-mono text-xs">{tokens != null ? fmtTokens(Number(tokens)) : '--'}</td>
                      <td className="px-3 py-2 font-mono text-xs">{cost != null ? `$${Number(cost).toFixed(4)}` : '--'}</td>
                      <td className="px-3 py-2 font-mono text-xs">{fmtDur(s.duration_ms)}</td>
                      <td className="px-3 py-2"><Badge variant={s.status === 'error' || s.status === 'ERROR' ? 'danger' : 'success'}>{s.status}</Badge></td>
                    </tr>
                  )
                })}</tbody>
              </table></div>
            ) : <EmptyState title="No spans" description="LLM call traces appear here automatically." />}
          </SectionCard>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default AlphaLabWorkspace
