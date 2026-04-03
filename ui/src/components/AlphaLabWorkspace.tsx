import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Clock,
  Cpu,
  Database,
  Eye,
  LibraryBig,
  RefreshCw,
  Save,
  SearchCode,
  Workflow,
  Zap,
} from 'lucide-react'

import type {
  AlphaLabEvaluationSummary,
  AlphaLabRunDetail,
  AlphaLabRunSummary,
  AlphaLabValidationReport,
  AlphaLabWorkspace as AlphaLabWorkspacePayload,
  AlphaLabZooEntry,
} from '../types'
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
import { API_BASE } from '../utils/api'
import { formatPercent, formatPrice } from '../utils/format'
import { EmptyState } from './layout/EmptyState'
import { MetricCard } from './layout/MetricCard'
import { SectionCard } from './layout/SectionCard'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'

/* -------------------------------------------------------------------------- */
/*  Constants                                                                 */
/* -------------------------------------------------------------------------- */

const PRIMARY_METRICS = [
  'sharpe',
  'rank_ic',
  'pnl_per_turnover',
  'total_return',
  'max_drawdown',
  'avg_turnover',
] as const

const PERCENT_METRICS = new Set([
  'total_return',
  'max_drawdown',
  'volatility',
  'signal_coverage',
  'active_bar_ratio',
])

const METRIC_LABELS: Record<string, string> = {
  sharpe: 'Sharpe',
  rank_ic: 'Rank IC',
  pnl_per_turnover: 'PnL / Turnover',
  total_return: '总收益',
  max_drawdown: '最大回撤',
  avg_turnover: '平均换手',
  stability: '稳定性',
  tail_penalty_adjusted_return: '回撤调整收益',
  signal_coverage: '信号覆盖率',
  active_bar_ratio: '活跃 K 线占比',
}

const TRACING_REFRESH_INTERVAL = 10_000

/* -------------------------------------------------------------------------- */
/*  Tracing types (inline)                                                    */
/* -------------------------------------------------------------------------- */

interface TracingSummary {
  llm_calls: number
  llm_errors: number
  total_tokens: number
  total_cost_usd: number
  total_latency_ms: number
  avg_latency_ms: number
  total_spans: number
}

interface TracingSpan {
  trace_id: string
  span_id: string
  parent_id: string | null
  operation: string
  kind: string
  status: string
  duration_ms: number
  attributes?: Record<string, any>
  events?: Array<Record<string, any>>
  error?: string
}

/* -------------------------------------------------------------------------- */
/*  Utilities                                                                 */
/* -------------------------------------------------------------------------- */

type MainTab = 'workbench' | 'zoo' | 'runs' | 'tracing'

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, init)
  const payload = await response.json().catch(() => null)
  if (!response.ok) {
    throw new Error(payload?.detail || payload?.message || 'Alpha Lab request failed')
  }
  return payload as T
}

function pad(value: number) {
  return String(value).padStart(2, '0')
}

function toDatetimeLocalValue(date: Date) {
  return [
    date.getFullYear(),
    '-',
    pad(date.getMonth() + 1),
    '-',
    pad(date.getDate()),
    'T',
    pad(date.getHours()),
    ':',
    pad(date.getMinutes()),
  ].join('')
}

function toApiDateTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toISOString()
}

function formatMetricValue(key: string, value?: number | null) {
  const numeric = Number(value)
  if (!Number.isFinite(numeric)) {
    return '--'
  }
  if (PERCENT_METRICS.has(key)) {
    return formatPercent(numeric, 2)
  }
  if (Math.abs(numeric) >= 1000) {
    return numeric.toLocaleString('en-US', { maximumFractionDigits: 2 })
  }
  if (Math.abs(numeric) >= 10) {
    return formatPrice(numeric, 2)
  }
  return formatPrice(numeric, 4)
}

function formatDateTime(value?: string | null) {
  if (!value) {
    return '未记录'
  }
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return date.toLocaleString('zh-CN', { hour12: false })
}

function formatRunDataset(run: AlphaLabRunSummary) {
  const dataset = run.dataset ?? {}
  const provider = typeof dataset.provider === 'string' ? dataset.provider : 'n/a'
  const interval = typeof dataset.interval === 'string' ? dataset.interval : 'n/a'
  const shape = Array.isArray(dataset.shape) ? dataset.shape.join(' x ') : 'n/a'
  return `${provider} · ${interval} · ${shape}`
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`
  return String(n)
}

/* -------------------------------------------------------------------------- */
/*  Component                                                                 */
/* -------------------------------------------------------------------------- */

export const AlphaLabWorkspace: React.FC = () => {
  /* ---- Core workspace state ---- */
  const [workspace, setWorkspace] = useState<AlphaLabWorkspacePayload | null>(null)
  const [formula, setFormula] = useState('')
  const [provider, setProvider] = useState('bitget')
  const [interval, setInterval] = useState('5m')
  const [symbolsInput, setSymbolsInput] = useState('BTCUSDT,ETHUSDT,SOLUSDT')
  const [startTime, setStartTime] = useState(() => toDatetimeLocalValue(new Date(Date.now() - 24 * 60 * 60 * 1000)))
  const [endTime, setEndTime] = useState(() => toDatetimeLocalValue(new Date()))
  const [validation, setValidation] = useState<AlphaLabValidationReport | null>(null)
  const [analysis, setAnalysis] = useState<AlphaLabEvaluationSummary | null>(null)

  /* ---- Runs state ---- */
  const [selectedRun, setSelectedRun] = useState<AlphaLabRunDetail | null>(null)
  const [expandedRunId, setExpandedRunId] = useState<string | null>(null)

  /* ---- Tracing state ---- */
  const [tracingSummary, setTracingSummary] = useState<TracingSummary | null>(null)
  const [tracingSpans, setTracingSpans] = useState<TracingSpan[]>([])
  const [tracingError, setTracingError] = useState<string | null>(null)

  /* ---- UI state ---- */
  const [mainTab, setMainTab] = useState<MainTab>('workbench')
  const [workspaceError, setWorkspaceError] = useState<string | null>(null)
  const [loadingWorkspace, setLoadingWorkspace] = useState(true)
  const [loadingRun, setLoadingRun] = useState(false)
  const [loadingTracing, setLoadingTracing] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [isSearching, setIsSearching] = useState(false)
  const [searchResult, setSearchResult] = useState<AlphaLabRunDetail | null>(null)
  const [searchPopSize, setSearchPopSize] = useState(6)
  const [searchOffspring, setSearchOffspring] = useState(3)
  const [searchGenerations, setSearchGenerations] = useState(3)
  const [searchTopK, setSearchTopK] = useState(5)
  const [searchSeeds, setSearchSeeds] = useState('')
  const [bootstrapped, setBootstrapped] = useState(false)
  const [zooSort, setZooSort] = useState<'fitness' | 'saved_at'>('fitness')

  const tracingTimerRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null)

  /* ---- Derived data ---- */
  const operatorCount = workspace?.operators.length ?? 0
  const zooCount = workspace?.zoo.length ?? 0
  const runCount = workspace?.runs.length ?? 0

  const defaultSymbols = useMemo(
    () =>
      Array.isArray(workspace?.defaults.crypto_market.default_symbols)
        ? (workspace?.defaults.crypto_market.default_symbols as string[])
        : [],
    [workspace],
  )

  const sampleFormulas = useMemo(
    () => (Array.isArray(workspace?.defaults.sample_formulas) ? workspace?.defaults.sample_formulas : []),
    [workspace],
  )

  const intervals = useMemo(
    () => (Array.isArray(workspace?.defaults.intervals) ? workspace?.defaults.intervals : ['1m', '5m', '15m', '1h', '4h']),
    [workspace],
  )

  const providers = useMemo(
    () => (Array.isArray(workspace?.defaults.providers) ? workspace?.defaults.providers : ['bitget']),
    [workspace],
  )

  const primaryMetricCards = useMemo(() => {
    const metrics = analysis?.metrics ?? {}
    return PRIMARY_METRICS.map((key) => ({
      key,
      label: METRIC_LABELS[key] ?? key,
      value: formatMetricValue(key, metrics[key]),
    }))
  }, [analysis])

  const sortedZoo = useMemo(() => {
    const entries = [...(workspace?.zoo ?? [])]
    if (zooSort === 'fitness') {
      entries.sort((a, b) => (b.fitness ?? -Infinity) - (a.fitness ?? -Infinity))
    } else {
      entries.sort((a, b) => {
        const ta = a.saved_at ? new Date(a.saved_at).getTime() : 0
        const tb = b.saved_at ? new Date(b.saved_at).getTime() : 0
        return tb - ta
      })
    }
    return entries
  }, [workspace?.zoo, zooSort])

  /* ---- API handlers ---- */
  const loadWorkspace = useCallback(async () => {
    try {
      setLoadingWorkspace(true)
      setWorkspaceError(null)
      const data = await requestJson<AlphaLabWorkspacePayload>('/alpha-lab/workspace')
      setWorkspace(data)
      if (!bootstrapped) {
        const nextProvider =
          typeof data.defaults.crypto_market.default_provider === 'string'
            ? data.defaults.crypto_market.default_provider
            : 'bitget'
        const nextInterval = data.defaults.intervals.includes('5m') ? '5m' : data.defaults.intervals[0] || '5m'
        const nextSymbols =
          Array.isArray(data.defaults.crypto_market.default_symbols) && data.defaults.crypto_market.default_symbols.length > 0
            ? data.defaults.crypto_market.default_symbols.join(',')
            : 'BTCUSDT,ETHUSDT,SOLUSDT'
        const nextFormula = data.zoo[0]?.formula || data.defaults.sample_formulas[0] || 'CSRank(ts_mean(close, 5) - close)'
        setProvider(nextProvider)
        setInterval(nextInterval)
        setSymbolsInput(nextSymbols)
        setFormula(nextFormula)
        setBootstrapped(true)
      }
    } catch (error) {
      setWorkspaceError(error instanceof Error ? error.message : 'Alpha Lab 工作台加载失败')
    } finally {
      setLoadingWorkspace(false)
    }
  }, [bootstrapped])

  const handleValidate = useCallback(async () => {
    try {
      setIsValidating(true)
      setWorkspaceError(null)
      const result = await requestJson<AlphaLabValidationReport>('/alpha-lab/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ formula }),
      })
      setValidation(result)
    } catch (error) {
      setValidation(null)
      setWorkspaceError(error instanceof Error ? error.message : '公式校验失败')
    } finally {
      setIsValidating(false)
    }
  }, [formula])

  const handleAnalyze = useCallback(async () => {
    try {
      setIsAnalyzing(true)
      setWorkspaceError(null)
      const result = await requestJson<AlphaLabEvaluationSummary>('/alpha-lab/evaluate-db', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          formula,
          provider,
          interval,
          symbols: symbolsInput
            .split(',')
            .map((item) => item.trim())
            .filter(Boolean),
          start_time: toApiDateTime(startTime),
          end_time: toApiDateTime(endTime),
          summary_only: true,
        }),
      })
      setAnalysis(result)
      setValidation({
        ok: true,
        normalized_formula: result.normalized_formula,
        errors: [],
        warnings: [],
      })
    } catch (error) {
      setAnalysis(null)
      setWorkspaceError(error instanceof Error ? error.message : '数据库评估失败')
    } finally {
      setIsAnalyzing(false)
    }
  }, [formula, provider, interval, symbolsInput, startTime, endTime])

  const handleSave = useCallback(async () => {
    try {
      setIsSaving(true)
      setWorkspaceError(null)
      await requestJson<AlphaLabZooEntry>('/alpha-lab/zoo', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          formula,
          fitness: analysis?.metrics?.sharpe,
          metrics: analysis?.metrics ?? {},
          lineage: { origin: analysis ? 'frontend-analysis' : 'frontend-manual' },
          note: analysis?.normalized_formula ? `normalized=${analysis.normalized_formula}` : undefined,
          source: 'frontend',
        }),
      })
      await loadWorkspace()
      setMainTab('zoo')
    } catch (error) {
      setWorkspaceError(error instanceof Error ? error.message : '保存因子失败')
    } finally {
      setIsSaving(false)
    }
  }, [formula, analysis, loadWorkspace])

  const handleSearch = useCallback(async () => {
    try {
      setIsSearching(true)
      setSearchResult(null)
      setWorkspaceError(null)
      const seeds = searchSeeds
        .split('\n')
        .map((s) => s.trim())
        .filter(Boolean)
      if (formula.trim() && !seeds.includes(formula.trim())) {
        seeds.unshift(formula.trim())
      }
      const result = await requestJson<AlphaLabRunDetail>('/alpha-lab/search-db', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider,
          interval,
          symbols: symbolsInput
            .split(',')
            .map((item) => item.trim())
            .filter(Boolean),
          start_time: toApiDateTime(startTime),
          end_time: toApiDateTime(endTime),
          seeds,
          population_size: searchPopSize,
          offspring_count: searchOffspring,
          generations: searchGenerations,
          top_k: searchTopK,
          persist: true,
        }),
      })
      setSearchResult(result)
      await loadWorkspace()
      setMainTab('runs')
    } catch (error) {
      setWorkspaceError(error instanceof Error ? error.message : 'GA 搜索运行失败')
    } finally {
      setIsSearching(false)
    }
  }, [formula, provider, interval, symbolsInput, startTime, endTime, searchSeeds, searchPopSize, searchOffspring, searchGenerations, searchTopK, loadWorkspace])

  const handleLoadRun = useCallback(async (runId: string) => {
    try {
      setLoadingRun(true)
      setWorkspaceError(null)
      const result = await requestJson<AlphaLabRunDetail>(`/alpha-lab/runs/${runId}`)
      setSelectedRun(result)
      setExpandedRunId(runId)
    } catch (error) {
      setSelectedRun(null)
      setWorkspaceError(error instanceof Error ? error.message : '运行详情加载失败')
    } finally {
      setLoadingRun(false)
    }
  }, [])

  const handleLoadFormula = useCallback((nextFormula: string) => {
    setFormula(nextFormula)
    setValidation(null)
    setAnalysis(null)
    setMainTab('workbench')
  }, [])

  const loadTracing = useCallback(async () => {
    try {
      setLoadingTracing(true)
      setTracingError(null)
      const [summary, spans] = await Promise.all([
        requestJson<TracingSummary>('/alpha-lab/tracing/summary'),
        requestJson<TracingSpan[]>('/alpha-lab/tracing/spans'),
      ])
      setTracingSummary(summary)
      setTracingSpans(spans)
    } catch (error) {
      setTracingError(error instanceof Error ? error.message : 'LLM 链路数据加载失败')
    } finally {
      setLoadingTracing(false)
    }
  }, [])

  const handleToggleRunExpand = useCallback(
    (runId: string) => {
      if (expandedRunId === runId) {
        setExpandedRunId(null)
        setSelectedRun(null)
      } else {
        void handleLoadRun(runId)
      }
    },
    [expandedRunId, handleLoadRun],
  )

  /* ---- Effects ---- */
  useEffect(() => {
    void loadWorkspace()
  }, [loadWorkspace])

  // Auto-refresh tracing data when the tracing tab is active
  useEffect(() => {
    if (mainTab === 'tracing') {
      void loadTracing()
      tracingTimerRef.current = globalThis.setInterval(() => {
        void loadTracing()
      }, TRACING_REFRESH_INTERVAL)
    }
    return () => {
      if (tracingTimerRef.current) {
        globalThis.clearInterval(tracingTimerRef.current)
        tracingTimerRef.current = null
      }
    }
  }, [mainTab, loadTracing])

  /* ---- Computed for overview ---- */
  const llmCallsCount = tracingSummary?.llm_calls ?? 0

  /* ======================================================================== */
  /*  Render                                                                  */
  /* ======================================================================== */

  return (
    <div className="space-y-6">
      {/* ------------------------------------------------------------------ */}
      {/*  Overview Metric Cards                                             */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="算子数量"
          value={operatorCount}
          hint="DSL / VM 可直接编排到工作台"
          trend={
            <span className="inline-flex items-center gap-2">
              <SearchCode className="size-4" />
              研究入口已贯通
            </span>
          }
        />
        <MetricCard
          label="因子库"
          value={zooCount}
          hint={defaultSymbols.length > 0 ? `默认关注 ${defaultSymbols.join(', ')}` : '可将分析结果一键沉淀为库内因子'}
          trend={
            <span className="inline-flex items-center gap-2">
              <LibraryBig className="size-4" />
              统一管理
            </span>
          }
        />
        <MetricCard
          label="搜索运行"
          value={runCount}
          hint="搜索运行和历史结果回收到同一入口"
          trend={
            <span className="inline-flex items-center gap-2">
              <Workflow className="size-4" />
              run / zoo 双视图
            </span>
          }
        />
        <MetricCard
          label="LLM 调用"
          value={llmCallsCount}
          hint={tracingSummary ? `${formatTokens(tracingSummary.total_tokens)} tokens · $${tracingSummary.total_cost_usd.toFixed(4)}` : '切换到 LLM 链路 tab 加载数据'}
          trend={
            <span className="inline-flex items-center gap-2">
              <Cpu className="size-4" />
              {tracingSummary ? `avg ${formatDuration(tracingSummary.avg_latency_ms)}` : '观测中'}
            </span>
          }
        />
      </div>

      {/* ------------------------------------------------------------------ */}
      {/*  Main Tabs                                                         */}
      {/* ------------------------------------------------------------------ */}
      <Tabs value={mainTab} onValueChange={(v) => setMainTab(v as MainTab)}>
        <TabsList>
          <TabsTrigger value="workbench">
            <Activity className="mr-1.5 size-3.5" />
            工作台
          </TabsTrigger>
          <TabsTrigger value="zoo">
            <LibraryBig className="mr-1.5 size-3.5" />
            因子库
          </TabsTrigger>
          <TabsTrigger value="runs">
            <Workflow className="mr-1.5 size-3.5" />
            搜索运行
          </TabsTrigger>
          <TabsTrigger value="tracing">
            <Eye className="mr-1.5 size-3.5" />
            LLM 链路
          </TabsTrigger>
        </TabsList>

        {/* ================================================================ */}
        {/*  Tab 1 - Workbench (因子工作台)                                   */}
        {/* ================================================================ */}
        <TabsContent value="workbench" className="space-y-6">
          <SectionCard
            title="因子工作台"
            description="在同一个面板里完成公式编写、数据窗口设定、校验、数据库评估和入库。"
            action={
              <Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loadingWorkspace}>
                <RefreshCw className={`size-4 ${loadingWorkspace ? 'animate-spin' : ''}`} />
                刷新
              </Button>
            }
          >
            {/* Formula editor */}
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <label className="text-sm font-semibold text-foreground">DSL 公式</label>
                {validation ? (
                  <Badge variant={validation.ok ? 'success' : 'danger'}>
                    {validation.ok ? '已校验' : '校验失败'}
                  </Badge>
                ) : null}
              </div>
              <textarea
                value={formula}
                onChange={(event) => setFormula(event.target.value)}
                placeholder="例如：CSRank(ts_mean(close, 5) - close)"
                className="min-h-32 w-full rounded-2xl border border-border bg-input px-4 py-3 text-sm text-foreground shadow-sm outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30"
              />
            </div>

            {/* Sample formulas */}
            {sampleFormulas.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {sampleFormulas.slice(0, 3).map((sample) => (
                  <button
                    key={sample}
                    type="button"
                    onClick={() => handleLoadFormula(sample)}
                    className="rounded-full border border-border/80 bg-secondary/60 px-3 py-1.5 text-xs font-medium text-muted-foreground transition hover:border-primary/30 hover:text-foreground"
                  >
                    {sample}
                  </button>
                ))}
              </div>
            ) : null}

            {/* Data window controls */}
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Provider</label>
                <select
                  value={provider}
                  onChange={(event) => setProvider(event.target.value)}
                  className="h-11 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30"
                >
                  {providers.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Interval</label>
                <select
                  value={interval}
                  onChange={(event) => setInterval(event.target.value)}
                  className="h-11 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30"
                >
                  {intervals.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Start</label>
                <Input type="datetime-local" value={startTime} onChange={(event) => setStartTime(event.target.value)} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">End</label>
                <Input type="datetime-local" value={endTime} onChange={(event) => setEndTime(event.target.value)} />
              </div>
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Symbols</label>
              <Input value={symbolsInput} onChange={(event) => setSymbolsInput(event.target.value)} placeholder="BTCUSDT,ETHUSDT,SOLUSDT" />
            </div>

            {/* Action buttons */}
            <div className="flex flex-wrap items-center gap-3">
              <Button variant="outline" onClick={() => void handleValidate()} disabled={isValidating || !formula.trim()}>
                {isValidating ? <RefreshCw className="animate-spin" /> : <CheckCircle2 />}
                校验公式
              </Button>
              <Button onClick={() => void handleAnalyze()} disabled={isAnalyzing || !formula.trim()}>
                {isAnalyzing ? <RefreshCw className="animate-spin" /> : <Activity />}
                数据库分析
              </Button>
              <Button variant="secondary" onClick={() => void handleSave()} disabled={isSaving || !formula.trim()}>
                {isSaving ? <RefreshCw className="animate-spin" /> : <Save />}
                保存到因子库
              </Button>
            </div>

            {/* Error display */}
            {workspaceError ? (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {workspaceError}
              </div>
            ) : null}
          </SectionCard>

          {/* ---- GA Search Panel ---- */}
          <SectionCard
            title="GA 因子搜索"
            description="基于当前数据窗口，运行遗传算法+LLM 进化搜索。当前公式会作为种子加入初始种群。"
          >
            <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">种群大小</label>
                <Input type="number" min={2} max={32} value={searchPopSize} onChange={(e) => setSearchPopSize(Number(e.target.value) || 6)} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">子代数量</label>
                <Input type="number" min={1} max={16} value={searchOffspring} onChange={(e) => setSearchOffspring(Number(e.target.value) || 3)} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">代数</label>
                <Input type="number" min={1} max={20} value={searchGenerations} onChange={(e) => setSearchGenerations(Number(e.target.value) || 3)} />
              </div>
              <div className="space-y-2">
                <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Top-K</label>
                <Input type="number" min={1} max={20} value={searchTopK} onChange={(e) => setSearchTopK(Number(e.target.value) || 5)} />
              </div>
            </div>
            <div className="space-y-2">
              <label className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">额外种子公式（每行一个，可选）</label>
              <textarea
                value={searchSeeds}
                onChange={(e) => setSearchSeeds(e.target.value)}
                placeholder="cs_rank(ts_std(close, 10))&#10;cs_rank(volatility_n(close, 20))"
                rows={3}
                className="w-full rounded-2xl border border-border bg-input px-4 py-3 text-xs font-mono text-foreground shadow-sm outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30"
              />
            </div>
            <div className="flex items-center gap-3">
              <Button onClick={() => void handleSearch()} disabled={isSearching}>
                {isSearching ? <RefreshCw className="animate-spin" /> : <Zap />}
                {isSearching ? '搜索中...' : '启动 GA 搜索'}
              </Button>
              {isSearching && (
                <span className="text-xs text-muted-foreground animate-pulse">
                  搜索可能需要数分钟，请勿关闭页面
                </span>
              )}
            </div>
            {searchResult && (
              <div className="rounded-2xl border border-emerald-500/20 bg-emerald-500/10 p-4 space-y-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-emerald-300">
                  <CheckCircle2 className="size-4" />
                  搜索完成
                  {searchResult.run_id && <Badge variant="info">{searchResult.run_id}</Badge>}
                </div>
                {searchResult.top_results?.slice(0, 3).map((item, idx) => (
                  <div key={item.expr_hash ?? idx} className="flex items-center justify-between gap-3 rounded-xl bg-card/70 px-3 py-2">
                    <div className="min-w-0">
                      <div className="truncate font-mono text-xs text-foreground">{item.formula}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        fit {formatMetricValue('sharpe', item.fitness)} · sharpe {formatMetricValue('sharpe', item.metrics?.sharpe)} · IC {formatMetricValue('rank_ic', item.metrics?.rank_ic)}
                      </div>
                    </div>
                    <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(item.formula)}>装载</Button>
                  </div>
                ))}
              </div>
            )}

          </SectionCard>

          {/* Validation result */}
          {validation ? (
            <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
              <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                {validation.ok ? (
                  <CheckCircle2 className="size-4 text-emerald-300" />
                ) : (
                  <AlertTriangle className="size-4 text-amber-300" />
                )}
                {validation.ok ? '公式通过校验' : '公式校验未通过'}
              </div>
              {validation.normalized_formula ? (
                <p className="mt-2 break-all font-mono text-xs text-muted-foreground">{validation.normalized_formula}</p>
              ) : null}
              {!validation.ok && validation.errors && validation.errors.length > 0 ? (
                <div className="mt-3 space-y-1 text-xs text-rose-200">
                  {validation.errors.map((item) => (
                    <div key={item}>{item}</div>
                  ))}
                </div>
              ) : null}
            </div>
          ) : null}

          {/* Analysis results */}
          <SectionCard
            title="分析结果"
            description="以数据库数据为基底输出表达式哈希、关键指标和尾部序列预览。"
          >
            {analysis ? (
              <div className="space-y-5">
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-3">
                  {primaryMetricCards.map((metric) => (
                    <div key={metric.key} className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                      <div className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        {metric.label}
                      </div>
                      <div className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-foreground">{metric.value}</div>
                    </div>
                  ))}
                </div>

                <div className="flex flex-wrap gap-2">
                  <Badge variant="info">{analysis.dataset?.provider ?? provider}</Badge>
                  <Badge>{analysis.dataset?.interval ?? interval}</Badge>
                  <Badge>{analysis.backend ?? 'backend:n/a'}</Badge>
                  <Badge>{analysis.device ?? 'device:n/a'}</Badge>
                  {analysis.dataset?.shape ? (
                    <Badge>
                      {analysis.dataset.shape[0]} x {analysis.dataset.shape[1]}
                    </Badge>
                  ) : null}
                </div>

                <div className="grid gap-4 lg:grid-cols-2">
                  <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Expr Hash</div>
                    <div className="mt-2 break-all font-mono text-xs text-foreground">{analysis.expr_hash ?? '未返回'}</div>
                  </div>
                  <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Normalized Formula</div>
                    <div className="mt-2 break-all font-mono text-xs text-foreground">
                      {analysis.normalized_formula ?? formula}
                    </div>
                  </div>
                </div>

                {/* Equity curve chart */}
                {(analysis as any).equity_series?.length > 0 && (
                  <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground mb-3">
                      权益曲线
                    </div>
                    <ResponsiveContainer width="100%" height={220}>
                      <ComposedChart data={(analysis as any).equity_series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                        <XAxis dataKey="i" tick={false} axisLine={false} />
                        <YAxis
                          domain={['auto', 'auto']}
                          tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                          width={52}
                          tickFormatter={(v: number) => v >= 1 ? v.toFixed(2) : v.toFixed(3)}
                        />
                        <Tooltip
                          contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 12, fontSize: 12 }}
                          formatter={(v: number) => [v.toFixed(4), '权益']}
                          labelFormatter={(i: number) => `Bar ${i}`}
                        />
                        <Area type="monotone" dataKey="v" stroke="hsl(var(--primary))" fill="hsl(var(--primary))" fillOpacity={0.1} strokeWidth={1.5} dot={false} />
                        <Line type="monotone" dataKey="v" stroke="hsl(var(--primary))" strokeWidth={1.5} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Drawdown chart */}
                {(analysis as any).drawdown_series?.length > 0 && (
                  <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground mb-3">
                      回撤曲线
                    </div>
                    <ResponsiveContainer width="100%" height={160}>
                      <ComposedChart data={(analysis as any).drawdown_series} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
                        <XAxis dataKey="i" tick={false} axisLine={false} />
                        <YAxis
                          domain={[0, 'auto']}
                          tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
                          width={52}
                          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                        />
                        <Tooltip
                          contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 12, fontSize: 12 }}
                          formatter={(v: number) => [`${(v * 100).toFixed(2)}%`, '回撤']}
                          labelFormatter={(i: number) => `Bar ${i}`}
                        />
                        <Area type="monotone" dataKey="v" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.15} strokeWidth={1.5} dot={false} />
                      </ComposedChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>
            ) : (
              <EmptyState
                title="尚未生成分析结果"
                description="先完成公式校验或直接发起数据库分析，结果会在这里汇总展示。"
                action={
                  <Button onClick={() => void handleAnalyze()} disabled={isAnalyzing || !formula.trim()}>
                    <Database className="size-4" />
                    开始分析
                  </Button>
                }
              />
            )}
          </SectionCard>
        </TabsContent>

        {/* ================================================================ */}
        {/*  Tab 2 - Factor Zoo (因子库)                                      */}
        {/* ================================================================ */}
        <TabsContent value="zoo">
          <SectionCard
            title="因子库"
            description="所有沉淀的因子。点击「载入」可装载到工作台进行再次分析。"
            action={
              <div className="flex items-center gap-2">
                <select
                  value={zooSort}
                  onChange={(e) => setZooSort(e.target.value as 'fitness' | 'saved_at')}
                  className="h-8 rounded-lg border border-border bg-input px-2 text-xs text-foreground outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30"
                >
                  <option value="fitness">按 Fitness 排序</option>
                  <option value="saved_at">按时间排序</option>
                </select>
                <Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loadingWorkspace}>
                  <RefreshCw className={`size-4 ${loadingWorkspace ? 'animate-spin' : ''}`} />
                  刷新
                </Button>
              </div>
            }
          >
            {loadingWorkspace ? (
              <div className="rounded-2xl border border-border/70 bg-secondary/35 px-4 py-8 text-center text-sm text-muted-foreground">
                正在加载因子库...
              </div>
            ) : sortedZoo.length > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border/70">
                      <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Formula
                      </th>
                      <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Fitness
                      </th>
                      <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Sharpe
                      </th>
                      <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Rank IC
                      </th>
                      <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Turnover
                      </th>
                      <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Source
                      </th>
                      <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        Tags
                      </th>
                      <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                        保存时间
                      </th>
                      <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground" />
                    </tr>
                  </thead>
                  <tbody>
                    {sortedZoo.map((entry) => (
                      <tr
                        key={entry.expr_hash ?? entry.formula}
                        className="border-b border-border/40 transition hover:bg-accent/30"
                      >
                        <td className="max-w-xs px-3 py-3">
                          <div className="truncate font-mono text-xs text-foreground" title={entry.formula}>
                            {entry.formula}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                          {formatMetricValue('sharpe', entry.fitness)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                          {formatMetricValue('sharpe', entry.metrics?.sharpe)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                          {formatMetricValue('rank_ic', entry.metrics?.rank_ic)}
                        </td>
                        <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                          {formatMetricValue('avg_turnover', entry.metrics?.avg_turnover)}
                        </td>
                        <td className="px-3 py-3">
                          {entry.source ? <Badge>{entry.source}</Badge> : <span className="text-xs text-muted-foreground">--</span>}
                        </td>
                        <td className="px-3 py-3">
                          <div className="flex flex-wrap gap-1">
                            {entry.tags?.slice(0, 3).map((tag) => (
                              <Badge key={tag}>{tag}</Badge>
                            ))}
                          </div>
                        </td>
                        <td className="whitespace-nowrap px-3 py-3 text-xs text-muted-foreground">
                          {formatDateTime(entry.saved_at)}
                        </td>
                        <td className="px-3 py-3 text-right">
                          <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(entry.formula)}>
                            载入
                          </Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <EmptyState
                title="因子库为空"
                description="当前还没有沉淀的因子，可从工作台分析后直接入库。"
              />
            )}
          </SectionCard>
        </TabsContent>

        {/* ================================================================ */}
        {/*  Tab 3 - Search Runs (搜索运行)                                   */}
        {/* ================================================================ */}
        <TabsContent value="runs">
          <SectionCard
            title="搜索运行"
            description="数据库搜索运行历史，点击展开查看 top results 和 lineage。"
            action={
              <Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loadingWorkspace}>
                <RefreshCw className={`size-4 ${loadingWorkspace ? 'animate-spin' : ''}`} />
                刷新
              </Button>
            }
          >
            {loadingWorkspace ? (
              <div className="rounded-2xl border border-border/70 bg-secondary/35 px-4 py-8 text-center text-sm text-muted-foreground">
                正在加载运行记录...
              </div>
            ) : workspace?.runs.length ? (
              <div className="space-y-3">
                {workspace.runs.map((run) => {
                  const isExpanded = expandedRunId === run.run_id
                  return (
                    <div key={run.run_id} className="rounded-2xl border border-border/70 bg-secondary/30 transition">
                      {/* Run header - clickable to expand */}
                      <button
                        type="button"
                        onClick={() => handleToggleRunExpand(run.run_id)}
                        className="flex w-full items-center justify-between gap-4 p-4 text-left transition hover:bg-accent/30"
                      >
                        <div className="flex items-center gap-3">
                          {isExpanded ? (
                            <ChevronDown className="size-4 shrink-0 text-muted-foreground" />
                          ) : (
                            <ChevronRight className="size-4 shrink-0 text-muted-foreground" />
                          )}
                          <div className="min-w-0">
                            <div className="truncate text-sm font-semibold text-foreground">{run.run_id}</div>
                            <div className="mt-1 text-xs text-muted-foreground">{formatRunDataset(run)}</div>
                          </div>
                        </div>
                        <div className="flex shrink-0 items-center gap-3">
                          <Badge variant="info">{run.top_results ?? 0} top</Badge>
                          <span className="text-xs text-muted-foreground">{formatDateTime(run.saved_at)}</span>
                        </div>
                      </button>

                      {/* Expanded detail */}
                      {isExpanded ? (
                        <div className="border-t border-border/40 p-4">
                          {loadingRun ? (
                            <div className="flex items-center justify-center gap-2 py-6 text-sm text-muted-foreground">
                              <RefreshCw className="size-4 animate-spin" />
                              正在加载运行详情...
                            </div>
                          ) : selectedRun ? (
                            <div className="space-y-4">
                              <div className="flex flex-wrap items-center gap-2">
                                <Badge variant="info">{selectedRun.run_id}</Badge>
                                {selectedRun.dataset?.provider ? <Badge>{selectedRun.dataset.provider}</Badge> : null}
                                {selectedRun.dataset?.interval ? <Badge>{selectedRun.dataset.interval}</Badge> : null}
                                {selectedRun.lineage ? (
                                  <Badge>
                                    lineage: {selectedRun.lineage.length}
                                  </Badge>
                                ) : null}
                              </div>

                              {(selectedRun.top_results ?? []).length > 0 ? (
                                <div className="space-y-2">
                                  <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                                    Top Results
                                  </div>
                                  {(selectedRun.top_results ?? []).slice(0, 8).map((item) => (
                                    <div
                                      key={item.expr_hash ?? item.formula}
                                      className="flex items-center justify-between gap-3 rounded-xl border border-border/50 bg-card/70 px-4 py-3"
                                    >
                                      <div className="min-w-0 flex-1">
                                        <div className="truncate font-mono text-[11px] leading-6 text-foreground">
                                          {item.formula}
                                        </div>
                                        <div className="mt-1 flex flex-wrap gap-3 text-xs text-muted-foreground">
                                          <span>fit {formatMetricValue('sharpe', item.fitness)}</span>
                                          {item.metrics?.sharpe != null ? (
                                            <span>sharpe {formatMetricValue('sharpe', item.metrics.sharpe)}</span>
                                          ) : null}
                                          {item.metrics?.rank_ic != null ? (
                                            <span>rank_ic {formatMetricValue('rank_ic', item.metrics.rank_ic)}</span>
                                          ) : null}
                                        </div>
                                      </div>
                                      <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(item.formula)}>
                                        载入
                                      </Button>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <div className="py-4 text-center text-xs text-muted-foreground">该运行暂无 top results</div>
                              )}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
                    </div>
                  )
                })}
              </div>
            ) : (
              <EmptyState title="暂无运行记录" description="数据库搜索运行落盘后，会自动出现在这里。" />
            )}
          </SectionCard>
        </TabsContent>

        {/* ================================================================ */}
        {/*  Tab 4 - LLM Observability (LLM 链路)                            */}
        {/* ================================================================ */}
        <TabsContent value="tracing">
          <div className="space-y-6">
            {/* Tracing summary cards */}
            <div className="grid gap-4 md:grid-cols-3 xl:grid-cols-5">
              <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  <Zap className="size-3.5" />
                  Total LLM Calls
                </div>
                <div className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-foreground">
                  {tracingSummary?.llm_calls ?? '--'}
                </div>
              </div>
              <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  <Database className="size-3.5" />
                  Total Tokens
                </div>
                <div className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-foreground">
                  {tracingSummary ? formatTokens(tracingSummary.total_tokens) : '--'}
                </div>
              </div>
              <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  <Activity className="size-3.5" />
                  Total Cost
                </div>
                <div className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-foreground">
                  {tracingSummary ? `$${tracingSummary.total_cost_usd.toFixed(4)}` : '--'}
                </div>
              </div>
              <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  <Clock className="size-3.5" />
                  Avg Latency
                </div>
                <div className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-foreground">
                  {tracingSummary ? formatDuration(tracingSummary.avg_latency_ms) : '--'}
                </div>
              </div>
              <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                  <AlertTriangle className="size-3.5" />
                  Error Rate
                </div>
                <div className="mt-3 text-2xl font-semibold tracking-[-0.03em] text-foreground">
                  {tracingSummary && tracingSummary.llm_calls > 0
                    ? formatPercent(tracingSummary.llm_errors / tracingSummary.llm_calls, 1)
                    : '--'}
                </div>
              </div>
            </div>

            {/* Tracing error */}
            {tracingError ? (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {tracingError}
              </div>
            ) : null}

            {/* Recent spans table */}
            <SectionCard
              title="Recent Spans"
              description="最近的 LLM 调用追踪记录，每 10 秒自动刷新。"
              action={
                <Button variant="outline" size="sm" onClick={() => void loadTracing()} disabled={loadingTracing}>
                  <RefreshCw className={`size-4 ${loadingTracing ? 'animate-spin' : ''}`} />
                  刷新
                </Button>
              }
            >
              {loadingTracing && tracingSpans.length === 0 ? (
                <div className="rounded-2xl border border-border/70 bg-secondary/35 px-4 py-8 text-center text-sm text-muted-foreground">
                  正在加载追踪数据...
                </div>
              ) : tracingSpans.length > 0 ? (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border/70">
                        <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Operation
                        </th>
                        <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Model
                        </th>
                        <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Tokens
                        </th>
                        <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Cost
                        </th>
                        <th className="px-3 py-3 text-right text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Duration
                        </th>
                        <th className="px-3 py-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">
                          Status
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {tracingSpans.map((span) => {
                        const model = span.attributes?.model ?? span.attributes?.['llm.model'] ?? '--'
                        const tokens =
                          span.attributes?.total_tokens ??
                          span.attributes?.['llm.total_tokens'] ??
                          null
                        const cost =
                          span.attributes?.cost_usd ??
                          span.attributes?.['llm.cost_usd'] ??
                          null
                        const isError = span.status === 'error' || span.status === 'ERROR'

                        return (
                          <tr
                            key={`${span.trace_id}-${span.span_id}`}
                            className="border-b border-border/40 transition hover:bg-accent/30"
                          >
                            <td className="max-w-xs px-3 py-3">
                              <div className="truncate text-xs font-medium text-foreground" title={span.operation}>
                                {span.operation}
                              </div>
                              {span.error ? (
                                <div className="mt-1 truncate text-[11px] text-rose-300" title={span.error}>
                                  {span.error}
                                </div>
                              ) : null}
                            </td>
                            <td className="whitespace-nowrap px-3 py-3 font-mono text-xs text-muted-foreground">
                              {String(model)}
                            </td>
                            <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                              {tokens != null ? formatTokens(Number(tokens)) : '--'}
                            </td>
                            <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                              {cost != null ? `$${Number(cost).toFixed(4)}` : '--'}
                            </td>
                            <td className="whitespace-nowrap px-3 py-3 text-right font-mono text-xs text-foreground">
                              {formatDuration(span.duration_ms)}
                            </td>
                            <td className="px-3 py-3">
                              <Badge variant={isError ? 'danger' : 'success'}>
                                {span.status}
                              </Badge>
                            </td>
                          </tr>
                        )
                      })}
                    </tbody>
                  </table>
                </div>
              ) : (
                <EmptyState
                  title="暂无追踪数据"
                  description="LLM 调用产生的追踪记录会在这里自动出现。"
                />
              )}
            </SectionCard>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  )
}

export default AlphaLabWorkspace
