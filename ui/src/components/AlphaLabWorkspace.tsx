import React, { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  CheckCircle2,
  Cpu,
  Database,
  LibraryBig,
  RefreshCw,
  Save,
  SearchCode,
  Workflow,
} from 'lucide-react'

import type {
  AlphaLabEvaluationSummary,
  AlphaLabRunDetail,
  AlphaLabRunSummary,
  AlphaLabValidationReport,
  AlphaLabWorkspace as AlphaLabWorkspacePayload,
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

type SideTab = 'zoo' | 'runs'

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

export const AlphaLabWorkspace: React.FC = () => {
  const [workspace, setWorkspace] = useState<AlphaLabWorkspacePayload | null>(null)
  const [formula, setFormula] = useState('')
  const [provider, setProvider] = useState('bitget')
  const [interval, setInterval] = useState('5m')
  const [symbolsInput, setSymbolsInput] = useState('BTCUSDT,ETHUSDT,SOLUSDT')
  const [startTime, setStartTime] = useState(() => toDatetimeLocalValue(new Date(Date.now() - 24 * 60 * 60 * 1000)))
  const [endTime, setEndTime] = useState(() => toDatetimeLocalValue(new Date()))
  const [validation, setValidation] = useState<AlphaLabValidationReport | null>(null)
  const [analysis, setAnalysis] = useState<AlphaLabEvaluationSummary | null>(null)
  const [selectedRun, setSelectedRun] = useState<AlphaLabRunDetail | null>(null)
  const [sideTab, setSideTab] = useState<SideTab>('zoo')
  const [workspaceError, setWorkspaceError] = useState<string | null>(null)
  const [loadingWorkspace, setLoadingWorkspace] = useState(true)
  const [loadingRun, setLoadingRun] = useState(false)
  const [isValidating, setIsValidating] = useState(false)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  const [bootstrapped, setBootstrapped] = useState(false)

  const operatorCount = workspace?.operators.length ?? 0
  const zooCount = workspace?.zoo.length ?? 0
  const runCount = workspace?.runs.length ?? 0
  const defaultSymbols = Array.isArray(workspace?.defaults.crypto_market.default_symbols)
    ? (workspace?.defaults.crypto_market.default_symbols as string[])
    : []
  const sampleFormulas = Array.isArray(workspace?.defaults.sample_formulas)
    ? workspace?.defaults.sample_formulas
    : []
  const intervals = Array.isArray(workspace?.defaults.intervals)
    ? workspace?.defaults.intervals
    : ['1m', '5m', '15m', '1h', '4h']
  const providers = Array.isArray(workspace?.defaults.providers)
    ? workspace?.defaults.providers
    : ['bitget']

  const primaryMetricCards = useMemo(() => {
    const metrics = analysis?.metrics ?? {}
    return PRIMARY_METRICS.map((key) => ({
      key,
      label: METRIC_LABELS[key] ?? key,
      value: formatMetricValue(key, metrics[key]),
    }))
  }, [analysis])

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

  useEffect(() => {
    void loadWorkspace()
  }, [loadWorkspace])

  const handleValidate = async () => {
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
  }

  const handleAnalyze = async () => {
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
  }

  const handleSave = async () => {
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
      setSideTab('zoo')
    } catch (error) {
      setWorkspaceError(error instanceof Error ? error.message : '保存因子失败')
    } finally {
      setIsSaving(false)
    }
  }

  const handleLoadRun = async (runId: string) => {
    try {
      setLoadingRun(true)
      setWorkspaceError(null)
      const result = await requestJson<AlphaLabRunDetail>(`/alpha-lab/runs/${runId}`)
      setSelectedRun(result)
      setSideTab('runs')
    } catch (error) {
      setSelectedRun(null)
      setWorkspaceError(error instanceof Error ? error.message : '运行详情加载失败')
    } finally {
      setLoadingRun(false)
    }
  }

  const handleLoadFormula = (nextFormula: string) => {
    setFormula(nextFormula)
    setValidation(null)
    setAnalysis(null)
  }

  return (
    <div className="space-y-6">
      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="算子目录"
          value={operatorCount}
          hint="DSL / VM 可直接编排到前端工作台。"
          trend={<span className="inline-flex items-center gap-2"><SearchCode className="size-4" />研究入口已贯通</span>}
        />
        <MetricCard
          label="因子库"
          value={zooCount}
          hint={defaultSymbols.length > 0 ? `默认关注 ${defaultSymbols.join(', ')}` : '可将分析结果一键沉淀为库内因子。'}
          trend={<span className="inline-flex items-center gap-2"><LibraryBig className="size-4" />统一管理</span>}
        />
        <MetricCard
          label="实验运行"
          value={runCount}
          hint="搜索运行和历史结果都回收到同一入口。"
          trend={<span className="inline-flex items-center gap-2"><Workflow className="size-4" />run / zoo 双视图</span>}
        />
        <MetricCard
          label="默认后端"
          value={String(workspace?.defaults.alpha_lab.llm_backend ?? 'n/a')}
          hint={`数据源 ${provider} · 评估周期 ${interval}`}
          trend={<span className="inline-flex items-center gap-2"><Cpu className="size-4" />前后端统一配置</span>}
        />
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.35fr)_minmax(360px,0.92fr)]">
        <div className="space-y-6">
          <SectionCard
            title="因子工作台"
            description="在同一个面板里完成公式编写、数据窗口设定、校验、数据库评估和入库，不再分散到 CLI 或多个页面。"
            action={
              <Button variant="outline" size="sm" onClick={() => void loadWorkspace()} disabled={loadingWorkspace}>
                <RefreshCw className={`size-4 ${loadingWorkspace ? 'animate-spin' : ''}`} />
                刷新
              </Button>
            }
          >
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

            {workspaceError ? (
              <div className="rounded-2xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                {workspaceError}
              </div>
            ) : null}

            {validation ? (
              <div className="rounded-2xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  {validation.ok ? <CheckCircle2 className="size-4 text-emerald-300" /> : <AlertTriangle className="size-4 text-amber-300" />}
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
          </SectionCard>

          <SectionCard
            title="分析结果"
            description="以数据库数据为基底输出表达式哈希、关键指标和尾部序列预览，足够支撑前端分析与入库决策。"
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
                  {analysis.dataset?.shape ? <Badge>{analysis.dataset.shape[0]} x {analysis.dataset.shape[1]}</Badge> : null}
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

                <div className="grid gap-4 lg:grid-cols-3">
                  <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Alpha Tail</div>
                    <pre className="mt-3 overflow-x-auto text-[11px] text-muted-foreground">{JSON.stringify(analysis.alpha_tail ?? [], null, 2)}</pre>
                  </div>
                  <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Weights Tail</div>
                    <pre className="mt-3 overflow-x-auto text-[11px] text-muted-foreground">{JSON.stringify(analysis.weights_tail ?? [], null, 2)}</pre>
                  </div>
                  <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
                    <div className="text-xs font-semibold uppercase tracking-[0.18em] text-muted-foreground">Equity Tail</div>
                    <pre className="mt-3 overflow-x-auto text-[11px] text-muted-foreground">{JSON.stringify(analysis.equity_tail ?? [], null, 2)}</pre>
                  </div>
                </div>
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
        </div>

        <SectionCard
          title="研究资产"
          description="左侧负责产出，右侧负责沉淀与复用。因子库和历史运行都能反向装载到当前工作台。"
        >
          <Tabs value={sideTab} onValueChange={(value) => setSideTab(value as SideTab)}>
            <TabsList>
              <TabsTrigger value="zoo">因子库</TabsTrigger>
              <TabsTrigger value="runs">运行记录</TabsTrigger>
            </TabsList>

            <TabsContent value="zoo" className="space-y-3">
              {loadingWorkspace ? (
                <div className="rounded-2xl border border-border/70 bg-secondary/35 px-4 py-8 text-center text-sm text-muted-foreground">
                  正在加载因子库...
                </div>
              ) : workspace?.zoo.length ? (
                workspace.zoo.map((entry) => (
                  <div key={entry.expr_hash ?? entry.formula} className="rounded-2xl border border-border/70 bg-secondary/30 p-4">
                    <div className="flex items-start justify-between gap-3">
                      <div className="min-w-0">
                        <div className="line-clamp-3 font-mono text-xs leading-6 text-foreground">{entry.formula}</div>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {typeof entry.fitness === 'number' ? <Badge variant="info">fit {formatMetricValue('sharpe', entry.fitness)}</Badge> : null}
                          {entry.source ? <Badge>{entry.source}</Badge> : null}
                          {entry.tags?.slice(0, 2).map((tag) => (
                            <Badge key={tag}>{tag}</Badge>
                          ))}
                        </div>
                      </div>
                      <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(entry.formula)}>
                        载入
                      </Button>
                    </div>
                    {entry.metrics ? (
                      <div className="mt-4 grid gap-2 sm:grid-cols-2">
                        {Object.entries(entry.metrics)
                          .slice(0, 4)
                          .map(([key, value]) => (
                            <div key={key} className="rounded-xl bg-card/70 px-3 py-2 text-xs text-muted-foreground">
                              <span className="font-semibold text-foreground">{METRIC_LABELS[key] ?? key}</span>
                              <span className="ml-2">{formatMetricValue(key, value)}</span>
                            </div>
                          ))}
                      </div>
                    ) : null}
                    {entry.note ? <p className="mt-3 text-xs leading-6 text-muted-foreground">{entry.note}</p> : null}
                  </div>
                ))
              ) : (
                <EmptyState title="因子库为空" description="当前还没有沉淀的前端因子，可从左侧工作台分析后直接入库。" />
              )}
            </TabsContent>

            <TabsContent value="runs" className="space-y-4">
              {loadingWorkspace ? (
                <div className="rounded-2xl border border-border/70 bg-secondary/35 px-4 py-8 text-center text-sm text-muted-foreground">
                  正在加载运行记录...
                </div>
              ) : workspace?.runs.length ? (
                <div className="space-y-3">
                  {workspace.runs.map((run) => (
                    <button
                      key={run.run_id}
                      type="button"
                      onClick={() => void handleLoadRun(run.run_id)}
                      className="w-full rounded-2xl border border-border/70 bg-secondary/30 p-4 text-left transition hover:border-primary/30 hover:bg-accent/45"
                    >
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0">
                          <div className="truncate text-sm font-semibold text-foreground">{run.run_id}</div>
                          <div className="mt-1 text-xs text-muted-foreground">{formatRunDataset(run)}</div>
                          <div className="mt-1 text-xs text-muted-foreground">{formatDateTime(run.saved_at)}</div>
                        </div>
                        <Badge variant="info">{run.top_results ?? 0} top</Badge>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <EmptyState title="暂无运行记录" description="数据库搜索运行落盘后，会自动出现在这里。" />
              )}

              {loadingRun ? (
                <div className="rounded-2xl border border-border/70 bg-secondary/35 px-4 py-8 text-center text-sm text-muted-foreground">
                  正在加载运行详情...
                </div>
              ) : selectedRun ? (
                <div className="rounded-3xl border border-border/70 bg-card/70 p-4">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant="info">{selectedRun.run_id}</Badge>
                    {selectedRun.dataset?.provider ? <Badge>{selectedRun.dataset.provider}</Badge> : null}
                    {selectedRun.dataset?.interval ? <Badge>{selectedRun.dataset.interval}</Badge> : null}
                  </div>
                  <div className="mt-4 space-y-3">
                    {(selectedRun.top_results ?? []).slice(0, 5).map((item) => (
                      <div key={item.expr_hash ?? item.formula} className="rounded-2xl border border-border/70 bg-secondary/30 p-3">
                        <div className="font-mono text-[11px] leading-6 text-foreground">{item.formula}</div>
                        <div className="mt-3 flex items-center justify-between gap-3">
                          <div className="text-xs text-muted-foreground">
                            fit {formatMetricValue('sharpe', item.fitness)}
                          </div>
                          <Button variant="ghost" size="sm" onClick={() => handleLoadFormula(item.formula)}>
                            装载
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </TabsContent>
          </Tabs>
        </SectionCard>
      </div>
    </div>
  )
}

export default AlphaLabWorkspace
