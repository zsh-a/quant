/**
 * Search tab — configure and execute automated alpha search.
 *
 * Supports multiple concurrent search jobs. Job state is managed by
 * useSearchJobs (in workspace) and survives page refresh via backend recovery.
 */
import React, { useCallback, useMemo, useState } from 'react'
import { ChevronRight, Info, Loader2, RotateCcw, Settings2, StopCircle, X, Zap } from 'lucide-react'
import type { AlphaLabSearchJob, AlphaLabWorkspace as WorkspacePayload, StrategyModeInfo } from '../../types'
import type { SearchJobsState } from '../../hooks/useSearchJobs'
import { SectionCard } from '../layout/SectionCard'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { SearchProgress } from './SearchProgress'
import { LLMAnalysis } from './LLMAnalysis'
import { DataScopeSection } from './DataScopeSection'
import { JobsDashboard } from './JobsDashboard'
import { toISO } from './shared'

/* ── Strategy descriptions (fallback when backend unavailable) ────── */

const FALLBACK_MODES: StrategyModeInfo[] = [
  {
    mode: 'evolution', label: 'Evolution (LLM + Enum)',
    brief: 'LLM 驱动的进化搜索 + 程序化枚举',
    detail: 'Round 0: 枚举种子 → 后续轮次: LLM 进化 + CPCV 评估。',
    strategies: ['enumeration', 'llm_evolution'],
    params: ['popSize', 'offspring', 'gens', 'topK', 'nSplits', 'enumMax', 'enumTopK'],
  },
  {
    mode: 'neural', label: 'Neural (Transformer + RL)',
    brief: 'Transformer 自回归采样 + REINFORCE 策略梯度',
    detail: '因果 Transformer 以 RPN 序列采样公式, rank-IC 作为 reward, 纯 neural 搜索。',
    strategies: ['neural'],
    params: ['gens', 'topK', 'nSplits', 'neuralBatch'],
  },
  {
    mode: 'mcts', label: 'MCTS (LLM-Guided Tree Search)',
    brief: '枚举种子 + LLM 进化 + MCTS 精炼',
    detail: 'Round 0 枚举种子, LLM 进化扩充 archive, MCTS 从精英出发树搜索精炼。',
    strategies: ['enumeration', 'llm_evolution', 'mcts'],
    params: ['popSize', 'offspring', 'gens', 'topK', 'nSplits', 'enumMax', 'enumTopK'],
  },
]

/* ── Param definition ──────────────────────────────────────────────── */

interface ParamDef {
  key: string
  label: string
  hint: string
  min?: number
  max?: number
  step?: number
}

const ALL_SEARCH_PARAMS: ParamDef[] = [
  { key: 'popSize',    label: 'Pop Size',        hint: '初始种群大小，种子+启发式填充', min: 2, max: 32 },
  { key: 'offspring',  label: 'Offspring/Round',  hint: '每轮 LLM 生成的变异公式数量', min: 1, max: 16 },
  { key: 'gens',       label: 'Generations',      hint: '进化迭代轮数，更多轮=更深搜索', min: 1, max: 20 },
  { key: 'topK',       label: 'Top-K',            hint: '最终保留的最优因子数量', min: 1, max: 20 },
  { key: 'nSplits',    label: 'CPCV Folds',       hint: '交叉验证折数，更多折=更稳健但更慢', min: 2, max: 10 },
  { key: 'enumMax',    label: 'Enum Total',       hint: 'Round 0 程序化枚举的公式总数', min: 50, max: 5000, step: 100 },
  { key: 'enumTopK',   label: 'Enum Top-K',       hint: '枚举后经 fast-IC 筛选保留的数量', min: 5, max: 200 },
  { key: 'neuralBatch', label: 'Neural Batch',    hint: '每步 Transformer 采样的 RPN 序列数', min: 64, max: 8192 },
]

/* ── Search presets ────────────────────────────────────────────────── */

export interface SearchPreset {
  key: string
  label: string
  hint: string
  strategy: string
  params: Record<string, number>
  budget?: { max_wall_time_sec?: number; max_full_eval?: number; max_llm_tokens?: number }
}

export const SEARCH_PRESETS: SearchPreset[] = [
  {
    key: 'quick', label: '快速扫描',
    hint: '~2 分钟出结果，小种群 + 1-2 轮；适合试水或因子草稿验证',
    strategy: 'evolution',
    params: { popSize: 4, offspring: 2, gens: 2, topK: 3, nSplits: 3, enumMax: 200, enumTopK: 10, neuralBatch: 2048 },
    budget: { max_wall_time_sec: 180, max_full_eval: 80 },
  },
  {
    key: 'balanced', label: '平衡探索 (推荐)',
    hint: '~10 分钟，中等种群 + 3 轮进化；大多数研究场景的默认选择',
    strategy: 'evolution',
    params: { popSize: 8, offspring: 4, gens: 3, topK: 5, nSplits: 5, enumMax: 500, enumTopK: 30, neuralBatch: 4096 },
    budget: { max_wall_time_sec: 900, max_full_eval: 400 },
  },
  {
    key: 'deep', label: '深度探索',
    hint: '~30 分钟以上，大种群 + 8 轮 + 更严 CPCV；用于最终选因子',
    strategy: 'evolution',
    params: { popSize: 16, offspring: 8, gens: 8, topK: 10, nSplits: 7, enumMax: 2000, enumTopK: 60, neuralBatch: 4096 },
    budget: { max_wall_time_sec: 2400, max_full_eval: 1500 },
  },
  {
    key: 'mcts', label: 'MCTS 精炼',
    hint: '枚举+LLM+MCTS 三段式，适合有明确前辈因子想精修时使用',
    strategy: 'mcts',
    params: { popSize: 8, offspring: 4, gens: 4, topK: 5, nSplits: 5, enumMax: 500, enumTopK: 30, neuralBatch: 4096 },
    budget: { max_wall_time_sec: 1200 },
  },
  {
    key: 'neural', label: 'Neural 大批量',
    hint: 'Transformer + REINFORCE 纯神经网络搜索，适合数据量大的市场',
    strategy: 'neural',
    params: { popSize: 4, offspring: 4, gens: 6, topK: 8, nSplits: 5, enumMax: 0, enumTopK: 0, neuralBatch: 8192 },
    budget: { max_wall_time_sec: 1800 },
  },
]

/* ── Labeled input with tooltip ────────────────────────────────────── */

function ParamInput({ def, value, onChange }: {
  def: ParamDef; value: number; onChange: (v: number) => void
}) {
  return (
    <div className="space-y-1.5">
      <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
        title={def.hint}>
        {def.label}
        <Info className="size-3 opacity-40" />
      </label>
      <Input type="number" min={def.min} max={def.max} step={def.step} value={value}
        onChange={e => onChange(+e.target.value || value)} />
    </div>
  )
}

/* ── Main component ────────────────────────────────────────────────── */

interface SearchTabProps {
  formula: string
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
  onLoadFormula: (f: string) => void
  searchJobs: SearchJobsState
  setErr: (e: string | null) => void
}

export const SearchTab: React.FC<SearchTabProps> = ({
  formula, interval, setInterval, symbols, setSymbols,
  startTime, setStartTime, endTime, setEndTime,
  intervals, symList, market, universe, setUniverse,
  excludeST, setExcludeST, ws, onLoadFormula, searchJobs, setErr,
}) => {
  const [searchSeeds, setSearchSeeds] = useState('')
  const [strategy, setStrategy] = useState<string>('evolution')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [presetKey, setPresetKey] = useState<string>('balanced')
  const activePreset = useMemo(
    () => SEARCH_PRESETS.find(p => p.key === presetKey) ?? SEARCH_PRESETS[1],
    [presetKey],
  )

  // Strategy modes loaded from backend (or fallback)
  const modes: StrategyModeInfo[] = ws?.strategy_modes_info?.length
    ? ws.strategy_modes_info
    : FALLBACK_MODES
  const info = modes.find(m => m.mode === strategy) ?? modes[0]
  const visibleParams = useMemo(() => {
    const allowed = new Set(info.params)
    return ALL_SEARCH_PARAMS.filter(p => allowed.has(p.key))
  }, [info.params])

  // Search params with defaults
  const [params, setParams] = useState<Record<string, number>>({
    popSize: 6, offspring: 3, gens: 3, topK: 5, nSplits: 5,
    enumMax: 500, enumTopK: 30, neuralBatch: 4096,
  })
  const setParam = useCallback((key: string, value: number) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }, [])

  const applyPreset = useCallback((key: string) => {
    setPresetKey(key)
    const preset = SEARCH_PRESETS.find(p => p.key === key)
    if (!preset) return
    setStrategy(preset.strategy)
    setParams(prev => ({ ...prev, ...preset.params }))
  }, [])

  // Estimated search volume based on active strategies
  const estimatedVolume = useMemo(() => {
    const strats = new Set(info.strategies)
    const enumPart = strats.has('enumeration') ? params.enumTopK : 0
    const llmPart = strats.has('llm_evolution') ? params.gens * params.offspring : 0
    const neuralPart = strats.has('neural_formula') ? params.gens * 30 : 0
    return enumPart + llmPart + neuralPart
  }, [info.strategies, params])

  const handleSearch = useCallback(async () => {
    try {
      setErr(null)
      const seeds = searchSeeds.split('\n').map(s => s.trim()).filter(Boolean)
      if (formula.trim() && !seeds.includes(formula.trim())) seeds.unshift(formula.trim())
      await searchJobs.submit({
        market, symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime), interval, seeds,
        population_size: params.popSize, offspring_count: params.offspring, generations: params.gens,
        top_k: params.topK, n_splits: params.nSplits, persist: true, strategy,
        neural_batch: params.neuralBatch, enum_max: params.enumMax, enum_top_k: params.enumTopK,
        ...(activePreset.budget?.max_wall_time_sec ? { max_wall_time_sec: activePreset.budget.max_wall_time_sec } : {}),
        ...(activePreset.budget?.max_full_eval ? { max_full_eval: activePreset.budget.max_full_eval } : {}),
        ...(activePreset.budget?.max_llm_tokens ? { max_llm_tokens: activePreset.budget.max_llm_tokens } : {}),
        ...(universe ? { universe } : {}),
        ...(excludeST ? { exclude_st: true } : {}),
      })
    } catch (e) { setErr(e instanceof Error ? e.message : 'Search submit failed') }
  }, [formula, interval, startTime, endTime, searchSeeds, params, symList, strategy, market, universe, excludeST, setErr, searchJobs, activePreset])

  return (
    <div className="space-y-6">
      <SectionCard title="Alpha Search" description="Configure and run automated factor discovery.">

        {/* ── Preset selector ── */}
        <div className="space-y-2">
          <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
            Preset
          </label>
          <div className="flex flex-wrap gap-2">
            {SEARCH_PRESETS.map(p => (
              <button
                key={p.key}
                type="button"
                onClick={() => applyPreset(p.key)}
                className={
                  'rounded-lg border px-3 py-1.5 text-xs transition ' +
                  (p.key === presetKey
                    ? 'border-primary bg-primary/10 text-foreground'
                    : 'border-border/50 bg-secondary/30 text-muted-foreground hover:text-foreground')
                }
                title={p.hint}
              >
                {p.label}
              </button>
            ))}
          </div>
          <div className="text-[11px] text-muted-foreground">{activePreset.hint}</div>
        </div>

        {/* ── Strategy selector + description ── */}
        <div className="space-y-3">
          <div className="grid gap-4 md:grid-cols-2">
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Strategy</label>
              <select value={strategy} onChange={e => setStrategy(e.target.value)}
                className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
                {modes.map(m => (
                  <option key={m.mode} value={m.mode}>{m.label}</option>
                ))}
              </select>
            </div>
            <div className="flex items-end">
              <div className="rounded-xl border border-border/50 bg-secondary/20 px-3 py-2 text-xs text-muted-foreground w-full">
                {info.brief}
              </div>
            </div>
          </div>

          {/* Collapsible strategy detail */}
          <details className="group rounded-xl border border-border/40 bg-card/50">
            <summary className="flex cursor-pointer items-center gap-2 px-4 py-2.5 text-xs font-medium text-muted-foreground hover:text-foreground transition select-none">
              <ChevronRight className="size-3.5 transition-transform group-open:rotate-90" />
              Algorithm Details
              <span className="ml-auto flex gap-1">
                {info.strategies.map(s => (
                  <span key={s} className="rounded-full bg-secondary/60 px-1.5 py-0.5 text-[10px]">{s.replace('_', ' ')}</span>
                ))}
              </span>
            </summary>
            <div className="border-t border-border/30 px-4 py-3 text-xs text-muted-foreground/90 whitespace-pre-line leading-relaxed">
              {info.detail}
            </div>
          </details>
        </div>

        {/* ── Data scope (shared component) ── */}
        <DataScopeSection
          interval={interval} setInterval={setInterval}
          symbols={symbols} setSymbols={setSymbols}
          startTime={startTime} setStartTime={setStartTime}
          endTime={endTime} setEndTime={setEndTime}
          intervals={intervals} symList={symList}
          market={market} universe={universe} setUniverse={setUniverse}
          excludeST={excludeST} setExcludeST={setExcludeST}
          ws={ws}
        />

        {/* ── Search params (collapsible) ── */}
        <details className="group" open={showAdvanced} onToggle={e => setShowAdvanced((e.target as HTMLDetailsElement).open)}>
          <summary className="flex cursor-pointer items-center gap-2 text-xs font-medium text-muted-foreground hover:text-foreground transition select-none">
            <Settings2 className="size-3.5" />
            <ChevronRight className="size-3 transition-transform group-open:rotate-90" />
            Search Parameters
            <span className="ml-2 rounded-full bg-secondary/60 px-2 py-0.5 text-[10px]">
              ~{estimatedVolume} formulas
            </span>
          </summary>
          <div className="mt-3 space-y-3">
            <div className="grid gap-4 grid-cols-2 md:grid-cols-4 xl:grid-cols-4">
              {visibleParams.map(p => (
                <ParamInput key={p.key} def={p} value={params[p.key]} onChange={v => setParam(p.key, v)} />
              ))}
            </div>
            <details className="group/hints rounded-lg border border-border/30 bg-card/30">
              <summary className="flex cursor-pointer items-center gap-2 px-3 py-2 text-[11px] text-muted-foreground hover:text-foreground transition select-none">
                <Info className="size-3" />
                <ChevronRight className="size-3 transition-transform group-open/hints:rotate-90" />
                Parameter Reference
              </summary>
              <div className="border-t border-border/20 px-3 py-2">
                <table className="w-full text-[11px]">
                  <tbody>
                    {visibleParams.map(p => (
                      <tr key={p.key} className="border-b border-border/10 last:border-0">
                        <td className="py-1 pr-3 font-medium text-foreground/80 whitespace-nowrap">{p.label}</td>
                        <td className="py-1 text-muted-foreground">{p.hint}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          </div>
        </details>

        {/* ── Seeds ── */}
        <div className="space-y-1.5">
          <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
            title="额外的种子公式，当前 Research 公式会自动加入">
            Extra Seeds (one per line)<Info className="size-3 opacity-40" />
          </label>
          <textarea value={searchSeeds} onChange={e => setSearchSeeds(e.target.value)} rows={2}
            className="w-full rounded-xl border border-border bg-input px-4 py-2 text-xs font-mono text-foreground outline-none" />
        </div>

        {/* ── Submit ── */}
        <div className="flex items-center gap-4">
          <Button onClick={() => void handleSearch()} disabled={!formula.trim()}>
            <Zap />Start Search
          </Button>
          <span className="text-xs text-muted-foreground">
            ~{estimatedVolume} formulas · {strategy}
          </span>
        </div>
      </SectionCard>

      {/* ── Jobs dashboard (compact summary) ── */}
      <JobsDashboard searchJobs={searchJobs} />

      {/* ── Job list ── */}
      {searchJobs.jobs.length > 0 && (
        <div className="space-y-4">
          {searchJobs.jobs.map(job => (
            <JobCard
              key={job.job_id}
              job={job}
              onLoadFormula={onLoadFormula}
              onApplySeeds={(seeds) => setSearchSeeds(prev => {
                const existing = prev.split('\n').map(s => s.trim()).filter(Boolean)
                const merged = Array.from(new Set([...existing, ...seeds]))
                return merged.join('\n')
              })}
              onCancel={() => searchJobs.cancel(job.job_id)}
              onHide={() => searchJobs.hide(job.job_id)}
              onRetry={() => searchJobs.retry(job.job_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Job card: wraps SearchProgress + cancel/retry/hide controls ─── */

function JobCard({ job, onLoadFormula, onApplySeeds, onCancel, onHide, onRetry }: {
  job: AlphaLabSearchJob
  onLoadFormula: (f: string) => void
  onApplySeeds: (seeds: string[]) => void
  onCancel: () => Promise<void>
  onHide: () => void
  onRetry: () => Promise<string | null>
}) {
  const isActive = job.status === 'pending' || job.status === 'running'
  const isCancelling = job.status === 'cancelling'
  const isCompleted = job.status === 'completed'
  const isFailed = job.status === 'failed'
  const isCancelled = job.status === 'cancelled'
  const [showAnalysis, setShowAnalysis] = useState(false)
  const [confirmingCancel, setConfirmingCancel] = useState(false)
  const [retrying, setRetrying] = useState(false)

  const handleCancel = async () => {
    setConfirmingCancel(false)
    await onCancel()
  }

  const handleRetry = async () => {
    setRetrying(true)
    try { await onRetry() } finally { setRetrying(false) }
  }

  return (
    <div className="relative">
      {/* Right-rail controls */}
      <div className="absolute right-3 top-3 z-10 flex items-center gap-1">
        {isActive && !confirmingCancel && (
          <button
            onClick={() => setConfirmingCancel(true)}
            className="rounded-lg px-2 py-1 text-[11px] text-red-400 hover:text-red-300 hover:bg-red-500/10 transition flex items-center gap-1"
            title="Cancel this running search"
          >
            <StopCircle className="size-3.5" />
            Cancel
          </button>
        )}
        {isActive && confirmingCancel && (
          <div className="flex items-center gap-1 rounded-lg bg-red-500/10 border border-red-500/30 px-2 py-1 text-[11px]">
            <span className="text-red-300">Stop this search?</span>
            <button
              onClick={() => void handleCancel()}
              className="rounded px-1.5 py-0.5 text-red-200 hover:bg-red-500/20 transition font-medium"
            >
              Yes
            </button>
            <button
              onClick={() => setConfirmingCancel(false)}
              className="rounded px-1.5 py-0.5 text-muted-foreground hover:bg-secondary/60 transition"
            >
              No
            </button>
          </div>
        )}
        {isCancelling && (
          <span className="text-[11px] text-amber-400 flex items-center gap-1">
            <Loader2 className="size-3 animate-spin" />
            cancelling…
          </span>
        )}
        {isFailed && (
          <button
            onClick={() => void handleRetry()}
            disabled={retrying}
            className="rounded-lg px-2 py-1 text-[11px] text-blue-300 hover:text-blue-200 hover:bg-blue-500/10 transition flex items-center gap-1 disabled:opacity-50"
            title="Re-submit using the original parameters"
          >
            <RotateCcw className={`size-3.5 ${retrying ? 'animate-spin' : ''}`} />
            Retry
          </button>
        )}
        {!isActive && (
          <button
            onClick={onHide}
            className="rounded-lg p-1 text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition"
            title="Hide from the list (does not cancel)"
          >
            <X className="size-3.5" />
          </button>
        )}
      </div>

      {/* Job header badge */}
      <div className="mb-1 flex items-center gap-2 text-[10px] text-muted-foreground">
        {(isActive || isCancelling) && <Loader2 className="size-3 animate-spin text-blue-400" />}
        <span className="font-mono">{job.job_id}</span>
        {job.strategy && <Badge variant="info" className="text-[9px] px-1.5 py-0">{job.strategy}</Badge>}
        {job.request_id && (
          <a
            href={`#/tracing/${job.request_id}`}
            className="font-mono text-[9px] text-muted-foreground/60 hover:text-foreground transition"
            title="Open trace for this request"
          >
            trace:{job.request_id.slice(0, 8)}
          </a>
        )}
        {isCancelled && <span className="text-amber-400">cancelled</span>}
        {job.created_at && <span>{new Date(job.created_at).toLocaleTimeString()}</span>}
      </div>

      <SearchProgress searchJob={job} onLoadFormula={onLoadFormula} />

      {isCompleted && job.job_id && (
        <>
          {!showAnalysis && (
            <div className="mt-2">
              <Button variant="outline" size="sm" onClick={() => setShowAnalysis(true)}>
                LLM Analysis
              </Button>
            </div>
          )}
          {showAnalysis && <LLMAnalysis jobId={job.job_id} onApplySeeds={onApplySeeds} />}
        </>
      )}
    </div>
  )
}
