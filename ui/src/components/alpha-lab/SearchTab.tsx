/**
 * Search tab — configure and execute automated alpha search.
 *
 * Supports multiple concurrent search jobs. Job state is managed by
 * useSearchJobs (in workspace) and survives page refresh via backend recovery.
 */
import React, { useCallback, useMemo, useState } from 'react'
import { ChevronRight, Info, Loader2, Settings2, X, Zap } from 'lucide-react'
import type { AlphaLabSearchJob, AlphaLabWorkspace as WorkspacePayload, StrategyModeInfo } from '../../types'
import type { SearchJobsState } from '../../hooks/useSearchJobs'
import { SectionCard } from '../layout/SectionCard'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { Badge } from '../ui/badge'
import { SearchProgress } from './SearchProgress'
import { LLMAnalysis } from './LLMAnalysis'
import { DataScopeSection } from './DataScopeSection'
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
        ...(universe ? { universe } : {}),
        ...(excludeST ? { exclude_st: true } : {}),
      })
    } catch (e) { setErr(e instanceof Error ? e.message : 'Search submit failed') }
  }, [formula, interval, startTime, endTime, searchSeeds, params, symList, strategy, market, universe, excludeST, setErr, searchJobs])

  return (
    <div className="space-y-6">
      <SectionCard title="Alpha Search" description="Configure and run automated factor discovery.">

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

      {/* ── Job list ── */}
      {searchJobs.jobs.length > 0 && (
        <div className="space-y-4">
          {searchJobs.jobs.map(job => (
            <JobCard
              key={job.job_id}
              job={job}
              onLoadFormula={onLoadFormula}
              onDismiss={() => searchJobs.dismiss(job.job_id)}
              onRefresh={() => searchJobs.refresh(job.job_id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}

/* ── Job card: wraps SearchProgress + dismiss button ────────────────── */

function JobCard({ job, onLoadFormula, onDismiss, onRefresh }: {
  job: AlphaLabSearchJob
  onLoadFormula: (f: string) => void
  onDismiss: () => void
  onRefresh: () => void
}) {
  const isActive = job.status === 'pending' || job.status === 'running'
  const isCompleted = job.status === 'completed'
  const [showAnalysis, setShowAnalysis] = useState(false)

  return (
    <div className="relative">
      {/* Dismiss button */}
      {!isActive && (
        <button
          onClick={onDismiss}
          className="absolute right-3 top-3 z-10 rounded-lg p-1 text-muted-foreground hover:text-foreground hover:bg-secondary/60 transition"
          title="Dismiss"
        >
          <X className="size-3.5" />
        </button>
      )}

      {/* Job header badge */}
      <div className="mb-1 flex items-center gap-2 text-[10px] text-muted-foreground">
        {isActive && <Loader2 className="size-3 animate-spin text-blue-400" />}
        <span className="font-mono">{job.job_id}</span>
        {job.strategy && <Badge variant="info" className="text-[9px] px-1.5 py-0">{job.strategy}</Badge>}
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
          {showAnalysis && <LLMAnalysis jobId={job.job_id} onApplySeeds={() => {}} />}
        </>
      )}
    </div>
  )
}
