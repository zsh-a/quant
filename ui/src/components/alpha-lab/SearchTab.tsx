/**
 * Search tab — configure and execute automated alpha search.
 */
import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { ChevronDown, ChevronRight, Info, Loader2, Settings2, Zap } from 'lucide-react'
import type { AlphaLabSearchJob, AlphaLabWorkspace as WorkspacePayload, StrategyModeInfo } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { SearchProgress } from './SearchProgress'
import { LLMAnalysis } from './LLMAnalysis'
import { toISO, SEARCH_POLL_MS } from './shared'
import { alphaApi } from '../../utils/alphaApi'

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
    brief: 'Transformer 自回归生成 + REINFORCE 训练',
    detail: 'Causal Transformer + RPN 序列生成 + REINFORCE 强化学习。',
    strategies: ['neural_formula'],
    params: ['gens', 'topK', 'nSplits', 'neuralBatch'],
  },
  {
    mode: 'full', label: 'Full (All Strategies)',
    brief: '枚举 + LLM 进化 + MCTS 精炼 + Neural 生成',
    detail: '组合所有搜索策略，多策略协作。',
    strategies: ['enumeration', 'llm_evolution', 'mcts_refinement', 'neural_formula'],
    params: ['popSize', 'offspring', 'gens', 'topK', 'nSplits', 'enumMax', 'enumTopK', 'neuralBatch'],
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
  show?: (strategy: string) => boolean
}

const DATA_PARAMS: ParamDef[] = [
  { key: 'interval',  label: 'Interval',  hint: 'K 线周期，决定数据粒度和单 bar 时长' },
  { key: 'startTime', label: 'Start',     hint: '回测数据起始时间' },
  { key: 'endTime',   label: 'End',       hint: '回测数据结束时间' },
  { key: 'symbols',   label: 'Symbols',   hint: '搜索覆盖的交易对，逗号分隔' },
]

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
  ws: WorkspacePayload | null
  onLoadFormula: (f: string) => void
  onSearchComplete: () => void
  setErr: (e: string | null) => void
}

export const SearchTab: React.FC<SearchTabProps> = ({
  formula, interval, setInterval, symbols, setSymbols,
  startTime, setStartTime, endTime, setEndTime,
  intervals, symList, ws, onLoadFormula, onSearchComplete, setErr,
}) => {
  const [searchJob, setSearchJob] = useState<AlphaLabSearchJob | null>(null)
  const [searchSeeds, setSearchSeeds] = useState('')
  const [strategy, setStrategy] = useState<string>('evolution')
  const [showAdvanced, setShowAdvanced] = useState(false)
  const pollRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null)

  // Strategy modes loaded from backend (or fallback)
  const modes: StrategyModeInfo[] = ws?.strategy_modes_info?.length
    ? ws.strategy_modes_info
    : FALLBACK_MODES
  const info = modes.find(m => m.mode === strategy) ?? modes[0]
  // Visible search params: only those listed in the current mode's params
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

  const isActive = searchJob?.status === 'pending' || searchJob?.status === 'running'

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
      const r = await alphaApi.submitSearch({
        symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime), interval, seeds,
        population_size: params.popSize, offspring_count: params.offspring, generations: params.gens,
        top_k: params.topK, n_splits: params.nSplits, persist: true, strategy,
        neural_batch: params.neuralBatch, enum_max: params.enumMax, enum_top_k: params.enumTopK,
      })
      setSearchJob({ job_id: r.job_id, status: 'pending' })
    } catch (e) { setErr(e instanceof Error ? e.message : 'Search submit failed') }
  }, [formula, interval, startTime, endTime, searchSeeds, params, symList, strategy, setErr])

  useEffect(() => {
    if (!searchJob || searchJob.status === 'completed' || searchJob.status === 'failed') {
      if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null }
      return
    }
    const poll = async () => {
      try {
        const r = await alphaApi.getSearchJob(searchJob.job_id)
        setSearchJob(r)
        if (r.status === 'completed' || r.status === 'failed') {
          if (r.status === 'completed') onSearchComplete()
        }
      } catch { /* ignore */ }
    }
    void poll()
    pollRef.current = globalThis.setInterval(poll, SEARCH_POLL_MS)
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null } }
  }, [searchJob?.job_id, searchJob?.status, onSearchComplete])

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

        {/* ── Data scope ── */}
        <div className="grid gap-4 md:grid-cols-4">
          {DATA_PARAMS.map(p => (
            <div key={p.key} className="space-y-1.5">
              <label className="flex items-center gap-1 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground"
                title={p.hint}>
                {p.label}<Info className="size-3 opacity-40" />
              </label>
              {p.key === 'interval' ? (
                <select value={interval} onChange={e => setInterval(e.target.value)}
                  className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
                  {intervals.map(i => <option key={i} value={i}>{i}</option>)}
                </select>
              ) : p.key === 'startTime' ? (
                <Input type="datetime-local" value={startTime} onChange={e => setStartTime(e.target.value)} />
              ) : p.key === 'endTime' ? (
                <Input type="datetime-local" value={endTime} onChange={e => setEndTime(e.target.value)} />
              ) : (
                <Input value={symbols} onChange={e => setSymbols(e.target.value)} placeholder="BTCUSDT,ETHUSDT" />
              )}
            </div>
          ))}
        </div>

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
            {/* Param hints table */}
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
          <Button onClick={() => void handleSearch()} disabled={isActive || !formula.trim()}>
            {isActive ? <Loader2 className="animate-spin" /> : <Zap />}{isActive ? 'Running...' : 'Start Search'}
          </Button>
          <span className="text-xs text-muted-foreground">
            ~{estimatedVolume} formulas will enter full CPCV evaluation
          </span>
        </div>
      </SectionCard>

      {searchJob && <SearchProgress searchJob={searchJob} onLoadFormula={onLoadFormula} />}

      {searchJob?.status === 'completed' && searchJob.job_id && (
        <LLMAnalysis jobId={searchJob.job_id} onApplySeeds={(seeds) => setSearchSeeds(seeds.join('\n'))} />
      )}
    </div>
  )
}
