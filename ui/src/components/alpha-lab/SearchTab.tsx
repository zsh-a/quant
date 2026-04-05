/**
 * Search tab — configure and execute automated alpha search.
 */
import React, { useCallback, useEffect, useRef, useState } from 'react'
import { Loader2, Zap } from 'lucide-react'
import type { AlphaLabSearchJob, AlphaLabWorkspace as WorkspacePayload } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { SearchProgress } from './SearchProgress'
import { LLMAnalysis } from './LLMAnalysis'
import { toISO, SEARCH_POLL_MS } from './shared'
import { alphaApi } from '../../utils/alphaApi'

interface SearchTabProps {
  formula: string
  interval: string
  symbols: string
  startTime: string
  endTime: string
  symList: () => string[]
  ws: WorkspacePayload | null
  onLoadFormula: (f: string) => void
  onSearchComplete: () => void
  setErr: (e: string | null) => void
}

export const SearchTab: React.FC<SearchTabProps> = ({
  formula, interval, symbols, startTime, endTime, symList, ws,
  onLoadFormula, onSearchComplete, setErr,
}) => {
  const [searchJob, setSearchJob] = useState<AlphaLabSearchJob | null>(null)
  const [searchSeeds, setSearchSeeds] = useState('')
  const [popSize, setPopSize] = useState(6)
  const [offspring, setOffspring] = useState(3)
  const [gens, setGens] = useState(3)
  const [topK, setTopK] = useState(5)
  const [nSplits, setNSplits] = useState(5)
  const [strategy, setStrategy] = useState<string>('evolution')
  const [neuralBatch, setNeuralBatch] = useState(4096)
  const pollRef = useRef<ReturnType<typeof globalThis.setInterval> | null>(null)

  const isActive = searchJob?.status === 'pending' || searchJob?.status === 'running'

  const handleSearch = useCallback(async () => {
    try {
      setErr(null)
      const seeds = searchSeeds.split('\n').map(s => s.trim()).filter(Boolean)
      if (formula.trim() && !seeds.includes(formula.trim())) seeds.unshift(formula.trim())
      const r = await alphaApi.submitSearch({
        symbols: symList(), start_time: toISO(startTime), end_time: toISO(endTime), interval, seeds,
        population_size: popSize, offspring_count: offspring, generations: gens,
        top_k: topK, n_splits: nSplits, persist: true, strategy, neural_batch: neuralBatch,
      })
      setSearchJob({ job_id: r.job_id, status: 'pending' })
    } catch (e) { setErr(e instanceof Error ? e.message : 'Search submit failed') }
  }, [formula, interval, startTime, endTime, searchSeeds, popSize, offspring, gens, topK, nSplits, symList, strategy, neuralBatch, setErr])

  // Poll search job status
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
      } catch { /* ignore poll errors */ }
    }
    void poll()
    pollRef.current = globalThis.setInterval(poll, SEARCH_POLL_MS)
    return () => { if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null } }
  }, [searchJob?.job_id, searchJob?.status, onSearchComplete])

  return (
    <div className="space-y-6">
      <SectionCard title="Alpha Search" description="Configure and run automated factor discovery.">
        <div className="grid gap-4 grid-cols-2 md:grid-cols-3">
          <div className="space-y-1.5">
            <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Strategy</label>
            <select value={strategy} onChange={e => setStrategy(e.target.value)}
              className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
              {(ws?.strategy_modes ?? ['evolution', 'neural', 'full']).map(m => (
                <option key={m} value={m}>{m === 'evolution' ? 'Evolution (LLM+Enum)' : m === 'neural' ? 'Neural (Transformer+RL)' : 'Full (All Strategies)'}</option>
              ))}
            </select>
          </div>
          {(strategy === 'neural' || strategy === 'full') && (
            <div className="space-y-1.5">
              <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Neural Batch</label>
              <Input type="number" min={64} max={8192} value={neuralBatch} onChange={e => setNeuralBatch(+e.target.value || 4096)} />
            </div>
          )}
        </div>
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
        <Button onClick={() => void handleSearch()} disabled={isActive || !formula.trim()}>
          {isActive ? <Loader2 className="animate-spin" /> : <Zap />}{isActive ? 'Running...' : 'Start Search'}</Button>
      </SectionCard>

      {searchJob && <SearchProgress searchJob={searchJob} onLoadFormula={onLoadFormula} />}

      {searchJob?.status === 'completed' && searchJob.job_id && (
        <LLMAnalysis jobId={searchJob.job_id} onApplySeeds={(seeds) => setSearchSeeds(seeds.join('\n'))} />
      )}
    </div>
  )
}
