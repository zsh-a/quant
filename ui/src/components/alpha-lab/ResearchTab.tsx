/**
 * Research tab — formula editing, validation, and single-formula analysis.
 */
import React, { useCallback, useState } from 'react'
import { Activity, AlertTriangle, CheckCircle2, Loader2, Save } from 'lucide-react'
import type { AlphaLabEvaluationSummary, AlphaLabValidationReport } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { MiniChart, MetricGrid } from './MiniChart'
import { fmt, toISO } from './shared'
import { alphaApi } from '../../utils/alphaApi'

const METRIC_KEYS = ['sharpe', 'rank_ic', 'ic_ir', 'total_return', 'max_drawdown', 'avg_turnover', 'signal_coverage', 'pnl_per_turnover'] as const
const IC_DETAIL_KEYS = ['rank_ic_1d', 'rank_ic_5d', 'rank_ic_10d', 'ic_decay', 'ic_std', 'turnover_proxy'] as const

interface ResearchTabProps {
  formula: string
  setFormula: (f: string) => void
  interval: string
  setInterval: (v: string) => void
  symbols: string
  setSymbols: (v: string) => void
  startTime: string
  setStartTime: (v: string) => void
  endTime: string
  setEndTime: (v: string) => void
  intervals: string[]
  samples: string[]
  symList: () => string[]
  onSaved: () => void
  setErr: (e: string | null) => void
}

export const ResearchTab: React.FC<ResearchTabProps> = ({
  formula, setFormula, interval, setInterval, symbols, setSymbols,
  startTime, setStartTime, endTime, setEndTime,
  intervals, samples, symList, onSaved, setErr,
}) => {
  const [validation, setValidation] = useState<AlphaLabValidationReport | null>(null)
  const [analysis, setAnalysis] = useState<AlphaLabEvaluationSummary | null>(null)
  const [validating, setValidating] = useState(false)
  const [analyzing, setAnalyzing] = useState(false)
  const [saving, setSaving] = useState(false)

  const handleValidate = useCallback(async () => {
    try {
      setValidating(true); setErr(null)
      const r = await alphaApi.validate(formula)
      setValidation(r)
    } catch (e) { setValidation(null); setErr(e instanceof Error ? e.message : 'Validation failed') }
    finally { setValidating(false) }
  }, [formula, setErr])

  const handleAnalyze = useCallback(async () => {
    try {
      setAnalyzing(true); setErr(null)
      const r = await alphaApi.evaluateDb({
        formula, interval, symbols: symList(),
        start_time: toISO(startTime), end_time: toISO(endTime), summary_only: true,
      })
      setAnalysis(r)
      if (r.normalized_formula) setValidation({ ok: true, normalized_formula: r.normalized_formula, errors: [], warnings: [] })
    } catch (e) { setAnalysis(null); setErr(e instanceof Error ? e.message : 'Analysis failed') }
    finally { setAnalyzing(false) }
  }, [formula, interval, symList, startTime, endTime, setErr])

  const handleSave = useCallback(async () => {
    try {
      setSaving(true); setErr(null)
      await alphaApi.saveToZoo({ formula, fitness: analysis?.metrics?.sharpe, metrics: analysis?.metrics ?? {}, lineage: {}, source: 'frontend' })
      onSaved()
    } catch (e) { setErr(e instanceof Error ? e.message : 'Save failed') }
    finally { setSaving(false) }
  }, [formula, analysis, onSaved, setErr])

  return (
    <div className="space-y-6">
      <SectionCard title="Formula">
        <textarea value={formula} onChange={e => setFormula(e.target.value)} placeholder="cs_rank(ts_mean(close, 5) - close)"
          className="min-h-28 w-full rounded-xl border border-border bg-input px-4 py-3 text-sm font-mono text-foreground outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30" />
        {samples.length > 0 && (
          <div className="flex flex-wrap gap-2">{samples.map(s => (
            <button key={s} type="button" onClick={() => { setFormula(s); setValidation(null); setAnalysis(null) }}
              className="rounded-full border border-border/80 bg-secondary/60 px-3 py-1 text-xs font-mono text-muted-foreground transition hover:text-foreground">{s}</button>
          ))}</div>
        )}
        <div className="grid gap-4 md:grid-cols-4">
          <div className="space-y-1.5">
            <label className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">Interval</label>
            <select value={interval} onChange={e => setInterval(e.target.value)}
              className="h-10 w-full rounded-xl border border-border bg-input px-3 text-sm text-foreground outline-none">
              {intervals.map(i => <option key={i} value={i}>{i}</option>)}
            </select>
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
              {analysis.dataset?.shape && <Badge>{analysis.dataset.shape[0]} x {analysis.dataset.shape[1]}</Badge>}
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
    </div>
  )
}
