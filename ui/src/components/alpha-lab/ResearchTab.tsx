/**
 * Research tab — formula editing, validation, and single-formula analysis.
 */
import React, { useCallback, useState } from 'react'
import { Activity, AlertTriangle, CheckCircle2, Loader2, Save } from 'lucide-react'
import type { AlphaLabEvaluationSummary, AlphaLabValidationReport, AlphaLabWorkspace as WorkspacePayload } from '../../types'
import { SectionCard } from '../layout/SectionCard'
import { EmptyState } from '../layout/EmptyState'
import { Badge } from '../ui/badge'
import { Button } from '../ui/button'
import { MiniChart, MetricGrid } from './MiniChart'
import { QuantileChart } from './QuantileChart'
import { toISO } from './shared'
import { DataScopeSection } from './DataScopeSection'
import { alphaApi } from '../../utils/alphaApi'

const EVAL_METHOD_LABELS: Record<string, string> = {
  long_short: '多空对冲',
  long_only: '纯多头 Top-K',
  quantile: '分层回测',
}

const METRIC_KEYS = ['sharpe', 'rank_ic', 'ic_ir', 'calmar', 'total_return', 'max_drawdown', 'win_rate', 'avg_turnover', 'signal_coverage', 'pnl_per_turnover'] as const
const IC_DETAIL_KEYS = ['rank_ic_1d', 'rank_ic_5d', 'rank_ic_10d', 'ic_decay', 'ic_std', 'turnover_proxy', 'skewness'] as const

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
  market: string
  universe: string | null
  setUniverse: (v: string | null) => void
  excludeST: boolean
  setExcludeST: (v: boolean) => void
  ws: WorkspacePayload | null
  onSaved: () => void
  setErr: (e: string | null) => void
}

export const ResearchTab: React.FC<ResearchTabProps> = ({
  formula, setFormula, interval, setInterval, symbols, setSymbols,
  startTime, setStartTime, endTime, setEndTime,
  intervals, samples, symList, market, universe, setUniverse,
  excludeST, setExcludeST, ws, onSaved, setErr,
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
        formula, interval, symbols: symList(), market,
        start_time: toISO(startTime), end_time: toISO(endTime), summary_only: true,
        ...(universe ? { universe } : {}),
        ...(excludeST ? { exclude_st: true } : {}),
      })
      setAnalysis(r)
      if (r.normalized_formula) setValidation({ ok: true, normalized_formula: r.normalized_formula, errors: [], warnings: [] })
    } catch (e) { setAnalysis(null); setErr(e instanceof Error ? e.message : 'Analysis failed') }
    finally { setAnalyzing(false) }
  }, [formula, interval, symList, startTime, endTime, market, universe, excludeST, setErr])

  const handleSave = useCallback(async () => {
    try {
      setSaving(true); setErr(null)
      await alphaApi.saveToZoo({ formula, fitness: analysis?.metrics?.sharpe, metrics: analysis?.metrics ?? {}, lineage: {}, source: 'frontend' })
      onSaved()
    } catch (e) { setErr(e instanceof Error ? e.message : 'Save failed') }
    finally { setSaving(false) }
  }, [formula, analysis, onSaved, setErr])

  return (
    <div className="space-y-4">
      {/* ── Formula + Data scope: side-by-side on wide screens ── */}
      <div className="grid gap-4 xl:grid-cols-[1fr_auto]">
        {/* Left: formula input */}
        <SectionCard title="Formula" contentClassName="space-y-3">
          <textarea value={formula} onChange={e => setFormula(e.target.value)} placeholder="cs_rank(ts_mean(close, 5) - close)"
            className="min-h-20 w-full rounded-lg border border-border bg-input px-3 py-2 text-sm font-mono text-foreground outline-none transition focus:border-ring/60 focus:ring-2 focus:ring-ring/30 resize-y" />
          {samples.length > 0 && (
            <div className="flex flex-wrap gap-1.5">{samples.map(s => (
              <button key={s} type="button" onClick={() => { setFormula(s); setValidation(null); setAnalysis(null) }}
                className="rounded-full border border-border/60 bg-secondary/40 px-2.5 py-0.5 text-[11px] font-mono text-muted-foreground transition hover:text-foreground">{s}</button>
            ))}</div>
          )}
          <div className="flex flex-wrap items-center gap-2">
            <Button size="sm" variant="outline" onClick={() => void handleValidate()} disabled={validating || !formula.trim()}>
              {validating ? <Loader2 className="animate-spin" /> : <CheckCircle2 />}Validate</Button>
            <Button size="sm" onClick={() => void handleAnalyze()} disabled={analyzing || !formula.trim()}>
              {analyzing ? <Loader2 className="animate-spin" /> : <Activity />}Analyze</Button>
            <Button size="sm" variant="default" onClick={() => void handleSave()} disabled={saving || !formula.trim()}>
              {saving ? <Loader2 className="animate-spin" /> : <Save />}Save to Zoo</Button>
          </div>
          {validation && (
            <div className={`rounded-lg border px-3 py-2 text-xs ${validation.ok ? 'border-emerald-500/20 bg-emerald-500/10 text-emerald-300' : 'border-amber-500/20 bg-amber-500/10 text-amber-300'}`}>
              <span className="inline-flex items-center gap-1 font-semibold">{validation.ok ? <CheckCircle2 className="size-3" /> : <AlertTriangle className="size-3" />}{validation.ok ? 'Valid' : 'Invalid'}</span>
              {validation.normalized_formula && <span className="ml-2 break-all font-mono opacity-80">{validation.normalized_formula}</span>}
              {validation.errors?.map(e => <p key={e} className="mt-1 text-rose-300">{e}</p>)}
            </div>
          )}
        </SectionCard>

        {/* Right: data scope (narrower on wide screens) */}
        <SectionCard title="Data Scope" className="xl:w-80" contentClassName="space-y-3">
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
        </SectionCard>
      </div>

      {/* ── Analysis results ── */}
      {analysis ? (
        <div className="space-y-3">
          {/* Badges + Metrics */}
          <SectionCard title="Analysis"
            action={
              <div className="flex flex-wrap gap-1.5">
                {analysis.metrics.eval_method && (
                  <Badge variant="info" className="text-[10px]">{EVAL_METHOD_LABELS[analysis.metrics.eval_method as any] ?? analysis.metrics.eval_method}</Badge>
                )}
                {analysis.dataset?.shape && <Badge variant="default" className="text-[10px]">{analysis.dataset.shape[0]}×{analysis.dataset.shape[1]}</Badge>}
                {analysis.expr_hash && <Badge variant="default" className="font-mono text-[9px]">{analysis.expr_hash.slice(0, 10)}</Badge>}
              </div>
            }
            contentClassName="space-y-3"
          >
            <MetricGrid metrics={analysis.metrics} keys={METRIC_KEYS} />
            {(analysis.metrics.rank_ic_1d != null || analysis.metrics.ic_decay != null) && (
              <MetricGrid metrics={analysis.metrics} keys={IC_DETAIL_KEYS} />
            )}
          </SectionCard>

          {/* Charts: equity+drawdown side by side, turnover below */}
          <div className="grid gap-3 lg:grid-cols-2">
            <MiniChart data={analysis.equity_series ?? []} label="Equity Curve" height={240} />
            <MiniChart data={analysis.drawdown_series ?? []} label="Drawdown" color="#f43f5e" height={240} pct />
          </div>
          <MiniChart data={analysis.turnover_series ?? []} label="Turnover" color="#a78bfa" height={180} />

          {/* Quantile analysis */}
          {analysis.quantile_analysis && <QuantileChart analysis={analysis.quantile_analysis} />}
        </div>
      ) : (
        <EmptyState title="No analysis yet" description="Enter a formula and click Analyze to see results."
          action={<Button size="sm" onClick={() => void handleAnalyze()} disabled={analyzing || !formula.trim()}><Activity className="size-3.5" />Analyze</Button>} />
      )}
    </div>
  )
}
