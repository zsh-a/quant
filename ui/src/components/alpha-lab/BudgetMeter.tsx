/**
 * BudgetMeter — compact progress bars for eval / tokens / wall-time / cost.
 *
 * Accepts a ``BudgetSnapshot`` from the pipeline record. Unlimited caps
 * (cap = 0) are rendered as a neutral running counter rather than a bar.
 */
import type { BudgetSnapshot } from '../../types'

function clamp01(v: number): number {
  if (!Number.isFinite(v) || v <= 0) return 0
  return Math.min(1, v)
}

function fmtSec(s: number | null | undefined): string {
  if (s == null || !Number.isFinite(s)) return '—'
  if (s < 90) return `${Math.round(s)}s`
  if (s < 5400) return `${Math.round(s / 60)}m`
  return `${(s / 3600).toFixed(1)}h`
}

function fmtInt(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(2)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`
  return String(n)
}

function fmtCost(n: number | null | undefined): string {
  if (n == null || !Number.isFinite(n)) return '—'
  return `$${n.toFixed(4)}`
}

interface BarProps {
  label: string
  used: number
  max: number
  formatter: (n: number) => string
  exhausted?: boolean
}

function Bar({ label, used, max, formatter, exhausted }: BarProps) {
  if (max <= 0) {
    return (
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-muted-foreground">{formatter(used)}</span>
      </div>
    )
  }
  const pct = clamp01(used / max)
  const danger = exhausted || pct >= 1.0
  const color = danger
    ? 'bg-red-500'
    : pct >= 0.8
    ? 'bg-amber-500'
    : 'bg-emerald-500'
  return (
    <div className="space-y-0.5">
      <div className="flex items-center justify-between text-[10px]">
        <span className="text-muted-foreground">{label}</span>
        <span className={`font-mono ${danger ? 'text-red-400' : 'text-foreground/80'}`}>
          {formatter(used)} / {formatter(max)}
        </span>
      </div>
      <div className="h-1 w-full overflow-hidden rounded-full bg-secondary/60">
        <div className={`h-full transition-all ${color}`} style={{ width: `${(pct * 100).toFixed(1)}%` }} />
      </div>
    </div>
  )
}

interface Props {
  snapshot: BudgetSnapshot | null | undefined
  compact?: boolean
}

export function BudgetMeter({ snapshot, compact = false }: Props) {
  if (!snapshot) return null
  const { exhausted_reasons: reasons = [] } = snapshot
  const reasonSet = new Set(reasons)
  return (
    <div className={compact ? 'space-y-1' : 'space-y-2 rounded-xl border border-border/40 bg-card/60 p-3'}>
      {!compact && (
        <div className="flex items-center justify-between text-[11px]">
          <span className="font-semibold uppercase tracking-wider text-muted-foreground">Budget</span>
          {snapshot.exhausted && (
            <span className="text-red-400">exhausted: {reasons.join(', ')}</span>
          )}
        </div>
      )}
      <Bar
        label="Full eval"
        used={snapshot.used_full_eval}
        max={snapshot.max_full_eval}
        formatter={fmtInt}
        exhausted={reasonSet.has('full_eval')}
      />
      <Bar
        label="LLM tokens"
        used={snapshot.used_llm_tokens}
        max={snapshot.max_llm_tokens}
        formatter={fmtInt}
        exhausted={reasonSet.has('llm_tokens')}
      />
      <Bar
        label="Cost (USD)"
        used={snapshot.used_cost_usd}
        max={snapshot.max_cost_usd}
        formatter={fmtCost}
        exhausted={reasonSet.has('cost_usd')}
      />
      <Bar
        label="Wall time"
        used={snapshot.elapsed_sec}
        max={snapshot.max_wall_time_sec}
        formatter={fmtSec}
        exhausted={reasonSet.has('wall_time')}
      />
    </div>
  )
}
