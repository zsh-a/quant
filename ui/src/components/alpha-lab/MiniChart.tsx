import React from 'react'
import { Area, CartesianGrid, ComposedChart, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { AlphaLabSeriesPoint } from '../../types'

export function MiniChart({ data, label, color = 'hsl(var(--primary))', height = 180, pct }: {
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

export function MetricGrid({ metrics, keys, labels }: {
  metrics: Record<string, number>; keys: readonly string[]; labels?: Record<string, string>
}) {
  const LABELS: Record<string, string> = {
    sharpe: 'Sharpe', rank_ic: 'Rank IC', ic_ir: 'IC IR', ic_std: 'IC Std', ic_decay: 'IC Decay',
    pnl_per_turnover: 'PnL/Turnover', total_return: 'Total Return', max_drawdown: 'Max DD',
    avg_turnover: 'Avg Turnover', signal_coverage: 'Coverage', turnover_proxy: 'Turnover Proxy',
    rank_ic_1d: 'IC 1D', rank_ic_5d: 'IC 5D', rank_ic_10d: 'IC 10D', ...labels,
  }
  const PCT = new Set(['total_return', 'max_drawdown', 'volatility', 'signal_coverage', 'turnover_proxy'])
  const format = (k: string, v: number) => {
    if (!Number.isFinite(v)) return '--'
    if (PCT.has(k)) return `${(v * 100).toFixed(2)}%`
    return Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(4)
  }
  return (
    <div className="grid gap-3 grid-cols-2 md:grid-cols-4 xl:grid-cols-8">
      {keys.map(k => metrics[k] != null ? (
        <div key={k} className="rounded-xl border border-border/60 bg-secondary/30 px-3 py-2.5">
          <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{LABELS[k] ?? k}</div>
          <div className="mt-1 text-base font-semibold text-foreground">{format(k, metrics[k])}</div>
        </div>
      ) : null)}
    </div>
  )
}
