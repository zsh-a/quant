/**
 * Quantile analysis visualization — 分层回测图表.
 *
 * Shows:
 *   1. Multi-line equity chart (one line per quantile group)
 *   2. Quantile stats table (return, sharpe, max DD per group)
 *   3. Monotonicity score badge
 */
import React, { useMemo } from 'react'
import { CartesianGrid, ComposedChart, Legend, Line, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import type { QuantileAnalysis } from '../../types'
import { Badge } from '../ui/badge'

// Distinct colors for up to 10 quantile groups
const Q_COLORS = [
  '#ef4444', // Q1 red (worst)
  '#f97316', // Q2 orange
  '#eab308', // Q3 yellow
  '#22c55e', // Q4 green
  '#3b82f6', // Q5 blue (best)
  '#8b5cf6', '#ec4899', '#14b8a6', '#f59e0b', '#6366f1',
]

const LS_COLOR = '#a855f7' // long-short purple

export const QuantileChart: React.FC<{ analysis: QuantileAnalysis }> = ({ analysis }) => {
  const { n_quantiles, quantile_equity, quantile_stats, long_short_equity, monotonicity } = analysis

  // Build chart data: each row = { i, q1, q2, ..., ls }
  const chartData = useMemo(() => {
    const len = quantile_equity[0]?.length ?? 0
    // Downsample if too many points
    const step = len > 500 ? Math.ceil(len / 500) : 1
    const data: Record<string, number>[] = []
    for (let i = 0; i < len; i += step) {
      const row: Record<string, number> = { i: i }
      for (let g = 0; g < n_quantiles; g++) {
        row[`q${g + 1}`] = quantile_equity[g]?.[i] ?? 1
      }
      if (long_short_equity.length > i) {
        row.ls = long_short_equity[i]
      }
      data.push(row)
    }
    return data
  }, [quantile_equity, long_short_equity, n_quantiles])

  const monoLabel = monotonicity >= 0.9 ? 'success' : monotonicity >= 0.5 ? 'info' : 'destructive'

  return (
    <div className="space-y-4">
      {/* Header badges */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">分层回测</span>
        <Badge variant={monoLabel as any}>
          单调性 {(monotonicity * 100).toFixed(0)}%
        </Badge>
        <Badge variant="secondary">{n_quantiles} 分位</Badge>
      </div>

      {/* Multi-line equity chart */}
      <div className="rounded-2xl border border-border/70 bg-card/70 p-4">
        <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-3">
          各分位累计收益
        </div>
        <ResponsiveContainer width="100%" height={260}>
          <ComposedChart data={chartData} margin={{ top: 4, right: 8, bottom: 0, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" opacity={0.4} />
            <XAxis dataKey="i" tick={false} axisLine={false} />
            <YAxis domain={['auto', 'auto']} width={52}
              tick={{ fill: 'hsl(var(--muted-foreground))', fontSize: 11 }}
              tickFormatter={(v: number) => `${((v - 1) * 100).toFixed(0)}%`} />
            <Tooltip
              contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 12, fontSize: 12 }}
              formatter={(v: number, name: string) => [`${((v as number) - 1) * 100 >= 0 ? '+' : ''}${(((v as number) - 1) * 100).toFixed(2)}%`, name]}
              labelFormatter={(i: number) => `Day ${i}`}
            />
            <Legend verticalAlign="top" height={28} iconType="line" wrapperStyle={{ fontSize: 11 }} />
            {Array.from({ length: n_quantiles }, (_, g) => (
              <Line key={`q${g + 1}`} type="monotone" dataKey={`q${g + 1}`}
                name={`Q${g + 1}`} stroke={Q_COLORS[g % Q_COLORS.length]}
                strokeWidth={1.5} dot={false} />
            ))}
            <Line type="monotone" dataKey="ls" name="L/S" stroke={LS_COLOR}
              strokeWidth={2} strokeDasharray="5 3" dot={false} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* Stats table */}
      <div className="rounded-2xl border border-border/70 bg-card/70 overflow-hidden">
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border/50 bg-secondary/30">
              <th className="px-3 py-2 text-left font-semibold text-muted-foreground">分组</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">累计收益</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">年化收益</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">年化夏普</th>
              <th className="px-3 py-2 text-right font-semibold text-muted-foreground">最大回撤</th>
            </tr>
          </thead>
          <tbody>
            {quantile_stats.map((s, idx) => (
              <tr key={s.group} className="border-b border-border/30 last:border-0">
                <td className="px-3 py-2 font-medium">
                  <span className="inline-block w-2.5 h-2.5 rounded-full mr-1.5"
                    style={{ backgroundColor: Q_COLORS[idx % Q_COLORS.length] }} />
                  Q{s.group}
                  {idx === 0 && <span className="ml-1 text-muted-foreground">(弱)</span>}
                  {idx === quantile_stats.length - 1 && <span className="ml-1 text-muted-foreground">(强)</span>}
                </td>
                <td className={`px-3 py-2 text-right font-mono ${s.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {s.total_return >= 0 ? '+' : ''}{(s.total_return * 100).toFixed(2)}%
                </td>
                <td className={`px-3 py-2 text-right font-mono ${s.annual_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>
                  {s.annual_return >= 0 ? '+' : ''}{(s.annual_return * 100).toFixed(2)}%
                </td>
                <td className="px-3 py-2 text-right font-mono">{s.annual_sharpe.toFixed(2)}</td>
                <td className="px-3 py-2 text-right font-mono text-rose-400">
                  -{(s.max_drawdown * 100).toFixed(2)}%
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
