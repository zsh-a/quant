/**
 * Interactive chart components for Alpha Lab using ECharts.
 *
 * Features: time-based x-axis, crosshair tooltip, dataZoom slider, responsive resize.
 */
import { useMemo } from 'react'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import * as echarts from 'echarts/core'
import { LineChart, BarChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, DataZoomComponent,
  LegendComponent, ToolboxComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { AlphaLabSeriesPoint } from '../../types'

echarts.use([
  LineChart, BarChart, GridComponent, TooltipComponent,
  DataZoomComponent, LegendComponent, ToolboxComponent, CanvasRenderer,
])

/* ── Helpers ── */

/** Format timestamp for axis display: "2024-03-15" or "03-15 14:00" */
function fmtAxisLabel(t: string): string {
  if (t.length <= 10) return t  // date only
  // datetime — show short form
  return t.slice(5, 16).replace('T', ' ')
}

/** Format timestamp for tooltip: full date or datetime */
function fmtTooltipLabel(t: string | undefined, idx: number): string {
  if (!t) return `Bar ${idx}`
  if (t.length <= 10) return t
  return t.slice(0, 16).replace('T', ' ')
}

/* ── Single-series chart (equity, drawdown, turnover) ── */

export function MiniChart({ data, label, color = '#6366f1', height = 260, pct }: {
  data: AlphaLabSeriesPoint[]; label: string; color?: string; height?: number; pct?: boolean
}) {
  const hasTime = data?.[0]?.t != null

  const option = useMemo(() => {
    if (!data?.length) return null
    const values = data.map(d => d.v)
    const xData = hasTime ? data.map(d => d.t!) : data.map(d => d.i)

    return {
      backgroundColor: 'transparent',
      grid: { left: 52, right: 16, top: 28, bottom: hasTime ? 56 : 48 },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'hsl(var(--card))',
        borderColor: 'hsl(var(--border))',
        textStyle: { color: 'hsl(var(--foreground))', fontSize: 11 },
        formatter: (params: any) => {
          const p = params[0]
          const v = pct ? `${(p.value * 100).toFixed(2)}%` : p.value.toFixed(4)
          const time = fmtTooltipLabel(hasTime ? data[p.dataIndex]?.t : undefined, p.dataIndex)
          return `<b>${label}</b><br/>${time}: ${v}`
        },
        axisPointer: { type: 'cross', lineStyle: { type: 'dashed', color: 'hsl(var(--muted-foreground))' } },
      },
      xAxis: {
        type: 'category' as const,
        data: xData,
        axisLabel: hasTime ? {
          color: 'hsl(var(--muted-foreground))',
          fontSize: 9,
          formatter: (v: string) => fmtAxisLabel(v),
          rotate: 0,
        } : { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value' as const,
        splitLine: { lineStyle: { color: 'hsl(var(--border))', opacity: 0.3 } },
        axisLabel: {
          color: 'hsl(var(--muted-foreground))', fontSize: 10,
          formatter: pct ? (v: number) => `${(v * 100).toFixed(0)}%` : (v: number) => v.toFixed(2),
        },
      },
      dataZoom: [{
        type: 'slider',
        height: 18,
        bottom: 4,
        borderColor: 'transparent',
        backgroundColor: 'hsl(var(--border))',
        fillerColor: color + '30',
        handleStyle: { color },
        textStyle: { color: 'hsl(var(--muted-foreground))', fontSize: 9 },
        labelFormatter: hasTime ? (_: number, val: string) => fmtAxisLabel(val) : undefined,
      }],
      series: [{
        type: 'line',
        data: values,
        smooth: 0.3,
        symbol: 'none',
        lineStyle: { width: 1.5, color },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
          { offset: 0, color: color + '25' },
          { offset: 1, color: color + '05' },
        ])},
      }],
    }
  }, [data, label, color, pct, hasTime])

  if (!option) return null

  return (
    <div className="rounded-xl border border-border/60 bg-card/60 overflow-hidden">
      <div className="px-3 pt-2 text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">{label}</div>
      <ReactEChartsCore echarts={echarts} option={option} style={{ height }} notMerge lazyUpdate />
    </div>
  )
}

/* ── MetricGrid ── */

export function MetricGrid({ metrics, keys, labels }: {
  metrics: Record<string, number>; keys: readonly string[]; labels?: Record<string, string>
}) {
  const LABELS: Record<string, string> = {
    sharpe: 'Sharpe', rank_ic: 'Rank IC', ic_ir: 'IC IR', ic_std: 'IC Std', ic_decay: 'IC Decay',
    pnl_per_turnover: 'PnL/Turn', total_return: 'Return', max_drawdown: 'Max DD',
    avg_turnover: 'Turnover', signal_coverage: 'Coverage', turnover_proxy: 'Turn Proxy',
    rank_ic_1d: 'IC 1D', rank_ic_5d: 'IC 5D', rank_ic_10d: 'IC 10D',
    calmar: 'Calmar', win_rate: 'Win Rate', skewness: 'Skewness', ...labels,
  }
  const PCT = new Set(['total_return', 'max_drawdown', 'volatility', 'signal_coverage', 'turnover_proxy', 'win_rate'])
  const format = (k: string, v: number) => {
    if (!Number.isFinite(v)) return '--'
    if (PCT.has(k)) return `${(v * 100).toFixed(2)}%`
    return Math.abs(v) >= 10 ? v.toFixed(2) : v.toFixed(4)
  }
  const colorFor = (k: string, v: number) => {
    if (k === 'max_drawdown') return 'text-rose-400'
    if (['sharpe', 'rank_ic', 'total_return', 'calmar', 'ic_ir'].includes(k)) {
      return v > 0 ? 'text-emerald-400' : v < 0 ? 'text-rose-400' : ''
    }
    return ''
  }
  return (
    <div className="grid gap-2 grid-cols-2 sm:grid-cols-3 md:grid-cols-5 xl:grid-cols-5">
      {keys.map(k => metrics[k] != null ? (
        <div key={k} className="rounded-lg border border-border/50 bg-secondary/20 px-2.5 py-2">
          <div className="text-[9px] font-semibold uppercase tracking-wider text-muted-foreground">{LABELS[k] ?? k}</div>
          <div className={`mt-0.5 text-sm font-semibold ${colorFor(k, metrics[k])}`}>{format(k, metrics[k])}</div>
        </div>
      ) : null)}
    </div>
  )
}
