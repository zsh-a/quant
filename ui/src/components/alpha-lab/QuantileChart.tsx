/**
 * Quantile analysis visualization — 分层回测.
 * ECharts interactive chart with time axis + compact stats table.
 */
import React, { useMemo } from 'react'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import * as echarts from 'echarts/core'
import { LineChart } from 'echarts/charts'
import {
  GridComponent, TooltipComponent, DataZoomComponent,
  LegendComponent,
} from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import type { QuantileAnalysis } from '../../types'
import { Badge } from '../ui/badge'

echarts.use([LineChart, GridComponent, TooltipComponent, DataZoomComponent, LegendComponent, CanvasRenderer])

const Q_COLORS = ['#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6', '#ec4899', '#14b8a6', '#f59e0b', '#6366f1']
const LS_COLOR = '#a855f7'

function fmtAxisLabel(t: string): string {
  if (t.length <= 10) return t
  return t.slice(5, 16).replace('T', ' ')
}

export const QuantileChart: React.FC<{ analysis: QuantileAnalysis }> = ({ analysis }) => {
  const { n_quantiles, quantile_equity, quantile_stats, long_short_equity, monotonicity, timestamps } = analysis
  const hasTime = timestamps && timestamps.length > 0

  const option = useMemo(() => {
    const len = quantile_equity[0]?.length ?? 0
    const step = len > 600 ? Math.ceil(len / 600) : 1
    const xData: string[] = []
    const seriesData: number[][] = Array.from({ length: n_quantiles + 1 }, () => [])

    for (let i = 0; i < len; i += step) {
      xData.push(hasTime && timestamps![i] ? timestamps![i] : String(i))
      for (let g = 0; g < n_quantiles; g++) seriesData[g].push(quantile_equity[g]?.[i] ?? 1)
      seriesData[n_quantiles].push(long_short_equity[i] ?? 1)
    }

    const series = Array.from({ length: n_quantiles }, (_, g) => ({
      name: `Q${g + 1}`,
      type: 'line' as const,
      data: seriesData[g],
      smooth: 0.2,
      symbol: 'none',
      lineStyle: { width: 1.5, color: Q_COLORS[g % Q_COLORS.length] },
    }))
    series.push({
      name: 'L/S',
      type: 'line' as const,
      data: seriesData[n_quantiles],
      smooth: 0.2,
      symbol: 'none',
      lineStyle: { width: 2, color: LS_COLOR, type: 'dashed' as any },
    })

    return {
      backgroundColor: 'transparent',
      grid: { left: 48, right: 12, top: 36, bottom: hasTime ? 56 : 44 },
      legend: {
        top: 0, left: 'center',
        textStyle: { color: 'hsl(var(--muted-foreground))', fontSize: 10 },
        itemWidth: 14, itemHeight: 2, itemGap: 8,
      },
      tooltip: {
        trigger: 'axis',
        backgroundColor: 'hsl(var(--card))',
        borderColor: 'hsl(var(--border))',
        textStyle: { color: 'hsl(var(--foreground))', fontSize: 11 },
        formatter: (params: any) => {
          const label = hasTime ? fmtAxisLabel(params[0].axisValue) : `#${params[0].dataIndex * step}`
          const lines = params.map((p: any) => {
            const ret = ((p.value - 1) * 100).toFixed(1)
            return `<span style="color:${p.color}">●</span> ${p.seriesName}: ${p.value >= 1 ? '+' : ''}${ret}%`
          })
          return `${label}<br/>${lines.join('<br/>')}`
        },
        axisPointer: { type: 'cross', lineStyle: { type: 'dashed', color: 'hsl(var(--muted-foreground))' } },
      },
      xAxis: {
        type: 'category' as const,
        data: xData,
        axisLabel: hasTime ? {
          color: 'hsl(var(--muted-foreground))', fontSize: 9,
          formatter: (v: string) => fmtAxisLabel(v),
        } : { show: false },
        axisLine: { show: false },
        axisTick: { show: false },
      },
      yAxis: {
        type: 'value' as const,
        splitLine: { lineStyle: { color: 'hsl(var(--border))', opacity: 0.3 } },
        axisLabel: {
          color: 'hsl(var(--muted-foreground))', fontSize: 10,
          formatter: (v: number) => `${((v - 1) * 100).toFixed(0)}%`,
        },
      },
      dataZoom: [{
        type: 'slider', height: 16, bottom: 4,
        borderColor: 'transparent',
        backgroundColor: 'hsl(var(--border))',
        fillerColor: LS_COLOR + '30',
        handleStyle: { color: LS_COLOR },
        textStyle: { color: 'hsl(var(--muted-foreground))', fontSize: 9 },
        labelFormatter: hasTime ? (_: number, val: string) => fmtAxisLabel(val) : undefined,
      }],
      series,
    }
  }, [quantile_equity, long_short_equity, n_quantiles, timestamps, hasTime])

  const monoVariant = monotonicity >= 0.9 ? 'success' : monotonicity >= 0.5 ? 'info' : 'destructive'
  const fmtPct = (v: number) => `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`

  return (
    <div className="rounded-xl border border-border/60 bg-card/60 overflow-hidden">
      {/* Header */}
      <div className="flex items-center gap-2 px-3 pt-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">分层回测</span>
        <Badge variant={monoVariant as any} className="text-[9px]">单调性 {(monotonicity * 100).toFixed(0)}%</Badge>
        <Badge variant="secondary" className="text-[9px]">{quantile_stats.length} 分位</Badge>
      </div>

      {/* Chart */}
      <ReactEChartsCore echarts={echarts} option={option} style={{ height: 280 }} notMerge lazyUpdate />

      {/* Compact stats table */}
      <div className="border-t border-border/40 px-3 py-2 overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr>
              <th className="px-1.5 py-0.5 text-left font-medium text-muted-foreground">组</th>
              <th className="px-1.5 py-0.5 text-right font-medium text-muted-foreground">累计</th>
              <th className="px-1.5 py-0.5 text-right font-medium text-muted-foreground">年化</th>
              <th className="px-1.5 py-0.5 text-right font-medium text-muted-foreground">夏普</th>
              <th className="px-1.5 py-0.5 text-right font-medium text-muted-foreground">回撤</th>
            </tr>
          </thead>
          <tbody>
            {quantile_stats.map((s, idx) => (
              <tr key={s.group}>
                <td className="px-1.5 py-0.5 font-medium whitespace-nowrap">
                  <span className="inline-block w-1.5 h-1.5 rounded-full mr-1" style={{ backgroundColor: Q_COLORS[idx % Q_COLORS.length] }} />
                  Q{s.group}
                </td>
                <td className={`px-1.5 py-0.5 text-right font-mono ${s.total_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtPct(s.total_return)}</td>
                <td className={`px-1.5 py-0.5 text-right font-mono ${s.annual_return >= 0 ? 'text-emerald-400' : 'text-rose-400'}`}>{fmtPct(s.annual_return)}</td>
                <td className="px-1.5 py-0.5 text-right font-mono">{s.annual_sharpe.toFixed(1)}</td>
                <td className="px-1.5 py-0.5 text-right font-mono text-rose-400">{fmtPct(-s.max_drawdown)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
