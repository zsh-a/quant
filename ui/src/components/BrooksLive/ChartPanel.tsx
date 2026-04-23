/**
 * ChartPanel — live candlestick chart driven by streaming bars.
 *
 * We already ship echarts-for-react; lightweight-charts is not in the
 * bundle yet, so we reuse the existing stack to keep the feature small.
 */

import { useMemo } from 'react';
import ReactECharts from 'echarts-for-react';
import { SectionCard } from '../layout/SectionCard';
import type { BarOHLCV } from '../../hooks/useBrooksLive';

interface ChartPanelProps {
  bars: BarOHLCV[];
  symbol: string;
}

export function ChartPanel({ bars, symbol }: ChartPanelProps) {
  const option = useMemo(() => {
    const categories = bars.map((b) => b.timestamp.slice(11, 19));
    const ohlc = bars.map((b) => [b.open, b.close, b.low, b.high]);
    return {
      animation: false,
      grid: { top: 24, left: 48, right: 24, bottom: 36 },
      xAxis: {
        type: 'category',
        data: categories,
        axisLabel: { fontSize: 10 },
      },
      yAxis: {
        scale: true,
        splitArea: { show: false },
      },
      tooltip: { trigger: 'axis', axisPointer: { type: 'cross' } },
      series: [
        {
          type: 'candlestick',
          data: ohlc,
          itemStyle: {
            color: '#10b981',
            color0: '#ef4444',
            borderColor: '#10b981',
            borderColor0: '#ef4444',
          },
        },
      ],
    };
  }, [bars]);

  return (
    <SectionCard
      title={symbol ? `${symbol} — realtime` : 'Realtime chart'}
      description={bars.length ? `${bars.length} bars received` : 'waiting for stream…'}
    >
      <div style={{ height: 320 }}>
        {bars.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-muted-foreground">
            等待第一根 bar…
          </div>
        ) : (
          <ReactECharts option={option} style={{ height: 320, width: '100%' }} notMerge lazyUpdate />
        )}
      </div>
    </SectionCard>
  );
}
