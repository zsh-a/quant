/**
 * Regime layer — bottom 5% color band coloured per-bar by `event.regime.name`.
 *
 * Implemented as a HistogramSeries on its own overlay price scale so it
 * occupies a thin strip beneath the candles without disturbing the price axis.
 */

import {
  HistogramSeries,
  type HistogramData,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { SessionTimeline } from '../types';

const REGIME_COLORS: Record<string, string> = {
  strong_bull_trend: 'rgba(38, 166, 154, 0.45)',
  weak_bull_trend: 'rgba(38, 166, 154, 0.22)',
  strong_bear_trend: 'rgba(239, 83, 80, 0.45)',
  weak_bear_trend: 'rgba(239, 83, 80, 0.22)',
  tight_tr: 'rgba(120, 120, 120, 0.30)',
  broad_tr: 'rgba(120, 120, 120, 0.18)',
  climax: 'rgba(255, 152, 0, 0.55)',
  channel: 'rgba(96, 165, 250, 0.35)',
  pullback: 'rgba(180, 180, 180, 0.20)',
};
const REGIME_FALLBACK = 'rgba(80, 80, 80, 0.10)';

export function regimeColorOf(name: string | undefined | null): string {
  if (!name) return REGIME_FALLBACK;
  return REGIME_COLORS[name] ?? REGIME_FALLBACK;
}

const PRICE_SCALE_ID = 'regime-band';

export const regimeLayer: ChartLayer = {
  id: 'regime',
  name: 'Regime band',
  swatch: REGIME_COLORS.strong_bull_trend,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    const { chart } = ctx;
    const series: ISeriesApi<'Histogram'> = chart.addSeries(HistogramSeries, {
      priceScaleId: PRICE_SCALE_ID,
      priceLineVisible: false,
      lastValueVisible: false,
      base: 0,
    });
    chart.priceScale(PRICE_SCALE_ID).applyOptions({
      scaleMargins: { top: 0.95, bottom: 0 },
    });

    return {
      update(timeline: SessionTimeline) {
        const eventByBar = new Map<number, string | null | undefined>();
        for (const ev of timeline.events) {
          eventByBar.set(ev.bar_idx, ev.regime?.name);
        }
        const seen = new Set<number>();
        const data: HistogramData<UTCTimestamp>[] = [];
        for (let i = 0; i < timeline.bars.length; i++) {
          const bar = timeline.bars[i];
          const t = Math.floor(bar.timestamp_ns / 1_000_000_000);
          if (seen.has(t)) continue;
          seen.add(t);
          data.push({
            time: t as UTCTimestamp,
            value: 1,
            color: regimeColorOf(eventByBar.get(i)),
          });
        }
        data.sort((a, b) => (a.time as number) - (b.time as number));
        series.setData(data);
      },
      unmount() {
        try {
          chart.removeSeries(series);
        } catch {
          // chart already disposed
        }
      },
    };
  },
};
