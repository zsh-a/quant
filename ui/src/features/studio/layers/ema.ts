/**
 * EMA layer — EMA20 / EMA200 lines pulled from `event.features.ema20|ema200`.
 *
 * Both fields are optional (`FeaturesView` doesn't declare them but the
 * backend may emit them as raw numbers); the layer silently skips bars where
 * the value is missing. EMA200 commonly isn't computed for short sessions —
 * it's defaulted off in the toggle panel.
 */

import {
  LineSeries,
  type ISeriesApi,
  type LineData,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { SessionTimeline } from '../types';

const EMA20_COLOR = '#FB8C00';
const EMA200_COLOR = '#9C27B0';

export function readEmaField(
  features: SessionTimeline['events'][number]['features'] | null | undefined,
  key: string,
): number | null {
  if (!features) return null;
  const v = (features as unknown as Record<string, unknown>)[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function buildEmaSeriesData(
  timeline: SessionTimeline,
  key: string,
): LineData<UTCTimestamp>[] {
  const eventByBar = new Map<number, SessionTimeline['events'][number]>();
  for (const ev of timeline.events) eventByBar.set(ev.bar_idx, ev);

  const seen = new Set<number>();
  const out: LineData<UTCTimestamp>[] = [];
  for (let i = 0; i < timeline.bars.length; i++) {
    const ev = eventByBar.get(i);
    const v = readEmaField(ev?.features, key);
    if (v === null) continue;
    const t = Math.floor(timeline.bars[i].timestamp_ns / 1_000_000_000);
    if (seen.has(t)) continue;
    seen.add(t);
    out.push({ time: t as UTCTimestamp, value: v });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

function makeEmaLayer(opts: {
  id: string;
  name: string;
  field: string;
  color: string;
  defaultVisible: boolean;
}): ChartLayer {
  return {
    id: opts.id,
    name: opts.name,
    swatch: opts.color,
    defaultVisible: opts.defaultVisible,

    mount(ctx: LayerCtx): LayerHandle {
      let series: ISeriesApi<'Line'> | null = ctx.chart.addSeries(LineSeries, {
        color: opts.color,
        lineWidth: 2,
        priceLineVisible: false,
        lastValueVisible: false,
        crosshairMarkerVisible: false,
      });

      return {
        update(timeline) {
          if (!series) return;
          series.setData(buildEmaSeriesData(timeline, opts.field));
        },
        unmount() {
          try {
            if (series) ctx.chart.removeSeries(series);
          } catch {
            // chart already disposed
          }
          series = null;
        },
      };
    },
  };
}

export const ema20Layer = makeEmaLayer({
  id: 'ema20',
  name: 'EMA 20',
  field: 'ema20',
  color: EMA20_COLOR,
  defaultVisible: true,
});

export const ema200Layer = makeEmaLayer({
  id: 'ema200',
  name: 'EMA 200',
  field: 'ema200',
  color: EMA200_COLOR,
  defaultVisible: false,
});
