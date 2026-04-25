/**
 * Fills layer — diamond-style markers for actual order fills.
 *
 * `buy` / `buy_to_cover` → green ▲ above bar.
 * `sell` / `sell_short`  → red ▼ below bar.
 * Marker text shows the fill price for quick read-off.
 */

import {
  createSeriesMarkers,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { FillView, SessionTimeline } from '../types';

const BUY_COLOR = '#26A69A';
const SELL_COLOR = '#EF5350';

const BUY_SIDES: ReadonlySet<FillView['side']> = new Set(['buy', 'buy_to_cover']);

export function isBuyFill(side: FillView['side']): boolean {
  return BUY_SIDES.has(side);
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

export function buildFillMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > currentBarIdx) continue;
    if (!ev.fill) continue;
    const t = timeForBar(timeline, ev.bar_idx);
    if (t === null) continue;
    const buy = isBuyFill(ev.fill.side);
    out.push({
      time: t as Time,
      position: buy ? 'aboveBar' : 'belowBar',
      shape: buy ? 'arrowUp' : 'arrowDown',
      color: buy ? BUY_COLOR : SELL_COLOR,
      text: `◆ ${ev.fill.price.toFixed(2)}`,
      size: 1,
      id: `fill-${ev.bar_idx}-${ev.fill.side}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export const fillsLayer: ChartLayer = {
  id: 'fills',
  name: 'Fills',
  swatch: BUY_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );

    return {
      update(timeline, currentBarIdx) {
        if (!plugin) return;
        plugin.setMarkers(buildFillMarkers(timeline, currentBarIdx));
      },
      unmount() {
        try {
          plugin?.detach();
        } catch {
          // chart already disposed
        }
        plugin = null;
      },
    };
  },
};
