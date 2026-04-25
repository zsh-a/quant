/**
 * Stop ladder layer — step-line of every `stop_adj` event up to the current
 * bar (initial → BE → trail).
 *
 * The stop value is held flat between adjustments and steps to the new value
 * exactly on the bar where the adjustment fired, giving a staircase-shaped
 * line. We additionally drop a small text marker on each step so users can
 * eyeball the reason (`init`, `be`, `trail`, `chandelier`, etc.).
 */

import {
  LineSeries,
  LineStyle,
  createSeriesMarkers,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type LineData,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { SessionTimeline, StopAdj } from '../types';

const STOP_COLOR = '#FF7043';

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

export interface StopAdjPoint {
  bar_idx: number;
  adj: StopAdj;
}

export function collectStopAdjustments(
  timeline: SessionTimeline,
  currentBarIdx: number,
): StopAdjPoint[] {
  const out: StopAdjPoint[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > currentBarIdx) break;
    if (ev.stop_adj) out.push({ bar_idx: ev.bar_idx, adj: ev.stop_adj });
  }
  return out;
}

export function buildStopLadderData(
  timeline: SessionTimeline,
  currentBarIdx: number,
): LineData<UTCTimestamp>[] {
  const points = collectStopAdjustments(timeline, currentBarIdx);
  if (points.length === 0) return [];

  // Build a value-per-bar series: hold flat until each adjustment's bar
  // then step to the new value. Lightweight-charts requires strictly
  // ascending unique timestamps; one sample per bar gives that for free.
  const adjByBar = new Map<number, StopAdj>();
  for (const { bar_idx, adj } of points) adjByBar.set(bar_idx, adj);

  const startBar = points[0].bar_idx;
  const endBar = Math.min(currentBarIdx, timeline.bars.length - 1);

  const out: LineData<UTCTimestamp>[] = [];
  let value = points[0].adj.from_px;
  for (let i = startBar; i <= endBar; i++) {
    const adj = adjByBar.get(i);
    if (adj) value = adj.to_px;
    const t = timeForBar(timeline, i);
    if (t === null) continue;
    out.push({ time: t, value });
  }
  return out;
}

export function buildStopAdjMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const { bar_idx, adj } of collectStopAdjustments(timeline, currentBarIdx)) {
    const t = timeForBar(timeline, bar_idx);
    if (t === null) continue;
    out.push({
      time: t as Time,
      position: 'aboveBar',
      shape: 'square',
      color: STOP_COLOR,
      text: adj.reason || 'stop',
      size: 0,
      id: `stop-adj-${bar_idx}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export const stopAdjLayer: ChartLayer = {
  id: 'stop_adj',
  name: 'Stop ladder',
  swatch: STOP_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    const { chart } = ctx;
    let series: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      color: STOP_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );

    return {
      update(timeline, currentBarIdx) {
        if (!series) return;
        series.setData(buildStopLadderData(timeline, currentBarIdx));
        plugin?.setMarkers(buildStopAdjMarkers(timeline, currentBarIdx));
      },
      unmount() {
        try {
          if (series) chart.removeSeries(series);
        } catch {
          // chart already disposed
        }
        try {
          plugin?.detach();
        } catch {
          // chart already disposed
        }
        series = null;
        plugin = null;
      },
    };
  },
};
