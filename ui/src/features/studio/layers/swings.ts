/**
 * Swings layer — ▲▼ markers for confirmed swing high/low points.
 *
 * Reads `event.structure.confirmed_swings` from the most recent event at or
 * before `currentBarIdx`. The Brooks pipeline accumulates this list, so the
 * latest event has every swing visible up to that bar.
 */

import {
  createSeriesMarkers,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { SessionTimeline, SwingPoint } from '../types';

const HIGH_COLOR = '#EF5350';
const LOW_COLOR = '#26A69A';

export function isHighSwing(kind: string): boolean {
  return /high/i.test(kind);
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

function latestStructureSwings(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SwingPoint[] {
  for (let i = Math.min(currentBarIdx, timeline.events.length - 1); i >= 0; i--) {
    const ev = timeline.events[i];
    if (ev?.structure?.confirmed_swings?.length) {
      return ev.structure.confirmed_swings;
    }
  }
  return [];
}

export function buildSwingMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const swings = latestStructureSwings(timeline, currentBarIdx);
  const out: SeriesMarker<Time>[] = [];
  for (const s of swings) {
    if (s.idx > currentBarIdx) continue;
    const t = timeForBar(timeline, s.idx);
    if (t === null) continue;
    const high = isHighSwing(s.kind);
    out.push({
      time: t as Time,
      position: high ? 'aboveBar' : 'belowBar',
      shape: high ? 'arrowDown' : 'arrowUp',
      color: high ? HIGH_COLOR : LOW_COLOR,
      text: `#${s.idx}`,
      size: 1,
      id: `swing-${s.idx}-${s.kind}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export const swingsLayer: ChartLayer = {
  id: 'swings',
  name: 'Swings',
  swatch: LOW_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );

    return {
      update(timeline, currentBarIdx) {
        if (!plugin) return;
        plugin.setMarkers(buildSwingMarkers(timeline, currentBarIdx));
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
