/**
 * Swings layer — Brooks-style HH/HL/LH/LL labels at confirmed swing points.
 *
 * Reads `event.structure.confirmed_swings` from the most recent event at or
 * before `currentBarIdx`. The Brooks pipeline accumulates this list, so the
 * latest event has every swing visible up to that bar.
 *
 * Labels follow the Higher-High / Higher-Low / Lower-High / Lower-Low
 * classification by comparing each new swing to the previous same-kind swing.
 * Each label gets a sequence number so users can trace structure development
 * across the session ("HH₃ broke above LH₂…").
 *
 * To keep the chart sparse we only label the trailing `RECENT_LABEL_COUNT`
 * swings; older ones still render an arrow but with no text. Brooks' own
 * notes only mark the most relevant pivots — earlier swings collapse into
 * unobtrusive context.
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

const RECENT_LABEL_COUNT = 8;

const SUBSCRIPTS: Record<string, string> = {
  '0': '₀',
  '1': '₁',
  '2': '₂',
  '3': '₃',
  '4': '₄',
  '5': '₅',
  '6': '₆',
  '7': '₇',
  '8': '₈',
  '9': '₉',
};

export function isHighSwing(kind: string): boolean {
  return /high/i.test(kind);
}

function subscript(n: number): string {
  return n
    .toString()
    .split('')
    .map((c) => SUBSCRIPTS[c] ?? c)
    .join('');
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

/**
 * `HH` / `HL` / `LH` / `LL` based on each swing's price relative to the most
 * recent same-kind swing. The first high/low gets the neutral `H` / `L`.
 */
export function classifySwingLabel(
  swings: SwingPoint[],
  index: number,
): 'HH' | 'HL' | 'LH' | 'LL' | 'H' | 'L' {
  const target = swings[index];
  const high = isHighSwing(target.kind);
  for (let i = index - 1; i >= 0; i--) {
    const prev = swings[i];
    if (isHighSwing(prev.kind) !== high) continue;
    if (high) return target.price > prev.price ? 'HH' : 'LH';
    return target.price > prev.price ? 'HL' : 'LL';
  }
  return high ? 'H' : 'L';
}

export function buildSwingMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const swings = latestStructureSwings(timeline, currentBarIdx);
  // Sort once by idx so the HH/HL classifier sees swings in chronological
  // order even if the structure list arrives out of order.
  const ordered = swings
    .filter((s) => s.idx <= currentBarIdx)
    .map((s, _i, arr) => ({ s, _i: arr.indexOf(s) }))
    .sort((a, b) => a.s.idx - b.s.idx)
    .map(({ s }) => s);

  // Per-kind running sequence number (HH₁, HH₂, …).
  const counts: Record<string, number> = {};
  const labelStart = Math.max(0, ordered.length - RECENT_LABEL_COUNT);

  const out: SeriesMarker<Time>[] = [];
  for (let i = 0; i < ordered.length; i++) {
    const swing = ordered[i];
    const t = timeForBar(timeline, swing.idx);
    if (t === null) continue;
    const high = isHighSwing(swing.kind);
    const label = classifySwingLabel(ordered, i);
    counts[label] = (counts[label] ?? 0) + 1;
    const seq = counts[label];
    const text = i >= labelStart ? `${label}${subscript(seq)}` : '';
    out.push({
      time: t as Time,
      position: high ? 'aboveBar' : 'belowBar',
      shape: high ? 'arrowDown' : 'arrowUp',
      color: high ? HIGH_COLOR : LOW_COLOR,
      text,
      size: text ? 1 : 0,
      id: `swing-${swing.idx}-${swing.kind}`,
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
