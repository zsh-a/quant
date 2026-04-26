/**
 * Reasoning layer — small ℹ️ marker on every bar that carries an LLM/VLM
 * decision reasoning string, so users can spot at a glance which bars have
 * model commentary.
 *
 * Hover preview is delivered through the `text` field on the marker (a 60-
 * char single-line excerpt); clicking jumps the user to the
 * DecisionInspector via the studio store's `setBar` action, which is what
 * the side panel is already keyed off.
 *
 * Future-info safety: bars with `bar_idx > currentBarIdx` are dropped.
 */

import {
  createSeriesMarkers,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { BarEvent, SessionTimeline } from '../types';

const HINT_COLOR = '#60a5fa';

const PREVIEW_LIMIT = 60;

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

export function reasoningSnippet(ev: BarEvent): string | null {
  const candidates: (string | undefined)[] = [
    ev.decision?.reasoning as string | undefined,
    ...((ev.signals ?? []).map((s) => s.reasoning as string | undefined)),
  ];
  for (const txt of candidates) {
    if (typeof txt !== 'string') continue;
    const trimmed = txt.replace(/\s+/g, ' ').trim();
    if (!trimmed) continue;
    return trimmed.length > PREVIEW_LIMIT
      ? `${trimmed.slice(0, PREVIEW_LIMIT - 1)}…`
      : trimmed;
  }
  return null;
}

export function buildReasoningMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > currentBarIdx) continue;
    const snippet = reasoningSnippet(ev);
    if (!snippet) continue;
    const t = timeForBar(timeline, ev.bar_idx);
    if (t === null) continue;
    out.push({
      time: t as Time,
      position: 'aboveBar',
      shape: 'circle',
      color: HINT_COLOR,
      text: `ℹ ${snippet}`,
      size: 0,
      id: `reasoning-${ev.bar_idx}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export const reasoningLayer: ChartLayer = {
  id: 'reasoning',
  name: 'Reasoning hints',
  swatch: HINT_COLOR,
  defaultVisible: false,

  mount(ctx: LayerCtx): LayerHandle {
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );

    return {
      update(timeline, currentBarIdx) {
        if (!plugin) return;
        plugin.setMarkers(buildReasoningMarkers(timeline, currentBarIdx));
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
