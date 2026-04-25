/**
 * Decisions layer — solid in-bar arrows for accepted Brooks decisions, plus
 * entry / stop / target price lines for any decision within
 * `currentBarIdx ± WINDOW`. Outside the window the lines are removed so a
 * busy chart doesn't drown in horizontals.
 */

import {
  LineStyle,
  createSeriesMarkers,
  type IPriceLine,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { Decision, SessionTimeline } from '../types';

const LONG_COLOR = '#26A69A';
const SHORT_COLOR = '#EF5350';
const ENTRY_COLOR = '#FFCA28';
const STOP_COLOR = '#EF5350';
const TARGET_COLOR = '#26A69A';

const PRICE_LINE_WINDOW = 5;

export function decisionColor(side: Decision['side']): string {
  return side === 'long' ? LONG_COLOR : SHORT_COLOR;
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

export interface BuiltDecision {
  decision: Decision;
  bar_idx: number;
}

export function collectDecisions(
  timeline: SessionTimeline,
  currentBarIdx: number,
): BuiltDecision[] {
  const out: BuiltDecision[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > currentBarIdx) continue;
    if (!ev.decision) continue;
    out.push({ decision: ev.decision, bar_idx: ev.bar_idx });
  }
  return out;
}

export function buildDecisionMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const { decision, bar_idx } of collectDecisions(timeline, currentBarIdx)) {
    const t = timeForBar(timeline, bar_idx);
    if (t === null) continue;
    out.push({
      time: t as Time,
      position: 'inBar',
      shape: decision.side === 'long' ? 'arrowUp' : 'arrowDown',
      color: decisionColor(decision.side),
      text: `${decision.side.toUpperCase()} ${decision.pattern}`,
      size: 2,
      id: `dec-${bar_idx}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export function decisionsInWindow(
  timeline: SessionTimeline,
  currentBarIdx: number,
  window: number = PRICE_LINE_WINDOW,
): BuiltDecision[] {
  const lo = Math.max(0, currentBarIdx - window);
  const hi = currentBarIdx + window;
  const out: BuiltDecision[] = [];
  for (const ev of timeline.events) {
    if (!ev.decision) continue;
    if (ev.bar_idx < lo || ev.bar_idx > hi) continue;
    if (ev.bar_idx > currentBarIdx) continue; // never leak the future
    out.push({ decision: ev.decision, bar_idx: ev.bar_idx });
  }
  return out;
}

export const decisionsLayer: ChartLayer = {
  id: 'decisions',
  name: 'Decisions',
  swatch: LONG_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );
    let priceLines: IPriceLine[] = [];

    const clearLines = () => {
      for (const pl of priceLines) {
        try {
          ctx.primarySeries.removePriceLine(pl);
        } catch {
          // series may already be detached
        }
      }
      priceLines = [];
    };

    const addLine = (price: number, color: string, title: string) => {
      priceLines.push(
        ctx.primarySeries.createPriceLine({
          price,
          color,
          lineWidth: 1,
          lineStyle: LineStyle.Solid,
          axisLabelVisible: true,
          title,
        }),
      );
    };

    return {
      update(timeline, currentBarIdx) {
        if (!plugin) return;
        plugin.setMarkers(buildDecisionMarkers(timeline, currentBarIdx));

        clearLines();
        for (const { decision, bar_idx } of decisionsInWindow(timeline, currentBarIdx)) {
          addLine(decision.entry_px, ENTRY_COLOR, `entry @${bar_idx}`);
          addLine(decision.stop_px, STOP_COLOR, `stop @${bar_idx}`);
          if (decision.target_px != null) {
            addLine(decision.target_px, TARGET_COLOR, `target @${bar_idx}`);
          }
        }
      },
      unmount() {
        clearLines();
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

