/**
 * HTF overlay layer — projects key swing levels from the first available HTF
 * timeframe onto the primary chart as horizontal dashed price lines.
 *
 * For each HTF bar at-or-before the current LTF bar's timestamp we derive a
 * "swing high" / "swing low" by comparing the bar's high/low against the
 * preceding HTF bars; the most recent few survive as `IPriceLine`s on the
 * primary series. This gives a quick visual reference for the dominant
 * higher-timeframe levels without leaving the primary candle view.
 */

import {
  LineStyle,
  type IPriceLine,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { Bar, SessionTimeline } from '../types';

const HIGH_COLOR = 'rgba(239, 83, 80, 0.65)';
const LOW_COLOR = 'rgba(38, 166, 154, 0.65)';

/** Maximum swings (high + low combined) drawn at any time. */
export const HTF_OVERLAY_MAX_LEVELS = 6;

/** Number of HTF bars on each side of a candidate that must be lower (for a high) or higher (for a low). */
const SWING_LOOKBACK = 2;

export interface HtfSwing {
  bar_idx: number;
  price: number;
  kind: 'high' | 'low';
  interval: string;
}

function pickInterval(timeline: SessionTimeline): string | null {
  if (timeline.htf_intervals.length > 0) return timeline.htf_intervals[0];
  const fromMap = Object.keys(timeline.htf_bars);
  return fromMap.length > 0 ? fromMap[0] : null;
}

/** Identify confirmed HTF swings within `bars[0..end]`. */
export function detectSwings(bars: Bar[], end: number): HtfSwing[] {
  const out: HtfSwing[] = [];
  const last = Math.min(end, bars.length - 1);
  for (let i = SWING_LOOKBACK; i <= last - SWING_LOOKBACK; i++) {
    const cur = bars[i];
    let isHigh = true;
    let isLow = true;
    for (let k = 1; k <= SWING_LOOKBACK; k++) {
      const left = bars[i - k];
      const right = bars[i + k];
      if (cur.high <= left.high || cur.high <= right.high) isHigh = false;
      if (cur.low >= left.low || cur.low >= right.low) isLow = false;
      if (!isHigh && !isLow) break;
    }
    if (isHigh) out.push({ bar_idx: i, price: cur.high, kind: 'high', interval: '' });
    if (isLow) out.push({ bar_idx: i, price: cur.low, kind: 'low', interval: '' });
  }
  return out;
}

export function selectHtfLevels(
  timeline: SessionTimeline,
  currentBarIdx: number,
  maxLevels: number = HTF_OVERLAY_MAX_LEVELS,
): HtfSwing[] {
  const interval = pickInterval(timeline);
  if (!interval) return [];
  const htfBars = timeline.htf_bars[interval];
  if (!htfBars?.length) return [];

  const cursorTs = timeline.bars[currentBarIdx]?.timestamp_ns;
  if (cursorTs == null) return [];

  // Drop HTF bars whose close lies after the LTF cursor — no future leak.
  let visibleEnd = -1;
  for (let i = 0; i < htfBars.length; i++) {
    if (htfBars[i].timestamp_ns <= cursorTs) visibleEnd = i;
    else break;
  }
  if (visibleEnd < 0) return [];

  const swings = detectSwings(htfBars, visibleEnd).map((s) => ({ ...s, interval }));
  // Keep most recent N (by HTF bar index) so the chart stays readable.
  swings.sort((a, b) => b.bar_idx - a.bar_idx);
  return swings.slice(0, maxLevels);
}

export const htfOverlayLayer: ChartLayer = {
  id: 'htf_overlay',
  name: 'HTF swings',
  swatch: HIGH_COLOR,
  defaultVisible: false,

  mount(ctx: LayerCtx): LayerHandle {
    let priceLines: IPriceLine[] = [];

    const clear = () => {
      for (const pl of priceLines) {
        try {
          ctx.primarySeries.removePriceLine(pl);
        } catch {
          // series already detached
        }
      }
      priceLines = [];
    };

    return {
      update(timeline, currentBarIdx) {
        clear();
        const levels = selectHtfLevels(timeline, currentBarIdx);
        for (const lvl of levels) {
          priceLines.push(
            ctx.primarySeries.createPriceLine({
              price: lvl.price,
              color: lvl.kind === 'high' ? HIGH_COLOR : LOW_COLOR,
              lineWidth: 1,
              lineStyle: LineStyle.Dashed,
              axisLabelVisible: true,
              title: `${lvl.interval} ${lvl.kind} #${lvl.bar_idx}`,
            }),
          );
        }
      },
      unmount() {
        clear();
      },
    };
  },
};
