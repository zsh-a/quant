/**
 * Pattern shapes layer — multi-bar Brooks structures: ``Trading Range``
 * rectangles and ``Wedge`` converging trendlines.
 *
 * Two shape kinds, each driven by data the backend already attaches to
 * the timeline (no extra HTTP calls):
 *
 * Trading Range
 *   Contiguous runs of bars where ``BarEvent.background.in_trading_range``
 *   is ``true`` form one rectangle. Top / bottom are the running max-high
 *   and min-low across the run, so the rectangle hugs the actual price
 *   action. The rectangle is rendered as two solid boundary
 *   ``LineSeries`` lines plus a small "TR" marker at its start — the
 *   lightweight-charts v5 line API can paint a translucent boundary, and
 *   leaving the interior unfilled keeps the candles legible.
 *
 * Wedge
 *   Each ``Signal`` whose ``pattern_type === 'wedge'`` carries
 *   ``meta.p1_idx`` / ``p2_idx`` / ``p3_idx`` — the three pivot bar
 *   indices the detector found. We draw two converging trendlines from
 *   ``p1_idx`` to the signal bar by connecting endpoint highs and lows.
 *   That's not a perfect geometric fit but it puts the wedge in the
 *   right place and lets the user see "the lines are squeezing toward
 *   the breakout bar", which is the important Brooks read.
 *
 * Future-info safety: every helper clamps to ``currentBarIdx`` so a
 * scrub-back hides shapes that haven't formed yet.
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
import type { BarEvent, SessionTimeline, Signal } from '../types';

const RANGE_TOP_COLOR = 'rgba(180, 180, 180, 0.9)';
const RANGE_BOT_COLOR = 'rgba(180, 180, 180, 0.9)';
const RANGE_LABEL_COLOR = '#9CA3AF';
const WEDGE_LONG_COLOR = 'rgba(38, 166, 154, 0.9)';
const WEDGE_SHORT_COLOR = 'rgba(239, 83, 80, 0.9)';
const WEDGE_LABEL_COLOR_LONG = '#26A69A';
const WEDGE_LABEL_COLOR_SHORT = '#EF5350';

export interface TradingRangeRegion {
  from_bar: number;
  to_bar: number;
  top: number;
  bottom: number;
}

export interface WedgeRegion {
  signal_bar: number;
  start_bar: number;
  side: 'long' | 'short';
  top_start: number;
  top_end: number;
  bottom_start: number;
  bottom_end: number;
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

function clampedEnd(timeline: SessionTimeline, currentBarIdx: number): number {
  return Math.min(currentBarIdx, timeline.bars.length - 1);
}

/**
 * Walk events in-order. A trading-range region is a maximal run of
 * consecutive events flagged ``in_trading_range``. Bars without an event
 * (e.g. the warm-up window) break the run.
 */
export function computeTradingRangeRegions(
  timeline: SessionTimeline,
  currentBarIdx: number,
): TradingRangeRegion[] {
  const out: TradingRangeRegion[] = [];
  const end = clampedEnd(timeline, currentBarIdx);
  if (end < 0) return out;

  let runStart: number | null = null;
  let runEnd = -1;

  const closeRun = () => {
    if (runStart === null) return;
    const region = bandFromBars(timeline, runStart, runEnd);
    if (region) out.push(region);
    runStart = null;
    runEnd = -1;
  };

  for (const ev of timeline.events) {
    if (ev.bar_idx > end) break;
    const inTR = ev.background?.in_trading_range === true;
    if (inTR) {
      if (runStart === null) {
        runStart = ev.bar_idx;
      } else if (ev.bar_idx !== runEnd + 1) {
        // Gap in events — close the prior run, start a new one.
        closeRun();
        runStart = ev.bar_idx;
      }
      runEnd = ev.bar_idx;
    } else {
      closeRun();
    }
  }
  closeRun();
  return out;
}

function bandFromBars(
  timeline: SessionTimeline,
  from: number,
  to: number,
): TradingRangeRegion | null {
  if (to < from) return null;
  let top = -Infinity;
  let bottom = Infinity;
  for (let i = from; i <= to; i++) {
    const bar = timeline.bars[i];
    if (!bar) continue;
    if (bar.high > top) top = bar.high;
    if (bar.low < bottom) bottom = bar.low;
  }
  if (!Number.isFinite(top) || !Number.isFinite(bottom)) return null;
  return { from_bar: from, to_bar: to, top, bottom };
}

function readMetaInt(sig: Signal, key: string): number | null {
  const meta = (sig as { meta?: Record<string, unknown> }).meta;
  if (!meta) return null;
  const v = meta[key];
  return typeof v === 'number' && Number.isFinite(v) ? Math.floor(v) : null;
}

/**
 * Pull every wedge signal off the timeline and resolve its converging
 * trendline endpoints. Signals that lack pivot metadata fall back to a
 * fixed 12-bar lookback so the user still sees something.
 */
export function computeWedgeRegions(
  timeline: SessionTimeline,
  currentBarIdx: number,
): WedgeRegion[] {
  const end = clampedEnd(timeline, currentBarIdx);
  if (end < 0) return [];
  const out: WedgeRegion[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > end) break;
    const sigs = ev.signals ?? [];
    for (const sig of sigs) {
      const ptype = (sig.pattern_type ?? '').toString().toLowerCase();
      if (ptype !== 'wedge') continue;
      const region = wedgeRegionFromSignal(timeline, ev, sig, end);
      if (region) out.push(region);
    }
  }
  return out;
}

function wedgeRegionFromSignal(
  timeline: SessionTimeline,
  ev: BarEvent,
  sig: Signal,
  end: number,
): WedgeRegion | null {
  const sigBar = Math.min(ev.bar_idx, end);
  if (sigBar < 0 || sigBar >= timeline.bars.length) return null;

  const p1 = readMetaInt(sig, 'p1_idx');
  const fallbackStart = Math.max(0, sigBar - 12);
  const start = p1 !== null && p1 >= 0 && p1 < sigBar ? p1 : fallbackStart;
  if (start >= sigBar) return null;

  const startBar = timeline.bars[start];
  const endBar = timeline.bars[sigBar];
  if (!startBar || !endBar) return null;

  // Endpoint highs/lows = the sweep extremes within the early- and
  // late-half of the wedge window. Splitting at the midpoint gives the
  // converging-lines effect; using whole-window extremes would flatten
  // both lines into horizontals.
  const mid = Math.floor((start + sigBar) / 2);
  const earlyHi = maxHighIn(timeline, start, mid);
  const earlyLo = minLowIn(timeline, start, mid);
  const lateHi = maxHighIn(timeline, mid + 1, sigBar);
  const lateLo = minLowIn(timeline, mid + 1, sigBar);

  return {
    signal_bar: sigBar,
    start_bar: start,
    side: sig.side === 'short' ? 'short' : 'long',
    top_start: earlyHi ?? startBar.high,
    top_end: lateHi ?? endBar.high,
    bottom_start: earlyLo ?? startBar.low,
    bottom_end: lateLo ?? endBar.low,
  };
}

function maxHighIn(timeline: SessionTimeline, from: number, to: number): number | null {
  let v = -Infinity;
  for (let i = from; i <= to; i++) {
    const b = timeline.bars[i];
    if (b && b.high > v) v = b.high;
  }
  return Number.isFinite(v) ? v : null;
}

function minLowIn(timeline: SessionTimeline, from: number, to: number): number | null {
  let v = Infinity;
  for (let i = from; i <= to; i++) {
    const b = timeline.bars[i];
    if (b && b.low < v) v = b.low;
  }
  return Number.isFinite(v) ? v : null;
}

export function buildTradingRangeBoundaries(
  timeline: SessionTimeline,
  regions: TradingRangeRegion[],
  which: 'top' | 'bottom',
): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  for (const region of regions) {
    const startT = timeForBar(timeline, region.from_bar);
    const endT = timeForBar(timeline, region.to_bar);
    if (startT === null || endT === null) continue;
    const value = which === 'top' ? region.top : region.bottom;
    out.push({ time: startT, value });
    out.push({ time: endT, value });
  }
  // lightweight-charts requires strictly ascending time inside one
  // LineSeries; if two regions touch at the same timestamp, drop the
  // duplicate point.
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return dedupeMonotonic(out);
}

export function buildWedgeLine(
  timeline: SessionTimeline,
  regions: WedgeRegion[],
  which: 'top' | 'bottom',
): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  for (const region of regions) {
    const startT = timeForBar(timeline, region.start_bar);
    const endT = timeForBar(timeline, region.signal_bar);
    if (startT === null || endT === null) continue;
    const startVal = which === 'top' ? region.top_start : region.bottom_start;
    const endVal = which === 'top' ? region.top_end : region.bottom_end;
    out.push({ time: startT, value: startVal });
    out.push({ time: endT, value: endVal });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return dedupeMonotonic(out);
}

function dedupeMonotonic(
  data: LineData<UTCTimestamp>[],
): LineData<UTCTimestamp>[] {
  const out: LineData<UTCTimestamp>[] = [];
  let lastT: number | null = null;
  for (const point of data) {
    const t = point.time as number;
    if (lastT !== null && t === lastT) {
      out[out.length - 1] = point;
      continue;
    }
    out.push(point);
    lastT = t;
  }
  return out;
}

export function buildPatternShapeMarkers(
  timeline: SessionTimeline,
  ranges: TradingRangeRegion[],
  wedges: WedgeRegion[],
): SeriesMarker<Time>[] {
  const markers: SeriesMarker<Time>[] = [];

  for (const region of ranges) {
    const t = timeForBar(timeline, region.from_bar);
    if (t === null) continue;
    markers.push({
      time: t as Time,
      position: 'aboveBar',
      shape: 'square',
      color: RANGE_LABEL_COLOR,
      text: 'Trading range',
      size: 0,
      id: `tr-${region.from_bar}-${region.to_bar}`,
    });
  }

  for (const region of wedges) {
    const t = timeForBar(timeline, region.signal_bar);
    if (t === null) continue;
    const long = region.side === 'long';
    markers.push({
      time: t as Time,
      position: long ? 'belowBar' : 'aboveBar',
      shape: 'square',
      color: long ? WEDGE_LABEL_COLOR_LONG : WEDGE_LABEL_COLOR_SHORT,
      text: 'Wedge',
      size: 0,
      id: `wedge-${region.signal_bar}-${region.start_bar}`,
    });
  }

  markers.sort((a, b) => (a.time as number) - (b.time as number));
  return markers;
}

export const patternShapesLayer: ChartLayer = {
  id: 'pattern_shapes',
  name: 'Pattern shapes',
  swatch: RANGE_TOP_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    const { chart } = ctx;
    const baseOpts = {
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    } as const;

    let rangeTop: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      ...baseOpts,
      color: RANGE_TOP_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
    });
    let rangeBot: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      ...baseOpts,
      color: RANGE_BOT_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Dashed,
    });
    let wedgeTopLong: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      ...baseOpts,
      color: WEDGE_LONG_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
    });
    let wedgeBotLong: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      ...baseOpts,
      color: WEDGE_LONG_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
    });
    let wedgeTopShort: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      ...baseOpts,
      color: WEDGE_SHORT_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
    });
    let wedgeBotShort: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      ...baseOpts,
      color: WEDGE_SHORT_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
    });
    let labelPlugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );

    return {
      update(timeline, currentBarIdx) {
        if (
          !rangeTop ||
          !rangeBot ||
          !wedgeTopLong ||
          !wedgeBotLong ||
          !wedgeTopShort ||
          !wedgeBotShort
        )
          return;
        const ranges = computeTradingRangeRegions(timeline, currentBarIdx);
        const wedges = computeWedgeRegions(timeline, currentBarIdx);

        rangeTop.setData(buildTradingRangeBoundaries(timeline, ranges, 'top'));
        rangeBot.setData(buildTradingRangeBoundaries(timeline, ranges, 'bottom'));

        const longWedges = wedges.filter((w) => w.side === 'long');
        const shortWedges = wedges.filter((w) => w.side === 'short');
        wedgeTopLong.setData(buildWedgeLine(timeline, longWedges, 'top'));
        wedgeBotLong.setData(buildWedgeLine(timeline, longWedges, 'bottom'));
        wedgeTopShort.setData(buildWedgeLine(timeline, shortWedges, 'top'));
        wedgeBotShort.setData(buildWedgeLine(timeline, shortWedges, 'bottom'));

        if (labelPlugin) {
          labelPlugin.setMarkers(buildPatternShapeMarkers(timeline, ranges, wedges));
        }
      },
      unmount() {
        try {
          if (rangeTop) chart.removeSeries(rangeTop);
          if (rangeBot) chart.removeSeries(rangeBot);
          if (wedgeTopLong) chart.removeSeries(wedgeTopLong);
          if (wedgeBotLong) chart.removeSeries(wedgeBotLong);
          if (wedgeTopShort) chart.removeSeries(wedgeTopShort);
          if (wedgeBotShort) chart.removeSeries(wedgeBotShort);
          labelPlugin?.detach();
        } catch {
          // chart already disposed
        }
        rangeTop = null;
        rangeBot = null;
        wedgeTopLong = null;
        wedgeBotLong = null;
        wedgeTopShort = null;
        wedgeBotShort = null;
        labelPlugin = null;
      },
    };
  },
};
