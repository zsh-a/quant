/**
 * Trades layer — connect each entry fill to its corresponding exit fill with
 * a single line segment, so completed round-trips read as a trajectory across
 * bars rather than a pile of independent arrows.
 *
 * Pairing rule: walk fills in order; an opening fill (`buy` / `sell_short`)
 * starts a leg, the next opposite-side fill (`sell` / `buy_to_cover`) closes
 * it. A position that is still open at `currentBarIdx` is drawn as a
 * "running" leg from the entry bar to the cursor (using `currentBarIdx`'s
 * close price as the live unrealised endpoint).
 *
 * Colour: gain (entry vs exit, signed by side) → green; loss → red. Stop is
 * a dashed line at `decision.stop_px` over the same bar range so the user
 * can see how far the actual exit was from the planned stop.
 *
 * Future-info safety: pairs whose entry bar > cursor are dropped; the
 * unrealised leg only extends to the cursor, never past.
 */

import {
  LineSeries,
  LineStyle,
  type ISeriesApi,
  type LineData,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { Decision, FillView, SessionTimeline } from '../types';

const PROFIT_COLOR = 'rgba(38, 166, 154, 0.95)';
const LOSS_COLOR = 'rgba(239, 83, 80, 0.95)';
const OPEN_COLOR = 'rgba(255, 213, 79, 0.85)';
const STOP_COLOR = 'rgba(239, 83, 80, 0.55)';

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

function isOpeningFill(side: FillView['side']): boolean {
  return side === 'buy' || side === 'sell_short';
}

function isClosingFill(side: FillView['side']): boolean {
  return side === 'sell' || side === 'buy_to_cover';
}

export type TradeStatus = 'closed' | 'open';
export type TradeDir = 'long' | 'short';

export interface TradeLeg {
  status: TradeStatus;
  dir: TradeDir;
  entry: { bar_idx: number; price: number };
  /** Exit bar_idx + price for closed; for `open`, this is the cursor + last
   * close price (so the running PnL line tracks live). */
  exit: { bar_idx: number; price: number };
  /** Planned stop price recorded with the entry decision, if any. */
  stop_px?: number | null;
  /** Signed PnL (per unit). Positive = profit, negative = loss. */
  pnl: number;
}

function pnlFor(dir: TradeDir, entry: number, exit: number): number {
  return dir === 'long' ? exit - entry : entry - exit;
}

/**
 * Pair up fills into trade legs. The pairing is intentionally simple: one
 * opening fill, one closing fill, in order. Real strategies that scale in/out
 * across multiple fills will produce one leg per scaled clip — that is good
 * enough for visual replay; precise per-trade PnL still lives in the PnL
 * strip / SignalSidebar.
 */
export function buildTradeLegs(
  timeline: SessionTimeline,
  currentBarIdx: number,
): TradeLeg[] {
  const legs: TradeLeg[] = [];
  let openLeg: {
    dir: TradeDir;
    entryBar: number;
    entryPrice: number;
    stop?: number | null;
  } | null = null;

  // Track the most recent decision before/at each fill bar so we can read
  // its planned stop_px.
  const events = [...timeline.events].sort((a, b) => a.bar_idx - b.bar_idx);
  let lastDecision: Decision | null = null;

  for (const ev of events) {
    if (ev.bar_idx > currentBarIdx) break;
    if (ev.decision) lastDecision = ev.decision;
    if (!ev.fill) continue;

    if (openLeg && isClosingFill(ev.fill.side)) {
      legs.push({
        status: 'closed',
        dir: openLeg.dir,
        entry: { bar_idx: openLeg.entryBar, price: openLeg.entryPrice },
        exit: { bar_idx: ev.bar_idx, price: ev.fill.price },
        stop_px: openLeg.stop ?? null,
        pnl: pnlFor(openLeg.dir, openLeg.entryPrice, ev.fill.price),
      });
      openLeg = null;
      continue;
    }
    if (!openLeg && isOpeningFill(ev.fill.side)) {
      openLeg = {
        dir: ev.fill.side === 'buy' ? 'long' : 'short',
        entryBar: ev.bar_idx,
        entryPrice: ev.fill.price,
        stop: lastDecision?.stop_px ?? null,
      };
    }
  }

  if (openLeg) {
    const liveBar = timeline.bars[currentBarIdx];
    if (liveBar) {
      legs.push({
        status: 'open',
        dir: openLeg.dir,
        entry: { bar_idx: openLeg.entryBar, price: openLeg.entryPrice },
        exit: { bar_idx: currentBarIdx, price: liveBar.close },
        stop_px: openLeg.stop ?? null,
        pnl: pnlFor(openLeg.dir, openLeg.entryPrice, liveBar.close),
      });
    }
  }

  return legs;
}

export function tradeColor(leg: TradeLeg): string {
  if (leg.status === 'open') return OPEN_COLOR;
  return leg.pnl >= 0 ? PROFIT_COLOR : LOSS_COLOR;
}

/** Sample two endpoints (entry + exit) into a 2-point LineSeries datum. */
export function tradeLegLineData(
  timeline: SessionTimeline,
  leg: TradeLeg,
): LineData<UTCTimestamp>[] {
  const t0 = timeForBar(timeline, leg.entry.bar_idx);
  const t1 = timeForBar(timeline, leg.exit.bar_idx);
  if (t0 === null || t1 === null) return [];
  if (t1 <= t0) return [];
  return [
    { time: t0, value: leg.entry.price },
    { time: t1, value: leg.exit.price },
  ];
}

export function tradeStopLineData(
  timeline: SessionTimeline,
  leg: TradeLeg,
): LineData<UTCTimestamp>[] {
  if (leg.stop_px == null) return [];
  const t0 = timeForBar(timeline, leg.entry.bar_idx);
  const t1 = timeForBar(timeline, leg.exit.bar_idx);
  if (t0 === null || t1 === null) return [];
  if (t1 <= t0) return [];
  return [
    { time: t0, value: leg.stop_px },
    { time: t1, value: leg.stop_px },
  ];
}

interface TradeSeriesPair {
  body: ISeriesApi<'Line'>;
  stop: ISeriesApi<'Line'>;
}

export const tradesLayer: ChartLayer = {
  id: 'trades',
  name: 'Trade trajectory',
  swatch: PROFIT_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    const { chart } = ctx;
    let pool: TradeSeriesPair[] = [];

    const acquirePair = (idx: number): TradeSeriesPair => {
      while (pool.length <= idx) {
        pool.push({
          body: chart.addSeries(LineSeries, {
            color: PROFIT_COLOR,
            lineWidth: 2,
            lineStyle: LineStyle.Solid,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          }),
          stop: chart.addSeries(LineSeries, {
            color: STOP_COLOR,
            lineWidth: 1,
            lineStyle: LineStyle.Dashed,
            priceLineVisible: false,
            lastValueVisible: false,
            crosshairMarkerVisible: false,
          }),
        });
      }
      return pool[idx];
    };

    const removeSeries = (s: ISeriesApi<'Line'>): void => {
      try {
        chart.removeSeries(s);
      } catch {
        // chart already disposed
      }
    };

    return {
      update(timeline, currentBarIdx) {
        const legs = buildTradeLegs(timeline, currentBarIdx);

        // Resize pool to legs.length, dropping surplus series.
        for (let i = pool.length - 1; i >= legs.length; i--) {
          removeSeries(pool[i].body);
          removeSeries(pool[i].stop);
          pool.pop();
        }

        legs.forEach((leg, i) => {
          const pair = acquirePair(i);
          pair.body.applyOptions({
            color: tradeColor(leg),
            lineStyle: leg.status === 'open' ? LineStyle.Dotted : LineStyle.Solid,
          });
          pair.body.setData(tradeLegLineData(timeline, leg));
          pair.stop.setData(tradeStopLineData(timeline, leg));
        });
      },
      unmount() {
        for (const pair of pool) {
          removeSeries(pair.body);
          removeSeries(pair.stop);
        }
        pool = [];
      },
    };
  },
};
