import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  LineSeries: { type: 'Line' },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
}));

import {
  buildTradeLegs,
  tradeColor,
  tradeLegLineData,
  tradeStopLineData,
  tradesLayer,
} from '../layers/trades';
import { makeLayerCtx } from './mockChart';
import { makeDecision, makeTimeline } from './fixtures';

describe('trades layer', () => {
  it('pairs an opening fill with the next closing fill into a closed leg', () => {
    const tl = makeTimeline(6);
    tl.events[1].decision = makeDecision({ side: 'long', stop_px: 90, entry_px: 100 });
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[4].fill = { side: 'sell', qty: 1, price: 105, reason: 'tp' };

    const legs = buildTradeLegs(tl, 5);
    expect(legs.length).toBe(1);
    expect(legs[0]).toMatchObject({
      status: 'closed',
      dir: 'long',
      entry: { bar_idx: 1, price: 100 },
      exit: { bar_idx: 4, price: 105 },
      stop_px: 90,
      pnl: 5,
    });
  });

  it('marks short legs and computes signed pnl correctly', () => {
    const tl = makeTimeline(5);
    tl.events[0].decision = makeDecision({ side: 'short', stop_px: 110 });
    tl.events[0].fill = { side: 'sell_short', qty: 1, price: 100, reason: 'entry' };
    tl.events[3].fill = { side: 'buy_to_cover', qty: 1, price: 95, reason: 'tp' };
    const [leg] = buildTradeLegs(tl, 4);
    expect(leg.dir).toBe('short');
    expect(leg.pnl).toBe(5); // 100 - 95
    expect(tradeColor(leg)).toContain('38, 166, 154'); // profit green
  });

  it('keeps an unrealised "open" leg from entry to cursor close price', () => {
    const tl = makeTimeline(6);
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    // bar 4 close = 100.5 + 4 = 104.5 (per fixtures formula)
    const legs = buildTradeLegs(tl, 4);
    expect(legs.length).toBe(1);
    expect(legs[0].status).toBe('open');
    expect(legs[0].exit.bar_idx).toBe(4);
    expect(legs[0].exit.price).toBeCloseTo(104.5);
  });

  it('does not extend an open leg past the cursor (no info leak)', () => {
    const tl = makeTimeline(6);
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[5].fill = { side: 'sell', qty: 1, price: 110, reason: 'tp' };
    // cursor before the exit fill — leg should still look open.
    const legs = buildTradeLegs(tl, 3);
    expect(legs.length).toBe(1);
    expect(legs[0].status).toBe('open');
    expect(legs[0].exit.bar_idx).toBe(3);
  });

  it('builds two-point line data from entry and exit, with optional stop', () => {
    const tl = makeTimeline(5);
    tl.events[0].decision = makeDecision({ side: 'long', stop_px: 95 });
    tl.events[0].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[3].fill = { side: 'sell', qty: 1, price: 102, reason: 'tp' };
    const [leg] = buildTradeLegs(tl, 4);
    const body = tradeLegLineData(tl, leg);
    const stop = tradeStopLineData(tl, leg);
    expect(body).toHaveLength(2);
    expect(body[0].value).toBe(100);
    expect(body[1].value).toBe(102);
    expect(stop).toHaveLength(2);
    expect(stop[0].value).toBe(95);
    expect(stop[1].value).toBe(95);
  });

  it('returns no stop data when no decision precedes the entry fill', () => {
    const tl = makeTimeline(4);
    tl.events[0].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[2].fill = { side: 'sell', qty: 1, price: 105, reason: 'tp' };
    const [leg] = buildTradeLegs(tl, 3);
    expect(leg.stop_px).toBeNull();
    expect(tradeStopLineData(tl, leg)).toEqual([]);
  });

  it('mounts, draws, and tears down line series for each trade leg', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = tradesLayer.mount(ctx);

    const tl = makeTimeline(6);
    tl.events[1].decision = makeDecision({ side: 'long', stop_px: 95 });
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[4].fill = { side: 'sell', qty: 1, price: 105, reason: 'tp' };
    handle.update(tl, 5);
    // body + stop series for the single leg
    expect(chart.added.length).toBe(2);
    expect(chart.added[0].data.length).toBe(2);
    expect(chart.added[1].data.length).toBe(2);

    // Now scrub before the entry — leg disappears, series shrunk.
    handle.update(tl, 0);
    expect(chart.removed.length).toBe(2);

    handle.unmount();
  });
});
