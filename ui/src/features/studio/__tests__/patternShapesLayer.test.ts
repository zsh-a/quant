import { describe, expect, it, vi } from 'vitest';

const markersByPlugin = new Map<unknown, unknown[]>();
const detached: unknown[] = [];

vi.mock('lightweight-charts', () => ({
  LineSeries: { type: 'Line' },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createSeriesMarkers: vi.fn((series: unknown, initial: unknown[] = []) => {
    const handle = {
      _series: series,
      setMarkers: vi.fn((m: unknown[]) => markersByPlugin.set(handle, m)),
      detach: vi.fn(() => detached.push(handle)),
    };
    markersByPlugin.set(handle, initial);
    return handle;
  }),
}));

import {
  buildPatternShapeMarkers,
  buildTradingRangeBoundaries,
  buildWedgeLine,
  computeTradingRangeRegions,
  computeWedgeRegions,
  patternShapesLayer,
} from '../layers/pattern_shapes';
import { makeLayerCtx } from './mockChart';
import { makeSignal, makeTimeline } from './fixtures';

describe('pattern_shapes layer — trading range', () => {
  it('groups consecutive in_trading_range bars into a single region', () => {
    const tl = makeTimeline(6);
    for (let i = 1; i <= 4; i++) {
      tl.events[i].background = { in_trading_range: true };
    }
    const regions = computeTradingRangeRegions(tl, 5);
    expect(regions.length).toBe(1);
    expect(regions[0]).toMatchObject({ from_bar: 1, to_bar: 4 });
    // top = max(high) over bars 1..4 = bars[i].high = 102..105 → 105
    expect(regions[0].top).toBe(105);
    // bottom = min(low) over bars 1..4 = 100..103 → 100
    expect(regions[0].bottom).toBe(100);
  });

  it('splits non-contiguous trading-range runs into multiple regions', () => {
    const tl = makeTimeline(8);
    tl.events[1].background = { in_trading_range: true };
    tl.events[2].background = { in_trading_range: true };
    tl.events[5].background = { in_trading_range: true };
    tl.events[6].background = { in_trading_range: true };
    const regions = computeTradingRangeRegions(tl, 7);
    expect(regions.map((r) => [r.from_bar, r.to_bar])).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });

  it('clamps the trailing region to currentBarIdx', () => {
    const tl = makeTimeline(6);
    for (let i = 1; i <= 5; i++) tl.events[i].background = { in_trading_range: true };
    const regions = computeTradingRangeRegions(tl, 3);
    expect(regions.length).toBe(1);
    expect(regions[0].to_bar).toBe(3);
  });

  it('builds 2-point line data per region for top / bottom boundaries', () => {
    const tl = makeTimeline(5);
    tl.events[1].background = { in_trading_range: true };
    tl.events[2].background = { in_trading_range: true };
    tl.events[3].background = { in_trading_range: true };
    const regions = computeTradingRangeRegions(tl, 4);
    const top = buildTradingRangeBoundaries(tl, regions, 'top');
    const bot = buildTradingRangeBoundaries(tl, regions, 'bottom');
    expect(top.length).toBe(2);
    expect(bot.length).toBe(2);
    expect(top[0].value).toBe(top[1].value);
    expect(bot[0].value).toBe(bot[1].value);
    expect(top[0].value).toBeGreaterThan(bot[0].value);
  });
});

describe('pattern_shapes layer — wedge', () => {
  it('extracts wedge regions from signals carrying p1_idx metadata', () => {
    const tl = makeTimeline(12);
    tl.events[10].signals = [
      makeSignal({
        id: 'w',
        pattern: 'wedge_long',
        pattern_type: 'wedge',
        side: 'long',
        meta: { p1_idx: 2, p2_idx: 5, p3_idx: 8 },
      }),
    ];
    const regions = computeWedgeRegions(tl, 11);
    expect(regions.length).toBe(1);
    expect(regions[0]).toMatchObject({
      signal_bar: 10,
      start_bar: 2,
      side: 'long',
    });
    expect(regions[0].top_start).toBeGreaterThanOrEqual(regions[0].bottom_start);
    expect(regions[0].top_end).toBeGreaterThanOrEqual(regions[0].bottom_end);
  });

  it('falls back to a 12-bar lookback when p1_idx is missing', () => {
    const tl = makeTimeline(20);
    tl.events[15].signals = [
      makeSignal({ id: 'w', pattern_type: 'wedge', side: 'short' }),
    ];
    const regions = computeWedgeRegions(tl, 19);
    expect(regions.length).toBe(1);
    expect(regions[0].start_bar).toBe(3);
    expect(regions[0].side).toBe('short');
  });

  it('skips wedge signals whose bar is past currentBarIdx', () => {
    const tl = makeTimeline(20);
    tl.events[18].signals = [
      makeSignal({
        id: 'w',
        pattern_type: 'wedge',
        side: 'long',
        meta: { p1_idx: 8 },
      }),
    ];
    expect(computeWedgeRegions(tl, 12).length).toBe(0);
    expect(computeWedgeRegions(tl, 18).length).toBe(1);
  });

  it('skips non-wedge signals', () => {
    const tl = makeTimeline(10);
    tl.events[5].signals = [
      makeSignal({ id: 's', pattern_type: 'pullback', side: 'long' }),
    ];
    expect(computeWedgeRegions(tl, 9).length).toBe(0);
  });

  it('buildWedgeLine emits 2 points per region for top / bottom', () => {
    const tl = makeTimeline(20);
    tl.events[15].signals = [
      makeSignal({
        id: 'w',
        pattern_type: 'wedge',
        side: 'long',
        meta: { p1_idx: 5 },
      }),
    ];
    const regions = computeWedgeRegions(tl, 19);
    const top = buildWedgeLine(tl, regions, 'top');
    const bot = buildWedgeLine(tl, regions, 'bottom');
    expect(top.length).toBe(2);
    expect(bot.length).toBe(2);
    expect((top[1].time as number) > (top[0].time as number)).toBe(true);
  });
});

describe('pattern_shapes layer — markers + lifecycle', () => {
  it('builds one TR / Wedge label marker per region', () => {
    const tl = makeTimeline(10);
    tl.events[1].background = { in_trading_range: true };
    tl.events[2].background = { in_trading_range: true };
    tl.events[5].signals = [
      makeSignal({
        id: 'w',
        pattern_type: 'wedge',
        side: 'long',
        meta: { p1_idx: 0 },
      }),
    ];
    const ranges = computeTradingRangeRegions(tl, 9);
    const wedges = computeWedgeRegions(tl, 9);
    const markers = buildPatternShapeMarkers(tl, ranges, wedges);
    expect(markers.length).toBe(2);
    const texts = markers.map((m) => m.text);
    expect(texts).toContain('Trading range');
    expect(texts).toContain('Wedge');
  });

  it('mounts series + marker plugin and pushes data on update', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = patternShapesLayer.mount(ctx);
    // 6 LineSeries — TR top/bot + 4 wedge sides (long top/bot + short top/bot)
    expect(chart.added.length).toBe(6);

    const tl = makeTimeline(10);
    tl.events[2].background = { in_trading_range: true };
    tl.events[3].background = { in_trading_range: true };
    tl.events[8].signals = [
      makeSignal({
        id: 'w',
        pattern_type: 'wedge',
        side: 'long',
        meta: { p1_idx: 4 },
      }),
    ];
    handle.update(tl, 9);
    expect(chart.added[0].data.length).toBe(2); // TR top
    expect(chart.added[1].data.length).toBe(2); // TR bot
    expect(chart.added[2].data.length).toBe(2); // wedge long top
    expect(chart.added[3].data.length).toBe(2); // wedge long bot
    // No short wedges in this fixture.
    expect(chart.added[4].data.length).toBe(0);
    expect(chart.added[5].data.length).toBe(0);

    handle.unmount();
    expect(chart.removed.length).toBe(6);
    expect(detached.length).toBeGreaterThan(0);
  });
});
