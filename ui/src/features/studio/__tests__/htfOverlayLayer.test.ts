import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
}));

import {
  HTF_OVERLAY_MAX_LEVELS,
  detectSwings,
  htfOverlayLayer,
  selectHtfLevels,
} from '../layers/htf_overlay';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';
import type { Bar } from '../types';

const NS = 1_000_000_000;

function htfBars(prices: number[]): Bar[] {
  return prices.map((p, i) => ({
    timestamp_ns: (1_700_000_000 + i * 3600) * NS,
    open: p,
    high: p + 1,
    low: p - 1,
    close: p,
    volume: 1,
  }));
}

describe('htf overlay layer', () => {
  it('finds local highs/lows with a 2-bar lookback', () => {
    const bars = htfBars([10, 12, 14, 12, 10, 11, 13, 11, 9]);
    const swings = detectSwings(bars, bars.length - 1);
    const idxs = swings.map((s) => `${s.kind}@${s.bar_idx}`);
    // Expect a high near idx 2 and a low near idx 4.
    expect(idxs).toContain('high@2');
    expect(idxs).toContain('low@4');
  });

  it('selects up to N most-recent levels and never leaks future bars', () => {
    const tl = makeTimeline(10);
    tl.htf_intervals = ['1h'];
    // The HTF bars are spaced wider than the LTF bars (1h vs 5m); the cursor at
    // ltf bar 5 corresponds to roughly htf bar 0 in terms of timestamps. We
    // seed enough HTF history to detect swings before the cursor.
    const htfTs = (i: number) =>
      tl.bars[0].timestamp_ns - 6 * 3600 * NS + i * 3600 * NS;
    tl.htf_bars['1h'] = [10, 12, 14, 12, 10, 11, 13, 11, 9].map((p, i) => ({
      timestamp_ns: htfTs(i),
      open: p,
      high: p + 1,
      low: p - 1,
      close: p,
      volume: 1,
    }));
    const levels = selectHtfLevels(tl, 5, HTF_OVERLAY_MAX_LEVELS);
    expect(levels.length).toBeGreaterThan(0);
    expect(levels.length).toBeLessThanOrEqual(HTF_OVERLAY_MAX_LEVELS);
    for (const lvl of levels) {
      expect(tl.htf_bars['1h'][lvl.bar_idx].timestamp_ns).toBeLessThanOrEqual(
        tl.bars[5].timestamp_ns,
      );
    }
  });

  it('adds and clears price lines on the primary series', () => {
    const { ctx, primarySeries } = makeLayerCtx();
    const handle = htfOverlayLayer.mount(ctx);

    const tl = makeTimeline(10);
    tl.htf_intervals = ['1h'];
    tl.htf_bars['1h'] = [10, 12, 14, 12, 10, 11, 13, 11, 9].map((p, i) => ({
      timestamp_ns: tl.bars[0].timestamp_ns - 6 * 3600 * NS + i * 3600 * NS,
      open: p,
      high: p + 1,
      low: p - 1,
      close: p,
      volume: 1,
    }));

    handle.update(tl, 5);
    expect(primarySeries.priceLines.length).toBeGreaterThan(0);

    // Clear path: an empty htf_bars set produces zero lines.
    const empty = makeTimeline(3);
    handle.update(empty, 2);
    expect(primarySeries.priceLines.length).toBe(0);

    handle.unmount();
  });
});
