import { describe, expect, it, vi } from 'vitest';

const markersByPlugin = new Map<unknown, unknown[]>();

vi.mock('lightweight-charts', () => ({
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createSeriesMarkers: vi.fn((series: unknown, initial: unknown[] = []) => {
    const handle = {
      _series: series,
      setMarkers: vi.fn((m: unknown[]) => markersByPlugin.set(handle, m)),
      detach: vi.fn(),
    };
    markersByPlugin.set(handle, initial);
    return handle;
  }),
}));

import {
  buildDecisionMarkers,
  collectDecisions,
  decisionColor,
  decisionsInWindow,
  decisionsLayer,
} from '../layers/decisions';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';
import type { Decision } from '../types';

const longDecision = (): Decision => ({
  side: 'long',
  entry_px: 100,
  stop_px: 95,
  target_px: 110,
  quantity: 1,
  probability: 0.6,
  expected_r: 1.5,
  regime: 'strong_bull_trend',
  pattern: 'BO',
  source: 'rule',
});

describe('decisions layer', () => {
  it('maps side to colour', () => {
    expect(decisionColor('long')).toBe('#26A69A');
    expect(decisionColor('short')).toBe('#EF5350');
  });

  it('collects decisions only up to currentBarIdx', () => {
    const tl = makeTimeline(5);
    tl.events[1].decision = longDecision();
    tl.events[3].decision = longDecision();
    expect(collectDecisions(tl, 2).length).toBe(1);
    expect(collectDecisions(tl, 4).length).toBe(2);
  });

  it('emits markers for each visible decision', () => {
    const tl = makeTimeline(3);
    tl.events[0].decision = longDecision();
    tl.events[2].decision = longDecision();
    expect(buildDecisionMarkers(tl, 2).length).toBe(2);
  });

  it('only includes decisions within ± window for priceLine windowing', () => {
    const tl = makeTimeline(20);
    tl.events[1].decision = longDecision();
    tl.events[10].decision = longDecision();
    tl.events[15].decision = longDecision();
    // currentBarIdx = 12, window 5 → only bar 10 (15 is past 12 → excluded)
    const inWin = decisionsInWindow(tl, 12, 5);
    expect(inWin.map((d) => d.bar_idx)).toEqual([10]);
  });

  it('draws entry/stop/target priceLines for in-window decisions and clears them on scrub', () => {
    const { ctx, primarySeries } = makeLayerCtx();
    const handle = decisionsLayer.mount(ctx);

    const tl = makeTimeline(10);
    tl.events[5].decision = longDecision();

    handle.update(tl, 5);
    // entry + stop + target = 3 lines
    expect(primarySeries.priceLines.length).toBe(3);

    // far away → window excludes it
    handle.update(tl, 50); // currentBarIdx clamped doesn't matter; we passed raw
    expect(primarySeries.priceLines.length).toBe(0);

    handle.unmount();
  });

  it('skips target priceLine when target is null', () => {
    const { ctx, primarySeries } = makeLayerCtx();
    const handle = decisionsLayer.mount(ctx);

    const tl = makeTimeline(3);
    tl.events[1].decision = { ...longDecision(), target_px: null };
    handle.update(tl, 1);
    expect(primarySeries.priceLines.length).toBe(2); // entry + stop only

    handle.unmount();
  });
});
