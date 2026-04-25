import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  LineSeries: { type: 'Line', isBuiltIn: true, defaultOptions: {} },
}));

import { buildEmaSeriesData, ema20Layer, ema200Layer, readEmaField } from '../layers/ema';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';

describe('ema layer', () => {
  it('reads numeric ema fields and skips junk', () => {
    expect(readEmaField({ ema20: 100.5 } as never, 'ema20')).toBe(100.5);
    expect(readEmaField({ ema20: 'oops' } as never, 'ema20')).toBeNull();
    expect(readEmaField(null, 'ema20')).toBeNull();
    expect(readEmaField(undefined, 'ema20')).toBeNull();
  });

  it('skips bars without an ema value rather than dropping the whole series', () => {
    const tl = makeTimeline(4);
    (tl.events[0].features as unknown as Record<string, number>) = { ema20: 100 } as never;
    (tl.events[2].features as unknown as Record<string, number>) = { ema20: 102 } as never;
    // events[1] and events[3] left without ema20 — should not appear

    const data = buildEmaSeriesData(tl, 'ema20');
    expect(data.map((d) => d.value)).toEqual([100, 102]);
  });

  it('mounts ema20 by default and ema200 hidden by default', () => {
    expect(ema20Layer.defaultVisible).toBe(true);
    expect(ema200Layer.defaultVisible).toBe(false);
  });

  it('writes line data on update and removes the series on unmount', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = ema20Layer.mount(ctx);
    expect(chart.added.length).toBe(1);
    expect(chart.added[0].type).toBe('Line');

    const tl = makeTimeline(2);
    (tl.events[0].features as unknown as Record<string, number>) = { ema20: 100 } as never;
    (tl.events[1].features as unknown as Record<string, number>) = { ema20: 101 } as never;
    handle.update(tl, 1);
    expect(chart.added[0].data).toHaveLength(2);

    handle.unmount();
    expect(chart.removed).toContain(chart.added[0]);
  });
});
