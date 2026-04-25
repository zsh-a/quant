import { describe, expect, it, vi } from 'vitest';

const markersByPlugin = new Map<unknown, unknown[]>();
const detached: unknown[] = [];

vi.mock('lightweight-charts', () => ({
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

import { buildSignalMarkers, collectSignals, signalColor, signalsLayer } from '../layers/signals';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';
import type { Signal } from '../types';

describe('signals layer', () => {
  it('maps side to color', () => {
    expect(signalColor('long')).toBe('#26A69A');
    expect(signalColor('short')).toBe('#EF5350');
  });

  it('hides signals from future bars (no info leak)', () => {
    const tl = makeTimeline(4);
    tl.events[1].signals = [{ id: 'a', pattern: 'bar1', side: 'long' } as Signal];
    tl.events[3].signals = [{ id: 'b', pattern: 'bar3', side: 'short' } as Signal];

    expect(collectSignals(tl, 2).map((s) => s.signal.id)).toEqual(['a']);
    expect(collectSignals(tl, 3).map((s) => s.signal.id)).toEqual(['a', 'b']);
  });

  it('emits one marker per past signal', () => {
    const tl = makeTimeline(3);
    tl.events[1].signals = [
      { id: 'a', pattern: 'p1', side: 'long' } as Signal,
      { id: 'b', pattern: 'p2', side: 'short' } as Signal,
    ];
    expect(buildSignalMarkers(tl, 2).length).toBe(2);
  });

  it('mounts and draws entry/stop priceLines for the current bar signal', () => {
    const { ctx, primarySeries } = makeLayerCtx();
    const handle = signalsLayer.mount(ctx);

    const tl = makeTimeline(3);
    tl.events[2].signals = [
      { id: 'a', pattern: 'wedge', side: 'long', entry_px: 105, stop_px: 100 } as Signal,
    ];
    handle.update(tl, 2);

    expect(primarySeries.priceLines.length).toBe(2);
    expect(primarySeries.priceLines.map((p) => p.opts.price).sort()).toEqual([100, 105]);

    // Scrub away from bar 2 — those lines should clear.
    handle.update(tl, 1);
    expect(primarySeries.priceLines.length).toBe(0);

    handle.unmount();
    expect(detached.length).toBeGreaterThan(0);
  });
});
