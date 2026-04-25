import { describe, expect, it, vi } from 'vitest';

const markersByPlugin = new Map<unknown, unknown[]>();
const detached: unknown[] = [];

vi.mock('lightweight-charts', () => ({
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

import { buildFillMarkers, fillsLayer, isBuyFill } from '../layers/fills';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';

describe('fills layer', () => {
  it('classifies buy/cover as buy and sell/short as sell', () => {
    expect(isBuyFill('buy')).toBe(true);
    expect(isBuyFill('buy_to_cover')).toBe(true);
    expect(isBuyFill('sell')).toBe(false);
    expect(isBuyFill('sell_short')).toBe(false);
  });

  it('emits one marker per past fill, none for future bars', () => {
    const tl = makeTimeline(4);
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[3].fill = { side: 'sell', qty: 1, price: 105, reason: 'tp' };

    expect(buildFillMarkers(tl, 2).length).toBe(1);
    expect(buildFillMarkers(tl, 3).length).toBe(2);
  });

  it('mounts, updates markers, and detaches on unmount', () => {
    const { ctx } = makeLayerCtx();
    const handle = fillsLayer.mount(ctx);

    const tl = makeTimeline(2);
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    handle.update(tl, 1);

    const lastMarkers = [...markersByPlugin.values()].pop() as unknown[];
    expect(lastMarkers.length).toBe(1);

    handle.unmount();
    expect(detached.length).toBeGreaterThan(0);
  });
});
