import { describe, expect, it, vi } from 'vitest';

const markerHandles: { setMarkers: ReturnType<typeof vi.fn>; detach: ReturnType<typeof vi.fn> }[] = [];

vi.mock('lightweight-charts', () => ({
  LineSeries: { type: 'Line' },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createSeriesMarkers: vi.fn(() => {
    const handle = { setMarkers: vi.fn(), detach: vi.fn() };
    markerHandles.push(handle);
    return handle;
  }),
}));

import {
  buildStopAdjMarkers,
  buildStopLadderData,
  collectStopAdjustments,
  stopAdjLayer,
} from '../layers/stop_adj';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';

describe('stop adjustment layer', () => {
  it('collects only adjustments at-or-before the cursor', () => {
    const tl = makeTimeline(6);
    tl.events[1].stop_adj = { from_px: 100, to_px: 99, reason: 'init' };
    tl.events[3].stop_adj = { from_px: 99, to_px: 101, reason: 'be' };
    tl.events[5].stop_adj = { from_px: 101, to_px: 103, reason: 'trail' };
    expect(collectStopAdjustments(tl, 4).map((p) => p.bar_idx)).toEqual([1, 3]);
    expect(collectStopAdjustments(tl, 5).length).toBe(3);
  });

  it('builds a step-line from first adjustment to current bar', () => {
    const tl = makeTimeline(6);
    tl.events[1].stop_adj = { from_px: 100, to_px: 99, reason: 'init' };
    tl.events[3].stop_adj = { from_px: 99, to_px: 101, reason: 'be' };
    const data = buildStopLadderData(tl, 4);
    // Bars 1..4 → 4 entries; values: 99 (after step), 99, 101 (after step), 101
    expect(data.length).toBe(4);
    expect(data.map((p) => p.value)).toEqual([99, 99, 101, 101]);
  });

  it('emits one marker per adjustment', () => {
    const tl = makeTimeline(6);
    tl.events[1].stop_adj = { from_px: 100, to_px: 99, reason: 'init' };
    tl.events[3].stop_adj = { from_px: 99, to_px: 101, reason: 'be' };
    expect(buildStopAdjMarkers(tl, 5).length).toBe(2);
  });

  it('mounts a series + markers plugin and tears them down', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = stopAdjLayer.mount(ctx);
    expect(chart.added.length).toBe(1); // ladder series
    expect(markerHandles.length).toBeGreaterThan(0);

    const tl = makeTimeline(3);
    tl.events[1].stop_adj = { from_px: 100, to_px: 99, reason: 'init' };
    handle.update(tl, 2);
    expect(chart.added[0].data.length).toBe(2); // bars 1..2

    handle.unmount();
    expect(chart.removed.length).toBe(1);
  });
});
