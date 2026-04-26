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

import { buildSwingMarkers, classifySwingLabel, isHighSwing, swingsLayer } from '../layers/swings';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';
import type { SwingPoint } from '../types';

describe('swings layer', () => {
  it('classifies high vs low by kind name', () => {
    expect(isHighSwing('hh')).toBe(false); // depends only on /high/
    expect(isHighSwing('higher_high')).toBe(true);
    expect(isHighSwing('low')).toBe(false);
  });

  it('classifies HH/HL/LH/LL based on prior same-kind swing prices', () => {
    const swings: SwingPoint[] = [
      { idx: 0, kind: 'low', price: 100 },   // L
      { idx: 1, kind: 'high', price: 110 },  // H
      { idx: 2, kind: 'low', price: 102 },   // HL (102 > 100)
      { idx: 3, kind: 'high', price: 115 },  // HH (115 > 110)
      { idx: 4, kind: 'high', price: 113 },  // LH (113 < 115)
      { idx: 5, kind: 'low', price: 99 },    // LL (99 < 102)
    ];
    expect(classifySwingLabel(swings, 0)).toBe('L');
    expect(classifySwingLabel(swings, 1)).toBe('H');
    expect(classifySwingLabel(swings, 2)).toBe('HL');
    expect(classifySwingLabel(swings, 3)).toBe('HH');
    expect(classifySwingLabel(swings, 4)).toBe('LH');
    expect(classifySwingLabel(swings, 5)).toBe('LL');
  });

  it('emits Brooks-style labels with sequence subscripts on recent swings', () => {
    const tl = makeTimeline(8);
    tl.events[7].structure = {
      always_in: 'long',
      confirmed_swings: [
        { idx: 0, kind: 'low', price: 100 },
        { idx: 1, kind: 'high', price: 110 },
        { idx: 2, kind: 'low', price: 102 },
        { idx: 3, kind: 'high', price: 115 },
      ],
    };
    const markers = buildSwingMarkers(tl, 7);
    const texts = markers.map((m) => m.text);
    expect(texts).toContain('L₁');
    expect(texts).toContain('H₁');
    expect(texts).toContain('HL₁');
    expect(texts).toContain('HH₁');
  });

  it('only emits markers for swings whose idx <= currentBarIdx', () => {
    const tl = makeTimeline(5);
    // bar 1: only the first swing is known
    tl.events[1].structure = {
      always_in: 'long',
      confirmed_swings: [{ idx: 1, kind: 'low', price: 99 }],
    };
    // bar 4: cumulative snapshot includes a future-leaking idx 4 entry
    tl.events[4].structure = {
      always_in: 'long',
      confirmed_swings: [
        { idx: 1, kind: 'low', price: 99 },
        { idx: 3, kind: 'high', price: 105 },
        { idx: 4, kind: 'high', price: 107 },
      ],
    };

    const past = buildSwingMarkers(tl, 2);
    expect(past.length).toBe(1);
    expect(past[0].id).toBe('swing-1-low');

    const present = buildSwingMarkers(tl, 4);
    expect(present.length).toBe(3);

    // scrubbing back to bar 3 takes the latest snapshot ≤ 3 (which is the
    // bar-1 snapshot), so we still see only the first swing — and the
    // future-idx swing from bar 4 is filtered even if present in scope.
    const between = buildSwingMarkers(tl, 3);
    expect(between.length).toBe(1);
  });

  it('mounts a marker plugin, updates markers, and detaches on unmount', () => {
    const { ctx } = makeLayerCtx();
    const handle = swingsLayer.mount(ctx);

    const tl = makeTimeline(3);
    tl.events[2].structure = {
      always_in: 'long',
      confirmed_swings: [
        { idx: 0, kind: 'low', price: 99 },
        { idx: 2, kind: 'high', price: 102 },
      ],
    };
    handle.update(tl, 2);
    const lastMarkers = [...markersByPlugin.values()].pop() as unknown[];
    expect(lastMarkers.length).toBe(2);

    handle.unmount();
    expect(detached.length).toBeGreaterThan(0);
  });
});
