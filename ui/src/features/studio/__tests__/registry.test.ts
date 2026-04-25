import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  HistogramSeries: { type: 'Histogram', isBuiltIn: true, defaultOptions: {} },
  LineSeries: { type: 'Line', isBuiltIn: true, defaultOptions: {} },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createSeriesMarkers: vi.fn(() => ({
    setMarkers: vi.fn(),
    detach: vi.fn(),
  })),
}));

import { DEFAULT_VISIBLE, LAYERS, findLayer } from '../layers/registry';

describe('layer registry', () => {
  it('exposes the six core layers (with EMA split)', () => {
    expect(LAYERS.map((l) => l.id)).toEqual([
      'regime',
      'swings',
      'ema20',
      'ema200',
      'signals',
      'decisions',
      'fills',
    ]);
  });

  it('default-visible omits ema200', () => {
    expect(DEFAULT_VISIBLE.has('ema200')).toBe(false);
    expect(DEFAULT_VISIBLE.has('ema20')).toBe(true);
    expect(DEFAULT_VISIBLE.has('regime')).toBe(true);
  });

  it('layer ids are unique', () => {
    const ids = LAYERS.map((l) => l.id);
    expect(new Set(ids).size).toBe(ids.length);
  });

  it('findLayer resolves by id', () => {
    expect(findLayer('regime')?.name).toBe('Regime band');
    expect(findLayer('does-not-exist')).toBeUndefined();
  });
});
