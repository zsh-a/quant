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
  it('exposes core + S5 advanced layers in registration order', () => {
    expect(LAYERS.map((l) => l.id)).toEqual([
      'regime',
      'swings',
      'ema20',
      'ema200',
      'annotations',
      'pattern_shapes',
      'trades',
      'signals',
      'decisions',
      'fills',
      'channels',
      'stop_adj',
      'htf_overlay',
      'reasoning',
    ]);
  });

  it('default-visible favours the aggregated annotations layer over raw fills', () => {
    expect(DEFAULT_VISIBLE.has('annotations')).toBe(true);
    expect(DEFAULT_VISIBLE.has('pattern_shapes')).toBe(true);
    expect(DEFAULT_VISIBLE.has('trades')).toBe(true);
    expect(DEFAULT_VISIBLE.has('fills')).toBe(false);
    expect(DEFAULT_VISIBLE.has('ema200')).toBe(false);
    expect(DEFAULT_VISIBLE.has('htf_overlay')).toBe(false);
    expect(DEFAULT_VISIBLE.has('reasoning')).toBe(false);
    expect(DEFAULT_VISIBLE.has('ema20')).toBe(true);
    expect(DEFAULT_VISIBLE.has('regime')).toBe(true);
    expect(DEFAULT_VISIBLE.has('channels')).toBe(true);
    expect(DEFAULT_VISIBLE.has('stop_adj')).toBe(true);
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
