import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  HistogramSeries: { type: 'Histogram', isBuiltIn: true, defaultOptions: {} },
}));

import { regimeColorOf, regimeLayer } from '../layers/regime';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';

describe('regime layer', () => {
  it('has a stable id and a default-on toggle', () => {
    expect(regimeLayer.id).toBe('regime');
    expect(regimeLayer.defaultVisible).toBe(true);
  });

  it('returns the configured color for known regimes and falls back otherwise', () => {
    expect(regimeColorOf('strong_bull_trend')).toContain('rgba(38, 166, 154');
    expect(regimeColorOf(undefined)).toBeTypeOf('string');
    expect(regimeColorOf('something-unknown')).toBeTypeOf('string');
  });

  it('mounts a histogram overlay and writes a colored band per bar', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = regimeLayer.mount(ctx);
    expect(chart.added.length).toBe(1);
    expect(chart.added[0].type).toBe('Histogram');
    expect(chart.added[0].options).toMatchObject({
      priceScaleId: 'regime-band',
      priceLineVisible: false,
      lastValueVisible: false,
    });

    const tl = makeTimeline(3);
    tl.events[0].regime = { name: 'strong_bull_trend', confidence: 0.9, reasons: [] };
    tl.events[2].regime = { name: 'climax', confidence: 0.7, reasons: [] };

    handle.update(tl, 2);
    const data = chart.added[0].data as { color: string }[];
    expect(data).toHaveLength(3);
    expect(data[0].color).toContain('rgba(38, 166, 154');
    expect(data[2].color).toContain('rgba(255, 152, 0');
  });

  it('removes its series on unmount', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = regimeLayer.mount(ctx);
    handle.unmount();
    expect(chart.removed.length).toBe(1);
  });
});
