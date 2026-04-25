import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  ColorType: { Solid: 0 },
  LineSeries: { type: 'Line' },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createChart: vi.fn(),
}));

import { buildPnLLineData } from '../components/timeline/PnLStrip';
import { makeTimeline } from './fixtures';

describe('PnLStrip', () => {
  it('returns an empty list when timeline is null', () => {
    expect(buildPnLLineData(null)).toEqual([]);
  });

  it('skips points whose bar_idx has no matching bar', () => {
    const tl = makeTimeline(3);
    tl.pnl_curve = [
      { bar_idx: 0, equity_r: 0 },
      { bar_idx: 99, equity_r: 0.5 }, // no matching bar — skipped
      { bar_idx: 2, equity_r: 1.0 },
    ];
    const data = buildPnLLineData(tl);
    expect(data.length).toBe(2);
    expect(data[0].value).toBe(0);
    expect(data[1].value).toBe(1.0);
  });

  it('produces strictly ascending unique timestamps', () => {
    const tl = makeTimeline(4);
    tl.pnl_curve = [
      { bar_idx: 0, equity_r: 0 },
      { bar_idx: 1, equity_r: 0.1 },
      { bar_idx: 2, equity_r: 0.2 },
      { bar_idx: 3, equity_r: 0.3 },
    ];
    const data = buildPnLLineData(tl);
    for (let i = 1; i < data.length; i++) {
      expect((data[i].time as number) > (data[i - 1].time as number)).toBe(true);
    }
  });
});
