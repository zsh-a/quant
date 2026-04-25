import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  ColorType: { Solid: 0 },
  CandlestickSeries: { type: 'Candlestick' },
  LineSeries: { type: 'Line' },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
  createChart: vi.fn(),
}));

import { findHtfBarForLtf, pickDefaultInterval } from '../components/chart/HTFInset';
import { makeTimeline } from './fixtures';
import type { Bar } from '../types';

const NS = 1_000_000_000;

function bar(tsSec: number, price = 100): Bar {
  return {
    timestamp_ns: tsSec * NS,
    open: price,
    high: price + 1,
    low: price - 1,
    close: price,
    volume: 1,
  };
}

describe('HTFInset helpers', () => {
  it('finds the latest HTF bar at-or-before the LTF cursor', () => {
    const htf = [bar(100), bar(200), bar(300), bar(400)];
    expect(findHtfBarForLtf(htf, 250 * NS)?.timestamp_ns).toBe(200 * NS);
    expect(findHtfBarForLtf(htf, 100 * NS)?.timestamp_ns).toBe(100 * NS);
    expect(findHtfBarForLtf(htf, 50 * NS)).toBeNull();
    expect(findHtfBarForLtf([], 100 * NS)).toBeNull();
  });

  it('picks the first declared HTF interval when populated', () => {
    const tl = makeTimeline(3);
    tl.htf_intervals = ['1h', '4h'];
    tl.htf_bars = { '1h': [bar(100)], '4h': [bar(100)] };
    expect(pickDefaultInterval(tl)).toBe('1h');
  });

  it('falls back to any populated entry in htf_bars', () => {
    const tl = makeTimeline(3);
    tl.htf_intervals = [];
    tl.htf_bars = { '15m': [bar(100)] };
    expect(pickDefaultInterval(tl)).toBe('15m');
  });

  it('returns null when no HTF data is available', () => {
    const tl = makeTimeline(3);
    tl.htf_intervals = [];
    tl.htf_bars = {};
    expect(pickDefaultInterval(tl)).toBeNull();
    expect(pickDefaultInterval(null)).toBeNull();
  });
});
