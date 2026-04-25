import type { BarEvent, SessionTimeline } from '../types';

const NS = 1_000_000_000;

export function makeTimeline(barCount = 5): SessionTimeline {
  const t0 = 1_700_000_000;
  return {
    session_id: 'sess-1',
    symbol: 'BTC/USDT',
    base_interval: '5m',
    htf_intervals: [],
    bars: Array.from({ length: barCount }, (_, i) => ({
      timestamp_ns: (t0 + i * 300) * NS,
      open: 100 + i,
      high: 101 + i,
      low: 99 + i,
      close: 100.5 + i,
      volume: 10,
    })),
    htf_bars: {},
    events: Array.from({ length: barCount }, (_, i) => ({
      bar_idx: i,
      timestamp_ns: (t0 + i * 300) * NS,
      signals: [],
    })),
    pnl_curve: [],
    config: {},
    created_at: '2026-04-25T15:00:00Z',
  };
}

export function makeBarEvent(barIdx: number, overrides: Partial<BarEvent> = {}): BarEvent {
  return {
    bar_idx: barIdx,
    timestamp_ns: (1_700_000_000 + barIdx * 300) * NS,
    signals: [],
    ...overrides,
  };
}
