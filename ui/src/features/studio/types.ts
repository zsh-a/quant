/**
 * Brooks Studio — frontend types mirroring `src/api/schemas/brooks_studio.py`.
 *
 * Keep this file in sync with the Pydantic schema. Optional fields use
 * `?: T | null` because Pydantic emits `null` for absent values rather than
 * dropping the key.
 */

export interface Bar {
  timestamp_ns: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RegimeView {
  name: string;
  confidence: number;
  reasons: string[];
}

export interface FeaturesView {
  is_bull: boolean;
  body_pct: number;
  close_position: 'high' | 'mid' | 'low';
  ema_relation: 'above' | 'at' | 'below';
  leg_dir: 'up' | 'down' | 'flat';
  leg_length: number;
  is_doji: boolean;
  is_inside_bar: boolean;
}

export interface SwingPoint {
  idx: number;
  kind: string;
  price: number;
}

export interface ChannelLine {
  slope: number;
  intercept: number;
  start: number;
  end: number;
}

export interface StructureView {
  always_in: 'long' | 'short' | 'neutral';
  confirmed_swings: SwingPoint[];
  micro_channel_top?: ChannelLine | null;
  micro_channel_bot?: ChannelLine | null;
  last_breakout_lookback_high?: number | null;
  last_breakout_lookback_low?: number | null;
}

/**
 * Loose mirror of `src.brooks.schema.Signal` — frontend treats it as
 * mostly opaque metadata for hover/inspector panels.
 */
export interface Signal {
  id?: string;
  bar_idx?: number;
  pattern?: string;
  side?: 'long' | 'short';
  source?: string;
  [key: string]: unknown;
}

export interface Decision {
  side: 'long' | 'short';
  entry_px: number;
  stop_px: number;
  target_px?: number | null;
  quantity: number;
  probability: number;
  expected_r: number;
  regime: string;
  htf_aligned?: boolean;
  pattern: string;
  source: string;
  reasoning?: string;
  [key: string]: unknown;
}

export interface FillView {
  side: 'buy' | 'sell' | 'buy_to_cover' | 'sell_short';
  qty: number;
  price: number;
  reason: string;
}

export interface StopAdj {
  from_px: number;
  to_px: number;
  reason: string;
}

export interface HTFView {
  regime?: string | null;
  always_in?: string | null;
  last_swing_idx?: number | null;
}

export interface BarEvent {
  bar_idx: number;
  timestamp_ns: number;
  regime?: RegimeView | null;
  features?: FeaturesView | null;
  structure?: StructureView | null;
  signals?: Signal[];
  decision?: Decision | null;
  fill?: FillView | null;
  stop_adj?: StopAdj | null;
  pnl_r?: number | null;
  htf?: Record<string, HTFView>;
}

export interface PnLPoint {
  bar_idx: number;
  equity_r: number;
}

export interface SessionTimeline {
  session_id: string;
  symbol: string;
  base_interval: string;
  htf_intervals: string[];
  bars: Bar[];
  htf_bars: Record<string, Bar[]>;
  events: BarEvent[];
  pnl_curve: PnLPoint[];
  config: Record<string, unknown>;
  created_at: string;
}

export type StudioMode = 'live' | 'replay';
export type PlayState = 'paused' | 'playing';
export type StudioSpeed = 1 | 2 | 5 | 10;

export const STUDIO_SPEEDS: readonly StudioSpeed[] = [1, 2, 5, 10] as const;
