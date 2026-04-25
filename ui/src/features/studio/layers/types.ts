/**
 * ChartLayer protocol — every overlay (regime band, swings, ema, signals,
 * decisions, fills, …) implements this so they can be mounted, updated, and
 * unmounted uniformly. New layers plug in by exporting one and adding it to
 * `LAYERS` in `./registry.ts`.
 */

import type { IChartApi, ISeriesApi } from 'lightweight-charts';
import type { SessionTimeline } from '../types';

export interface LayerCtx {
  chart: IChartApi;
  primarySeries: ISeriesApi<'Candlestick'>;
  htfSeries?: ISeriesApi<'Candlestick'>;
  theme: 'dark' | 'light';
}

export interface LayerHandle {
  /** Re-render based on full timeline + cursor; layer decides diff strategy. */
  update(timeline: SessionTimeline, currentBarIdx: number): void;
  unmount(): void;
}

export interface ChartLayer {
  id: string;
  name: string;
  /** Small color sample for the toggle panel (CSS color). */
  swatch: string;
  defaultVisible: boolean;
  mount(ctx: LayerCtx): LayerHandle;
}
