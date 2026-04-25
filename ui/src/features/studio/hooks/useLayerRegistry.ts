/**
 * Layer registry stub for Phase S3.
 *
 * S2 ships only the bare candle chart. The registry plug point exists now
 * so `ChartCanvas` doesn't need re-shaping later: S3 will populate the
 * `LAYERS` array and pass each layer's `mount()` through this hook.
 */

import { useEffect } from 'react';
import type { IChartApi, ISeriesApi } from 'lightweight-charts';
import type { SessionTimeline } from '../types';

export interface LayerCtx {
  chart: IChartApi;
  primarySeries: ISeriesApi<'Candlestick'>;
  theme: 'dark' | 'light';
}

export interface LayerHandle {
  update: (timeline: SessionTimeline, currentBarIdx: number) => void;
  unmount: () => void;
}

export interface ChartLayer {
  id: string;
  name: string;
  defaultVisible: boolean;
  mount: (ctx: LayerCtx) => LayerHandle;
}

/**
 * S2 ships an empty registry. S3 will fill this in with regime / swings /
 * ema / signals / decisions / fills layers.
 */
export const LAYERS: readonly ChartLayer[] = [];

/**
 * Mount any registered layers onto the given chart instance and update them
 * whenever timeline / currentBarIdx change. Currently a no-op (empty
 * registry); kept here so `ChartCanvas` can call it unconditionally.
 */
export function useLayerRegistry(
  ctx: LayerCtx | null,
  timeline: SessionTimeline | null,
  currentBarIdx: number,
) {
  useEffect(() => {
    if (!ctx) return;
    const handles = LAYERS.map((layer) => ({ layer, handle: layer.mount(ctx) }));
    return () => {
      for (const { handle } of handles) handle.unmount();
    };
  }, [ctx]);

  useEffect(() => {
    if (!ctx || !timeline) return;
    // S3: iterate through visible layers and call .update().
  }, [ctx, timeline, currentBarIdx]);
}
