/**
 * Tiny mock of the lightweight-charts API surface that ChartLayer
 * implementations use. jsdom can't render canvas, so we just record calls
 * to the methods our layers actually invoke and let tests assert on them.
 */

import { vi } from 'vitest';
import type { LayerCtx } from '../layers/types';

export interface MockSeries {
  type: string;
  options: Record<string, unknown>;
  data: unknown[];
  setData: ReturnType<typeof vi.fn>;
  update: ReturnType<typeof vi.fn>;
  setMarkers: ReturnType<typeof vi.fn>;
  markers: unknown[];
  priceLines: { id: number; opts: Record<string, unknown> }[];
  createPriceLine: ReturnType<typeof vi.fn>;
  removePriceLine: ReturnType<typeof vi.fn>;
}

export interface MockChart {
  added: MockSeries[];
  removed: MockSeries[];
  priceScales: Record<string, Record<string, unknown>>;
  addSeries: ReturnType<typeof vi.fn>;
  removeSeries: ReturnType<typeof vi.fn>;
  priceScale: ReturnType<typeof vi.fn>;
}

function makeSeries(type: string, options: Record<string, unknown>): MockSeries {
  let plId = 0;
  const series: MockSeries = {
    type,
    options,
    data: [],
    setData: vi.fn((d: unknown[]) => {
      series.data = d;
    }),
    update: vi.fn(),
    setMarkers: vi.fn((markers: unknown[]) => {
      series.markers = markers;
    }),
    markers: [],
    priceLines: [],
    createPriceLine: vi.fn((opts: Record<string, unknown>) => {
      const handle = { id: ++plId, opts };
      series.priceLines.push(handle);
      return handle;
    }),
    removePriceLine: vi.fn((handle: { id: number }) => {
      series.priceLines = series.priceLines.filter((p) => p.id !== handle.id);
    }),
  };
  return series;
}

export function createMockChart(): MockChart {
  const added: MockSeries[] = [];
  const removed: MockSeries[] = [];
  const priceScales: Record<string, Record<string, unknown>> = {};

  const chart: MockChart = {
    added,
    removed,
    priceScales,
    addSeries: vi.fn((definition: unknown, options: Record<string, unknown> = {}) => {
      const def = definition as { type?: string; isBuiltIn?: boolean };
      const series = makeSeries(def.type ?? 'unknown', options);
      added.push(series);
      return series;
    }),
    removeSeries: vi.fn((s: MockSeries) => {
      removed.push(s);
    }),
    priceScale: vi.fn((id: string) => ({
      applyOptions: (opts: Record<string, unknown>) => {
        priceScales[id] = { ...(priceScales[id] ?? {}), ...opts };
      },
    })),
  };
  return chart;
}

export function makeLayerCtx(): { ctx: LayerCtx; chart: MockChart; primarySeries: MockSeries } {
  const chart = createMockChart();
  const primarySeries = makeSeries('Candlestick', {});
  // primarySeries is created externally in real code; record-only for tests.
  const ctx: LayerCtx = {
    // Cast: our mock implements the exact slice of the API our layers touch.
    chart: chart as unknown as LayerCtx['chart'],
    primarySeries: primarySeries as unknown as LayerCtx['primarySeries'],
    theme: 'dark',
  };
  return { ctx, chart, primarySeries };
}
