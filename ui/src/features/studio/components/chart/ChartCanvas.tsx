/**
 * ChartCanvas — bare lightweight-charts candlestick container with the
 * pluggable layer registry mounted on top.
 *
 * Owns the `IChartApi` instance via `useEffect(create / cleanup)` and surfaces
 * it through `useLayerRegistry` so every registered layer mounts onto the
 * same chart. Live bar updates use `series.update()` (single-bar diff) when
 * the new bar shares the same timestamp as the last point, otherwise we
 * append.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ColorType,
  CandlestickSeries,
  CrosshairMode,
  createChart,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { Bar } from '../../types';
import {
  useEffectiveBarIdx,
  useStudioActions,
  useTimelineState,
  useVisibleLayers,
} from '../../store';
import { useLayerRegistry } from '../../hooks/useLayerRegistry';
import type { LayerCtx } from '../../layers/types';

const THEME = {
  background: '#0e1116',
  text: '#cbd2dc',
  grid: '#1c2230',
  bull: '#26A69A',
  bear: '#EF5350',
} as const;

function toCandlestickData(bars: Bar[]): CandlestickData<UTCTimestamp>[] {
  const seen = new Set<number>();
  const out: CandlestickData<UTCTimestamp>[] = [];
  for (const b of bars) {
    const t = Math.floor(b.timestamp_ns / 1_000_000_000);
    // lightweight-charts requires strictly ascending unique timestamps;
    // drop duplicates rather than throwing.
    if (seen.has(t)) continue;
    seen.add(t);
    out.push({ time: t as UTCTimestamp, open: b.open, high: b.high, low: b.low, close: b.close });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export function ChartCanvas() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const lastBarCountRef = useRef(0);
  const fittedRef = useRef(false);

  const [layerCtx, setLayerCtx] = useState<LayerCtx | null>(null);

  const timeline = useTimelineState();
  const currentBarIdx = useEffectiveBarIdx();
  const visibleLayers = useVisibleLayers();
  const { setHoveredBar } = useStudioActions();

  const candles = useMemo(() => (timeline ? toCandlestickData(timeline.bars) : []), [timeline]);

  useLayerRegistry(layerCtx, timeline, currentBarIdx, visibleLayers);

  // Mount chart once.
  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: THEME.background },
        textColor: THEME.text,
        fontSize: 12,
      },
      grid: {
        vertLines: { color: THEME.grid },
        horzLines: { color: THEME.grid },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: THEME.grid },
      timeScale: { borderColor: THEME.grid, timeVisible: true, secondsVisible: false },
      autoSize: false,
      width: container.clientWidth,
      height: container.clientHeight,
    });
    const series = chart.addSeries(CandlestickSeries, {
      upColor: THEME.bull,
      downColor: THEME.bear,
      wickUpColor: THEME.bull,
      wickDownColor: THEME.bear,
      borderVisible: false,
    });
    chartRef.current = chart;
    seriesRef.current = series;

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        chart.resize(width, height);
      }
    });
    ro.observe(container);

    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !timeline) {
        setHoveredBar(null);
        return;
      }
      const t = param.time as number;
      const idx = timeline.bars.findIndex(
        (b) => Math.floor(b.timestamp_ns / 1_000_000_000) === t,
      );
      setHoveredBar(idx >= 0 ? idx : null);
    });

    setLayerCtx({ chart, primarySeries: series, theme: 'dark' });

    return () => {
      setLayerCtx(null);
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      lastBarCountRef.current = 0;
      fittedRef.current = false;
    };
    // We intentionally mount the chart once; parent prop changes don't
    // re-create it (would lose the WebGL canvas + layer state).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push data updates: full setData on first load / changed length, otherwise
  // a single-point update for the latest bar.
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

    if (candles.length === 0) {
      series.setData([]);
      lastBarCountRef.current = 0;
      return;
    }

    const prevCount = lastBarCountRef.current;
    const sizeShrunk = candles.length < prevCount;
    if (prevCount === 0 || sizeShrunk || candles.length - prevCount > 1) {
      series.setData(candles);
    } else if (candles.length === prevCount) {
      series.update(candles[candles.length - 1]);
    } else {
      // appended exactly one bar
      series.update(candles[candles.length - 1]);
    }

    lastBarCountRef.current = candles.length;

    if (!fittedRef.current && candles.length > 0) {
      chartRef.current?.timeScale().fitContent();
      fittedRef.current = true;
    }
  }, [candles]);

  // Scrub the visible range when the user moves currentBarIdx far from view.
  useEffect(() => {
    if (currentBarIdx < 0) return;
    const series = seriesRef.current;
    const chart = chartRef.current;
    if (!series || !chart || candles.length === 0) return;
    const target = candles[Math.min(currentBarIdx, candles.length - 1)];
    if (!target) return;
    chart.timeScale().scrollToPosition(0, false);
    // Soft auto-pan: ensure target bar is visible.
    chart.timeScale().scrollToRealTime();
  }, [currentBarIdx, candles]);

  return (
    <div
      ref={containerRef}
      className="h-full w-full bg-[#0e1116]"
      data-testid="studio-chart-canvas"
    />
  );
}
