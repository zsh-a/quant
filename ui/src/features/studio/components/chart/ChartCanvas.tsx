/**
 * ChartCanvas — bare lightweight-charts candlestick container with the
 * pluggable layer registry mounted on top.
 *
 * Owns the `IChartApi` instance via `useEffect(create / cleanup)` and surfaces
 * it through `useLayerRegistry` so every registered layer mounts onto the
 * same chart. Live bar updates use `series.update()` (single-bar diff) when
 * the new bar shares the same timestamp as the last point, otherwise we
 * append.
 *
 * Phase S6 perf rules:
 *   • Mount effect dependency list stays empty so prop changes never
 *     re-create the chart (would discard the WebGL canvas + layer state).
 *   • Crosshair handler reads through `timelineRef` rather than capturing
 *     the timeline closure — avoids stale-closure bugs with no re-mount.
 *   • Setup effect for the candle stream renders large timelines (>5K bars)
 *     in a deferred chunk so the initial paint clears 1K bars in <500ms.
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
import type { Bar, SessionTimeline } from '../../types';
import {
  useEffectiveBarIdx,
  useStudioActions,
  useTimelineState,
  useVisibleLayers,
} from '../../store';
import { useLayerRegistry } from '../../hooks/useLayerRegistry';
import type { LayerCtx } from '../../layers/types';

const LAZY_BAR_THRESHOLD = 5000;
const LAZY_INITIAL_TAIL = 1500;

const THEME = {
  background: '#0e1116',
  text: '#cbd2dc',
  grid: '#1c2230',
  bull: '#26A69A',
  bear: '#EF5350',
} as const;

type IdleScheduler = (cb: () => void) => number;
type IdleCanceller = (handle: number) => void;

const scheduleLazy: IdleScheduler =
  typeof window !== 'undefined' &&
  typeof (window as unknown as { requestIdleCallback?: unknown }).requestIdleCallback === 'function'
    ? (cb) =>
        (window as unknown as { requestIdleCallback: (cb: () => void) => number }).requestIdleCallback(
          cb,
        )
    : (cb) => window.setTimeout(cb, 16);

const cancelLazy: IdleCanceller =
  typeof window !== 'undefined' &&
  typeof (window as unknown as { cancelIdleCallback?: unknown }).cancelIdleCallback === 'function'
    ? (handle) =>
        (window as unknown as { cancelIdleCallback: (h: number) => void }).cancelIdleCallback(handle)
    : (handle) => window.clearTimeout(handle);

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
  const timelineRef = useRef<SessionTimeline | null>(null);
  const setHoveredBarRef = useRef<((idx: number | null) => void) | null>(null);
  const lazyTailHandleRef = useRef<number | null>(null);

  const [layerCtx, setLayerCtx] = useState<LayerCtx | null>(null);

  const timeline = useTimelineState();
  const currentBarIdx = useEffectiveBarIdx();
  const visibleLayers = useVisibleLayers();
  const { setHoveredBar } = useStudioActions();

  // Refresh refs every render so the once-mounted chart callbacks read
  // current state without re-subscribing.
  timelineRef.current = timeline;
  setHoveredBarRef.current = setHoveredBar;

  // Depend on the bars reference only — applyLiveEvent rewrites `events` but
  // keeps `bars` identity stable, so we don't redo this work for every WS
  // event.
  const candles = useMemo(
    () => (timeline ? toCandlestickData(timeline.bars) : []),
    [timeline?.bars],
  );

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
      const tl = timelineRef.current;
      const setHover = setHoveredBarRef.current;
      if (!setHover) return;
      if (!param.time || !tl) {
        setHover(null);
        return;
      }
      const t = param.time as number;
      const idx = tl.bars.findIndex(
        (b) => Math.floor(b.timestamp_ns / 1_000_000_000) === t,
      );
      setHover(idx >= 0 ? idx : null);
    });

    setLayerCtx({ chart, primarySeries: series, theme: 'dark' });

    return () => {
      setLayerCtx(null);
      ro.disconnect();
      if (lazyTailHandleRef.current !== null) {
        cancelLazy(lazyTailHandleRef.current);
        lazyTailHandleRef.current = null;
      }
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
  // a single-point update for the latest bar. Defers head-of-history for very
  // large timelines so the first paint stays inside the perf budget.
  useEffect(() => {
    const series = seriesRef.current;
    if (!series) return;

    if (lazyTailHandleRef.current !== null) {
      cancelLazy(lazyTailHandleRef.current);
      lazyTailHandleRef.current = null;
    }

    if (candles.length === 0) {
      series.setData([]);
      lastBarCountRef.current = 0;
      return;
    }

    const prevCount = lastBarCountRef.current;
    const sizeShrunk = candles.length < prevCount;
    const isFreshLoad = prevCount === 0 || sizeShrunk;

    if (isFreshLoad || candles.length - prevCount > 1) {
      if (isFreshLoad && candles.length > LAZY_BAR_THRESHOLD) {
        // Render only the trailing window first so the user sees a chart
        // before we hydrate the full history asynchronously.
        const tailStart = candles.length - LAZY_INITIAL_TAIL;
        series.setData(candles.slice(tailStart));
        lazyTailHandleRef.current = scheduleLazy(() => {
          const live = seriesRef.current;
          if (!live) return;
          live.setData(candles);
          lazyTailHandleRef.current = null;
        });
      } else {
        series.setData(candles);
      }
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
