/**
 * PnLStrip — bottom 60px equity-R sub-chart, time-axis aligned with the main
 * chart's timeline.
 *
 * Each point in `timeline.pnl_curve` maps to a `{time, value}` entry on a
 * `LineSeries`; the entry's `time` is the timestamp of the matching bar in
 * `timeline.bars` so that a horizontal scroll of the main chart and a scrub
 * of `currentBarIdx` both align with the strip. A vertical cursor line marks
 * the current bar and updates as the user drags the scrubber.
 */

import { useEffect, useMemo, useRef } from 'react';
import {
  ColorType,
  LineSeries,
  LineStyle,
  createChart,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type UTCTimestamp,
} from 'lightweight-charts';
import {
  useEffectiveBarIdx,
  useTimelineState,
} from '../../store';
import type { SessionTimeline } from '../../types';

const THEME = {
  background: '#0e1116',
  text: '#9ca3af',
  grid: '#1c2230',
  line: '#60a5fa',
  cursor: '#e5e7eb',
} as const;

const STRIP_HEIGHT = 60;

export function buildPnLLineData(timeline: SessionTimeline | null): LineData<UTCTimestamp>[] {
  if (!timeline) return [];
  const out: LineData<UTCTimestamp>[] = [];
  const seen = new Set<number>();
  for (const point of timeline.pnl_curve) {
    const bar = timeline.bars[point.bar_idx];
    if (!bar) continue;
    const t = Math.floor(bar.timestamp_ns / 1_000_000_000);
    if (seen.has(t)) continue;
    seen.add(t);
    out.push({ time: t as UTCTimestamp, value: point.equity_r });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export function PnLStrip() {
  const timeline = useTimelineState();
  const currentBarIdx = useEffectiveBarIdx();

  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Line'> | null>(null);
  const cursorRef = useRef<ISeriesApi<'Line'> | null>(null);

  const data = useMemo(() => buildPnLLineData(timeline), [timeline]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: THEME.background },
        textColor: THEME.text,
        fontSize: 10,
      },
      grid: { vertLines: { visible: false }, horzLines: { color: THEME.grid } },
      rightPriceScale: { borderColor: THEME.grid },
      timeScale: { borderColor: THEME.grid, timeVisible: true, secondsVisible: false },
      handleScroll: false,
      handleScale: false,
      autoSize: false,
      width: container.clientWidth,
      height: STRIP_HEIGHT,
    });
    const line = chart.addSeries(LineSeries, {
      color: THEME.line,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: true,
      crosshairMarkerVisible: false,
    });
    const cursor = chart.addSeries(LineSeries, {
      color: THEME.cursor,
      lineWidth: 1,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    chartRef.current = chart;
    seriesRef.current = line;
    cursorRef.current = cursor;

    const ro = new ResizeObserver(() => {
      chart.resize(container.clientWidth, STRIP_HEIGHT);
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
      seriesRef.current = null;
      cursorRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Push line data.
  useEffect(() => {
    seriesRef.current?.setData(data);
    if (data.length > 0) chartRef.current?.timeScale().fitContent();
  }, [data]);

  // Cursor line (drawn the same way as in HTFInset: two near-touching points
  // spanning the visible value range).
  useEffect(() => {
    const cursor = cursorRef.current;
    if (!cursor) return;
    if (!timeline || data.length === 0 || currentBarIdx < 0) {
      cursor.setData([]);
      return;
    }
    const bar = timeline.bars[currentBarIdx];
    if (!bar) {
      cursor.setData([]);
      return;
    }
    const cursorTime = Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (const p of data) {
      if (p.value < lo) lo = p.value;
      if (p.value > hi) hi = p.value;
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi) || lo === hi) {
      // Avoid degenerate line by widening the range a touch.
      lo = lo - 0.001;
      hi = hi + 0.001;
    }
    cursor.setData([
      { time: cursorTime, value: lo },
      { time: ((cursorTime as number) + 1) as UTCTimestamp, value: hi },
    ]);
  }, [timeline, currentBarIdx, data]);

  if (!timeline || data.length === 0) {
    return (
      <div
        className="flex h-[60px] items-center justify-center px-3 text-[11px] text-muted-foreground"
        data-testid="pnl-strip-empty"
      >
        No PnL data
      </div>
    );
  }

  return (
    <div
      ref={containerRef}
      className="h-[60px] w-full bg-[#0e1116]"
      data-testid="pnl-strip"
    />
  );
}
