/**
 * HTFInset — picture-in-picture HTF candlestick mini-chart anchored top-right
 * of the primary chart.
 *
 * The chart shows `timeline.htf_bars[interval]` for the user-selected HTF
 * interval (defaulted to the first listed in `timeline.htf_intervals`) and
 * highlights the HTF bar that contains the current LTF cursor with a red
 * vertical line. It can be:
 *   - collapsed to a small "1h ▾" pill (saving screen real estate);
 *   - switched between any HTF intervals the timeline ships with;
 *   - expanded into a full-screen modal with the same chart at large size.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CandlestickSeries,
  ColorType,
  LineSeries,
  LineStyle,
  createChart,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type LineData,
  type UTCTimestamp,
} from 'lightweight-charts';
import { Maximize2, Minimize2, Monitor, X } from 'lucide-react';
import { Button } from '../../../../components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from '../../../../components/ui/dialog';
import {
  useEffectiveBarIdx,
  useTimelineState,
} from '../../store';
import type { Bar, SessionTimeline } from '../../types';

const THEME = {
  background: '#0b0e13',
  text: '#cbd2dc',
  grid: '#1c2230',
  bull: '#26A69A',
  bear: '#EF5350',
  cursor: '#EF5350',
} as const;

const INSET_WIDTH = 240;
const INSET_HEIGHT = 120;

function toCandlesticks(bars: Bar[]): CandlestickData<UTCTimestamp>[] {
  const seen = new Set<number>();
  const out: CandlestickData<UTCTimestamp>[] = [];
  for (const b of bars) {
    const t = Math.floor(b.timestamp_ns / 1_000_000_000);
    if (seen.has(t)) continue;
    seen.add(t);
    out.push({ time: t as UTCTimestamp, open: b.open, high: b.high, low: b.low, close: b.close });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

/**
 * Find the HTF bar that contains the LTF bar's timestamp — the most recent
 * HTF bar with `timestamp_ns <= ltfTs`.
 */
export function findHtfBarForLtf(htfBars: Bar[], ltfTs: number): Bar | null {
  let chosen: Bar | null = null;
  for (const b of htfBars) {
    if (b.timestamp_ns <= ltfTs) chosen = b;
    else break;
  }
  return chosen;
}

/** Pick a sensible default interval: prefer `htf_intervals` then anything in `htf_bars`. */
export function pickDefaultInterval(timeline: SessionTimeline | null): string | null {
  if (!timeline) return null;
  if (timeline.htf_intervals.length > 0) {
    const first = timeline.htf_intervals[0];
    if (timeline.htf_bars[first]?.length) return first;
  }
  for (const k of Object.keys(timeline.htf_bars)) {
    if (timeline.htf_bars[k]?.length) return k;
  }
  return null;
}

interface InsetChartProps {
  bars: Bar[];
  cursorTime: UTCTimestamp | null;
  width: number;
  height: number;
}

function InsetChart({ bars, cursorTime, width, height }: InsetChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const cursorRef = useRef<ISeriesApi<'Line'> | null>(null);

  const candles = useMemo(() => toCandlesticks(bars), [bars]);

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
      width,
      height,
    });
    const candle = chart.addSeries(CandlestickSeries, {
      upColor: THEME.bull,
      downColor: THEME.bear,
      wickUpColor: THEME.bull,
      wickDownColor: THEME.bear,
      borderVisible: false,
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
    candleRef.current = candle;
    cursorRef.current = cursor;

    return () => {
      chart.remove();
      chartRef.current = null;
      candleRef.current = null;
      cursorRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Keep the chart sized to its container.
  useEffect(() => {
    chartRef.current?.resize(width, height);
  }, [width, height]);

  // Push candles.
  useEffect(() => {
    candleRef.current?.setData(candles);
    if (candles.length > 0) chartRef.current?.timeScale().fitContent();
  }, [candles]);

  // Update cursor (a vertical line by drawing two points at min/max price on the same time).
  useEffect(() => {
    const cursor = cursorRef.current;
    if (!cursor) return;
    if (cursorTime == null || candles.length === 0) {
      cursor.setData([]);
      return;
    }
    let lo = Number.POSITIVE_INFINITY;
    let hi = Number.NEGATIVE_INFINITY;
    for (const c of candles) {
      if (c.low < lo) lo = c.low;
      if (c.high > hi) hi = c.high;
    }
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) {
      cursor.setData([]);
      return;
    }
    // lightweight-charts requires two points with strictly distinct times to
    // draw a line; we cheat with a 1-second offset so it shows as a near-
    // vertical sliver on the screen.
    const data: LineData<UTCTimestamp>[] = [
      { time: cursorTime, value: lo },
      { time: ((cursorTime as number) + 1) as UTCTimestamp, value: hi },
    ];
    cursor.setData(data);
  }, [cursorTime, candles]);

  return (
    <div
      ref={containerRef}
      className="bg-[#0b0e13]"
      data-testid="htf-inset-chart"
      style={{ width, height }}
    />
  );
}

export function HTFInset() {
  const timeline = useTimelineState();
  const currentBarIdx = useEffectiveBarIdx();

  const defaultInterval = pickDefaultInterval(timeline);
  const [selected, setSelected] = useState<string | null>(defaultInterval);
  const [collapsed, setCollapsed] = useState(false);
  const [expanded, setExpanded] = useState(false);

  // Reset selection if timeline changes & current selection is stale.
  useEffect(() => {
    if (!timeline) return;
    if (selected && timeline.htf_bars[selected]?.length) return;
    setSelected(pickDefaultInterval(timeline));
    // intentionally one-shot per timeline identity
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [timeline]);

  if (!timeline) return null;
  const intervals = Object.keys(timeline.htf_bars).filter(
    (k) => timeline.htf_bars[k]?.length,
  );
  if (intervals.length === 0) return null;

  const interval = selected ?? intervals[0];
  const htfBars = timeline.htf_bars[interval] ?? [];

  const ltfTs = timeline.bars[currentBarIdx]?.timestamp_ns ?? null;
  const htfBar = ltfTs != null ? findHtfBarForLtf(htfBars, ltfTs) : null;
  const cursorTime = htfBar
    ? (Math.floor(htfBar.timestamp_ns / 1_000_000_000) as UTCTimestamp)
    : null;

  if (collapsed) {
    return (
      <div className="flex items-center gap-1 rounded-md border border-border/70 bg-card/70 p-1 text-xs shadow-sm">
        <button
          type="button"
          onClick={() => setCollapsed(false)}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
          title="Expand HTF inset"
          data-testid="htf-inset-expand"
        >
          <Monitor className="size-3.5" />
          <span className="tabular-nums">{interval}</span>
        </button>
      </div>
    );
  }

  return (
    <div
      className="flex flex-col gap-1 rounded-md border border-border/70 bg-card/70 p-1.5 text-xs shadow-sm"
      data-testid="htf-inset"
    >
      <div className="flex items-center justify-between gap-2 px-1">
        <div className="flex items-center gap-1.5 text-muted-foreground">
          <Monitor className="size-3.5" />
          <span>HTF</span>
        </div>
        <div className="flex items-center gap-0.5">
          <select
            aria-label="HTF interval"
            data-testid="htf-inset-select"
            className="rounded bg-secondary/40 px-1.5 py-0.5 text-[11px] text-foreground"
            value={interval}
            onChange={(e) => setSelected(e.target.value)}
          >
            {intervals.map((iv) => (
              <option key={iv} value={iv}>
                {iv}
              </option>
            ))}
          </select>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setExpanded(true)}
            title="Expand to fullscreen"
            aria-label="Expand HTF inset"
            data-testid="htf-inset-fullscreen"
            className="h-6 w-6"
          >
            <Maximize2 className="size-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setCollapsed(true)}
            title="Collapse"
            aria-label="Collapse HTF inset"
            data-testid="htf-inset-collapse"
            className="h-6 w-6"
          >
            <Minimize2 className="size-3" />
          </Button>
        </div>
      </div>
      <InsetChart
        bars={htfBars}
        cursorTime={cursorTime}
        width={INSET_WIDTH}
        height={INSET_HEIGHT}
      />
      <Dialog open={expanded} onOpenChange={setExpanded}>
        <DialogContent className="max-w-[80vw]">
          <DialogTitle className="flex items-center gap-2 text-sm">
            HTF {interval}
            <button
              type="button"
              onClick={() => setExpanded(false)}
              className="ml-auto rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground"
              aria-label="Close"
            >
              <X className="size-4" />
            </button>
          </DialogTitle>
          <InsetChart
            bars={htfBars}
            cursorTime={cursorTime}
            width={Math.min(window.innerWidth * 0.75, 1100)}
            height={Math.min(window.innerHeight * 0.6, 500)}
          />
        </DialogContent>
      </Dialog>
    </div>
  );
}
