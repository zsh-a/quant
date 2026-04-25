/**
 * Signals layer — circle markers for every detector signal up to (and
 * including) the current bar, plus entry / stop price lines for any signals
 * landing on the current bar.
 *
 * Future-info safety: signals from `bar_idx > currentBarIdx` are filtered out
 * before render, so scrubbing back hides them.
 */

import {
  LineStyle,
  createSeriesMarkers,
  type IPriceLine,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { SessionTimeline, Signal } from '../types';

const LONG_COLOR = '#26A69A';
const SHORT_COLOR = '#EF5350';
const NEUTRAL_COLOR = '#90A4AE';
const ENTRY_LINE_COLOR = '#FFD54F';
const STOP_LINE_COLOR = '#EF5350';

export function signalColor(side: Signal['side']): string {
  if (side === 'long') return LONG_COLOR;
  if (side === 'short') return SHORT_COLOR;
  return NEUTRAL_COLOR;
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

function readSignalPrice(sig: Signal, key: string): number | null {
  const v = (sig as Record<string, unknown>)[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export interface BuiltSignal {
  signal: Signal;
  bar_idx: number;
}

export function collectSignals(
  timeline: SessionTimeline,
  currentBarIdx: number,
): BuiltSignal[] {
  const out: BuiltSignal[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > currentBarIdx) continue;
    if (!ev.signals?.length) continue;
    for (const s of ev.signals) {
      out.push({ signal: s, bar_idx: ev.bar_idx });
    }
  }
  return out;
}

export function buildSignalMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const { signal, bar_idx } of collectSignals(timeline, currentBarIdx)) {
    const t = timeForBar(timeline, bar_idx);
    if (t === null) continue;
    out.push({
      time: t as Time,
      position: signal.side === 'short' ? 'aboveBar' : 'belowBar',
      shape: 'circle',
      color: signalColor(signal.side),
      text: signal.pattern ?? signal.source ?? '✦',
      size: 1,
      id: signal.id ?? `sig-${bar_idx}-${signal.pattern ?? 'x'}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export const signalsLayer: ChartLayer = {
  id: 'signals',
  name: 'Signals',
  swatch: LONG_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );
    let priceLines: IPriceLine[] = [];

    const clearLines = () => {
      for (const pl of priceLines) {
        try {
          ctx.primarySeries.removePriceLine(pl);
        } catch {
          // series may already be detached
        }
      }
      priceLines = [];
    };

    return {
      update(timeline, currentBarIdx) {
        if (!plugin) return;
        plugin.setMarkers(buildSignalMarkers(timeline, currentBarIdx));

        clearLines();
        const ev = timeline.events.find((e) => e.bar_idx === currentBarIdx);
        if (!ev?.signals?.length) return;
        for (const sig of ev.signals) {
          const entry = readSignalPrice(sig, 'entry_px') ?? readSignalPrice(sig, 'entry');
          const stop = readSignalPrice(sig, 'stop_px') ?? readSignalPrice(sig, 'stop');
          if (entry !== null) {
            priceLines.push(
              ctx.primarySeries.createPriceLine({
                price: entry,
                color: ENTRY_LINE_COLOR,
                lineWidth: 1,
                lineStyle: LineStyle.Dotted,
                axisLabelVisible: true,
                title: `entry ${sig.pattern ?? ''}`,
              }),
            );
          }
          if (stop !== null) {
            priceLines.push(
              ctx.primarySeries.createPriceLine({
                price: stop,
                color: STOP_LINE_COLOR,
                lineWidth: 1,
                lineStyle: LineStyle.Dotted,
                axisLabelVisible: true,
                title: `stop ${sig.pattern ?? ''}`,
              }),
            );
          }
        }
      },
      unmount() {
        clearLines();
        try {
          plugin?.detach();
        } catch {
          // chart already disposed
        }
        plugin = null;
      },
    };
  },
};
