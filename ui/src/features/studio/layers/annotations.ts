/**
 * Annotations layer — Al Brooks-style sparse markers, one per bar.
 *
 * Driven by the backend schema introduced in QUA-68:
 *   - `BarEvent.fill`             → entry arrow (green ↑ / red ↓)
 *   - `BarEvent.signals[].pattern_type` → strong-pattern text label
 *                                  ("Wedge", "Double top", …)
 *   - `BarEvent.failed_signals`   → red dot (signal that ContextFilter
 *                                  rejected; never traded)
 *
 * Priority when several apply on the same bar: **fill > pattern label >
 * failed-signal dot**. Anything not in that priority list (every H1/L2/MTR
 * shorthand the legacy layer used to spam) is intentionally dropped from
 * the chart — those still show up in the tooltip / SignalSidebar so the
 * structure is recoverable on demand.
 *
 * Future-info safety: bars with `bar_idx > currentBarIdx` are skipped, so
 * scrubbing back hides them.
 */

import {
  createSeriesMarkers,
  type ISeriesMarkersPluginApi,
  type SeriesMarker,
  type SeriesMarkerShape,
  type Time,
  type UTCTimestamp,
} from 'lightweight-charts';

/**
 * Subset of `SeriesMarkerPosition` that the lightweight-charts type system
 * recognises as bar-relative — the `atPrice*` variants additionally require
 * a `price` field that we never provide. Narrowing here keeps TS from
 * defaulting to the strictest union member.
 */
type BarMarkerPosition = 'aboveBar' | 'belowBar' | 'inBar';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { BarEvent, FillView, SessionTimeline, Signal } from '../types';

const LONG_COLOR = '#26A69A';
const SHORT_COLOR = '#EF5350';
const FAILED_DOT_COLOR = '#EF5350';
const FILL_BUY_COLOR = '#26A69A';
const FILL_SELL_COLOR = '#EF5350';

/**
 * Pattern categories worth a verbose chart label. Anything not listed here
 * (pullbacks, breakouts, micro-channels, generic "unknown") stays out of
 * the chart text — Brooks notes only foreground the strong reversals /
 * failed breakouts that mark a structural turn.
 */
const PATTERN_LABEL: Record<string, string> = {
  wedge: 'Wedge',
  double_top: 'Double top',
  double_bottom: 'Double bottom',
  mtr: 'MTR',
  final_flag: 'Final flag',
  failed_breakout: 'Failed BO',
};

/**
 * Detector-name → category fallback for older replays that pre-date the
 * server-side ``pattern_type`` field. Keeping this list narrow; legacy
 * shorthand (h2/l1/mtr_long…) maps onto the same six categories the Brooks
 * `PATTERN_TYPE_MAP` uses.
 */
const PATTERN_NAME_FALLBACK: Record<string, string> = {
  wedge: 'wedge',
  wedge_long: 'wedge',
  wedge_short: 'wedge',
  double_top: 'double_top',
  double_bottom: 'double_bottom',
  mtr: 'mtr',
  mtr_long: 'mtr',
  mtr_short: 'mtr',
  major_trend_reversal: 'mtr',
  final_flag: 'final_flag',
  failed_breakout: 'failed_breakout',
  ff: 'failed_breakout',
};

/**
 * Brooks shorthand — kept exported for tooltips / SidePanel use; the chart
 * itself no longer uses these as default text. Anything missing falls back
 * to the first-letter compaction of the raw pattern name.
 */
const BROOKS_SHORTHAND: Record<string, string> = {
  high_1: 'H1',
  high_2: 'H2',
  high_3: 'H3',
  low_1: 'L1',
  low_2: 'L2',
  low_3: 'L3',
  h1: 'H1',
  h2: 'H2',
  h3: 'H3',
  l1: 'L1',
  l2: 'L2',
  l3: 'L3',
  mtr: 'MTR',
  major_trend_reversal: 'MTR',
  bo: 'BO',
  breakout: 'BO',
  failed_breakout: 'FF',
  ff: 'FF',
  inside_inside: 'ii',
  ii: 'ii',
  ioi: 'ioi',
  m2b: 'M2B',
  m2s: 'M2S',
  micro_double_bottom: 'M2B',
  micro_double_top: 'M2T',
  m2t: 'M2T',
  mm: 'MM',
  measured_move: 'MM',
  wedge: 'Wdg',
  channel: 'Chn',
  spike: 'Spk',
  climax: 'Clx',
  pullback: 'PB',
};

export function brooksShorthand(pattern: string | undefined | null): string {
  if (!pattern) return '?';
  const key = pattern.toLowerCase().replace(/[\s-]+/g, '_');
  if (BROOKS_SHORTHAND[key]) return BROOKS_SHORTHAND[key];
  const compact = key
    .split('_')
    .map((chunk) => chunk.charAt(0).toUpperCase())
    .join('');
  return compact.slice(0, 4) || pattern.slice(0, 4);
}

/**
 * Resolve a signal's category. Prefers the server-provided
 * `pattern_type` (populated since QUA-68) and falls back to the detector
 * name for older replays.
 */
export function categoryOf(sig: Signal | undefined | null): string {
  if (!sig) return '';
  const explicit = (sig.pattern_type ?? '').toString().toLowerCase();
  if (explicit) return explicit;
  const name = (sig.pattern ?? '').toLowerCase().replace(/[\s-]+/g, '_');
  return PATTERN_NAME_FALLBACK[name] ?? '';
}

/**
 * Pick the strong-reversal signal worth foregrounding. The detector list
 * is small, so a linear scan is fine even on the busiest bars.
 */
export function strongPatternSignal(signals: Signal[] | undefined): Signal | null {
  if (!signals || signals.length === 0) return null;
  for (const s of signals) {
    if (PATTERN_LABEL[categoryOf(s)]) return s;
  }
  return null;
}

function fillSideColor(side: FillView['side']): string {
  if (side === 'buy' || side === 'buy_to_cover') return FILL_BUY_COLOR;
  return FILL_SELL_COLOR;
}

function isBuySide(side: FillView['side']): boolean {
  return side === 'buy' || side === 'buy_to_cover';
}

export type AnnotationKind = 'fill' | 'pattern' | 'failed_signal';

export interface BarAnnotation {
  bar_idx: number;
  text: string;
  color: string;
  position: BarMarkerPosition;
  shape: SeriesMarkerShape;
  size: number;
  kind: AnnotationKind;
  /** Title — full pattern name for the marker tooltip / a11y. */
  title: string;
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

/**
 * Build at most one annotation per bar. Priority:
 *   1. fill           — entry arrow, no text
 *   2. pattern label  — strong-pattern text (Wedge / Double top / …)
 *   3. failed signal  — red dot
 */
export function buildAnnotations(
  timeline: SessionTimeline,
  currentBarIdx: number,
): BarAnnotation[] {
  const out: BarAnnotation[] = [];
  for (const ev of timeline.events) {
    if (ev.bar_idx > currentBarIdx) continue;
    const ann = annotationForEvent(ev);
    if (ann) out.push(ann);
  }
  return out;
}

export function annotationForEvent(ev: BarEvent): BarAnnotation | null {
  if (ev.fill) {
    const buy = isBuySide(ev.fill.side);
    return {
      bar_idx: ev.bar_idx,
      text: '',
      color: fillSideColor(ev.fill.side),
      position: buy ? 'belowBar' : 'aboveBar',
      shape: buy ? 'arrowUp' : 'arrowDown',
      size: 2,
      kind: 'fill',
      title: `${ev.fill.side} @${ev.fill.price.toFixed(2)} (${ev.fill.reason || 'fill'})`,
    };
  }

  const strong = strongPatternSignal(ev.signals);
  if (strong) {
    const cat = categoryOf(strong);
    const label = PATTERN_LABEL[cat] ?? brooksShorthand(strong.pattern);
    const long = strong.side === 'long';
    return {
      bar_idx: ev.bar_idx,
      text: label,
      color: long ? LONG_COLOR : SHORT_COLOR,
      position: long ? 'belowBar' : 'aboveBar',
      shape: long ? 'arrowUp' : 'arrowDown',
      size: 1,
      kind: 'pattern',
      title: `${label}${strong.side ? ` (${strong.side})` : ''}`,
    };
  }

  const failed = ev.failed_signals && ev.failed_signals[0];
  if (failed) {
    const reason = ev.background?.reason ?? '';
    const labelTitle = failed.pattern ?? 'signal';
    return {
      bar_idx: ev.bar_idx,
      text: '',
      color: FAILED_DOT_COLOR,
      position: failed.side === 'short' ? 'aboveBar' : 'belowBar',
      shape: 'circle',
      size: 0,
      kind: 'failed_signal',
      title: reason ? `Failed: ${labelTitle} — ${reason}` : `Failed: ${labelTitle}`,
    };
  }

  return null;
}

export function buildAnnotationMarkers(
  timeline: SessionTimeline,
  currentBarIdx: number,
): SeriesMarker<Time>[] {
  const out: SeriesMarker<Time>[] = [];
  for (const ann of buildAnnotations(timeline, currentBarIdx)) {
    const t = timeForBar(timeline, ann.bar_idx);
    if (t === null) continue;
    out.push({
      time: t as Time,
      position: ann.position,
      shape: ann.shape,
      color: ann.color,
      text: ann.text,
      size: ann.size,
      id: `ann-${ann.bar_idx}-${ann.kind}`,
    });
  }
  out.sort((a, b) => (a.time as number) - (b.time as number));
  return out;
}

export const annotationsLayer: ChartLayer = {
  id: 'annotations',
  name: 'Annotations (Brooks)',
  swatch: LONG_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    let plugin: ISeriesMarkersPluginApi<Time> | null = createSeriesMarkers(
      ctx.primarySeries,
      [],
    );

    return {
      update(timeline, currentBarIdx) {
        if (!plugin) return;
        plugin.setMarkers(buildAnnotationMarkers(timeline, currentBarIdx));
      },
      unmount() {
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
