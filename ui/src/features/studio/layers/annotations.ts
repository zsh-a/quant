/**
 * Annotations layer — Al Brooks-style sparse labels, one per bar.
 *
 * The legacy stack of layers (signals + decisions + fills + reasoning) each
 * pushed their own marker on the same bar, producing a wall of overlapping
 * text. This layer collapses every event into at most one short, bar-anchored
 * label using Brooks shorthand (H1/H2/L1/L2/MTR/BO/ii/FF/M2B/MM/…).
 *
 * Priority when a single bar carries multiple kinds of event:
 *   fill > decision > top-priority signal
 *
 * The full per-bar detail (all signals, full reasoning) is still available in
 * the side panel; here we deliberately keep the chart sparse so the user can
 * read structure at a glance, then drill in.
 *
 * Future-info safety: bars with `bar_idx > currentBarIdx` are dropped before
 * label construction, so scrubbing back hides them.
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
const SIGNAL_NEUTRAL = '#90A4AE';
const FILL_BUY_COLOR = '#26A69A';
const FILL_SELL_COLOR = '#EF5350';

/**
 * Brooks shorthand — keep these short (≤4 chars) so they don't crowd the
 * chart. Anything missing falls through to the raw `pattern` text.
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

/**
 * Built-in priority ordering when a bar has multiple raw signals — Brooks
 * traders care about MTR/BO over H1/L1, so we surface the strongest one and
 * drop the rest into the inspector.
 */
const SIGNAL_PRIORITY = new Map<string, number>([
  ['mtr', 100],
  ['major_trend_reversal', 100],
  ['bo', 90],
  ['breakout', 90],
  ['failed_breakout', 85],
  ['ff', 85],
  ['m2b', 80],
  ['m2t', 80],
  ['micro_double_bottom', 80],
  ['micro_double_top', 80],
  ['mm', 75],
  ['measured_move', 75],
  ['wedge', 70],
  ['ii', 60],
  ['ioi', 60],
  ['inside_inside', 60],
  ['h2', 50],
  ['l2', 50],
  ['high_2', 50],
  ['low_2', 50],
  ['h1', 30],
  ['l1', 30],
  ['high_1', 30],
  ['low_1', 30],
]);

export function brooksShorthand(pattern: string | undefined | null): string {
  if (!pattern) return '?';
  const key = pattern.toLowerCase().replace(/[\s-]+/g, '_');
  if (BROOKS_SHORTHAND[key]) return BROOKS_SHORTHAND[key];
  // Fallback: take the first letter of each `_`-separated chunk, uppercased.
  const compact = key
    .split('_')
    .map((chunk) => chunk.charAt(0).toUpperCase())
    .join('');
  return compact.slice(0, 4) || pattern.slice(0, 4);
}

function topSignal(signals: Signal[] | undefined): Signal | null {
  if (!signals || signals.length === 0) return null;
  let best: Signal | null = null;
  let bestScore = -1;
  for (const s of signals) {
    const key = (s.pattern ?? '').toLowerCase().replace(/[\s-]+/g, '_');
    const score = SIGNAL_PRIORITY.get(key) ?? 10;
    if (score > bestScore) {
      bestScore = score;
      best = s;
    }
  }
  return best;
}

function fillSideColor(side: FillView['side']): string {
  if (side === 'buy' || side === 'buy_to_cover') return FILL_BUY_COLOR;
  return FILL_SELL_COLOR;
}

function isBuySide(side: FillView['side']): boolean {
  return side === 'buy' || side === 'buy_to_cover';
}

export interface BarAnnotation {
  bar_idx: number;
  text: string;
  color: string;
  position: BarMarkerPosition;
  shape: SeriesMarkerShape;
  /** Coarse classification for the inspector. */
  kind: 'fill' | 'decision' | 'signal';
  /** Title — full pattern name for the marker tooltip / a11y. */
  title: string;
}

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

/**
 * Build at most one annotation per bar. Priority: fill > decision > top
 * signal. Fills get a ▲/▼ arrow; decisions a smaller arrow; signals a circle.
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

function annotationForEvent(ev: BarEvent): BarAnnotation | null {
  if (ev.fill) {
    const buy = isBuySide(ev.fill.side);
    const tag = ev.fill.reason ? ev.fill.reason.toUpperCase().slice(0, 4) : (buy ? 'BUY' : 'SLL');
    return {
      bar_idx: ev.bar_idx,
      text: `${tag} ${ev.fill.price.toFixed(2)}`,
      color: fillSideColor(ev.fill.side),
      position: buy ? 'belowBar' : 'aboveBar',
      shape: buy ? 'arrowUp' : 'arrowDown',
      kind: 'fill',
      title: `${ev.fill.side} @${ev.fill.price.toFixed(2)} (${ev.fill.reason || 'fill'})`,
    };
  }
  if (ev.decision) {
    const tag = brooksShorthand(ev.decision.pattern);
    const arrow = ev.decision.side === 'long' ? 'arrowUp' : 'arrowDown';
    return {
      bar_idx: ev.bar_idx,
      text: tag,
      color: ev.decision.side === 'long' ? LONG_COLOR : SHORT_COLOR,
      position: ev.decision.side === 'long' ? 'belowBar' : 'aboveBar',
      shape: arrow,
      kind: 'decision',
      title: `${ev.decision.side} ${ev.decision.pattern} → ${ev.decision.entry_px}`,
    };
  }
  const sig = topSignal(ev.signals);
  if (sig) {
    const tag = brooksShorthand(sig.pattern);
    const color =
      sig.side === 'long' ? LONG_COLOR : sig.side === 'short' ? SHORT_COLOR : SIGNAL_NEUTRAL;
    return {
      bar_idx: ev.bar_idx,
      text: tag,
      color,
      position: sig.side === 'short' ? 'aboveBar' : 'belowBar',
      shape: 'circle',
      kind: 'signal',
      title: `${sig.pattern ?? 'signal'}${sig.side ? ` (${sig.side})` : ''}`,
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
      // Decisions/fills are size 1; signals are slightly smaller so the eye is
      // drawn to actual entries first.
      size: ann.kind === 'signal' ? 0 : 1,
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
