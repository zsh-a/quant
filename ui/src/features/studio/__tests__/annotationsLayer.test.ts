import { describe, expect, it, vi } from 'vitest';

const markersByPlugin = new Map<unknown, unknown[]>();
const detached: unknown[] = [];

vi.mock('lightweight-charts', () => ({
  createSeriesMarkers: vi.fn((series: unknown, initial: unknown[] = []) => {
    const handle = {
      _series: series,
      setMarkers: vi.fn((m: unknown[]) => markersByPlugin.set(handle, m)),
      detach: vi.fn(() => detached.push(handle)),
    };
    markersByPlugin.set(handle, initial);
    return handle;
  }),
}));

import {
  annotationsLayer,
  brooksShorthand,
  buildAnnotationMarkers,
  buildAnnotations,
  categoryOf,
  strongPatternSignal,
} from '../layers/annotations';
import { makeLayerCtx } from './mockChart';
import { makeSignal, makeTimeline } from './fixtures';

describe('annotations layer (Brooks-style sparse markers)', () => {
  it('keeps the brooks shorthand helper for tooltips / sidebar', () => {
    expect(brooksShorthand('high_2')).toBe('H2');
    expect(brooksShorthand('Wedge')).toBe('Wdg');
    expect(brooksShorthand('flag_pole_rally')).toBe('FPR');
    expect(brooksShorthand(null)).toBe('?');
  });

  it('reads pattern_type from signals, falling back to detector name', () => {
    expect(categoryOf(makeSignal({ id: 'a', pattern_type: 'wedge' }))).toBe('wedge');
    expect(categoryOf(makeSignal({ id: 'b', pattern: 'wedge_long' }))).toBe('wedge');
    expect(categoryOf(makeSignal({ id: 'c', pattern: 'h2' }))).toBe('');
    expect(categoryOf(null)).toBe('');
  });

  it('only surfaces strong reversal patterns on the chart', () => {
    expect(strongPatternSignal([makeSignal({ id: 'a', pattern: 'h2' })])).toBeNull();
    expect(
      strongPatternSignal([
        makeSignal({ id: 'a', pattern: 'h2' }),
        makeSignal({ id: 'b', pattern_type: 'double_top', side: 'short' }),
      ]),
    ).toMatchObject({ id: 'b' });
  });

  it('emits one annotation per bar — fill > pattern label > failed signal', () => {
    const tl = makeTimeline(5);
    // bar 0: fill wins everything
    tl.events[0].fill = { side: 'buy', qty: 1, price: 101.5, reason: 'entry' };
    tl.events[0].signals = [makeSignal({ id: 'sw', pattern_type: 'wedge', side: 'long' })];
    tl.events[0].failed_signals = [makeSignal({ id: 'fs', pattern: 'h2' })];
    // bar 1: strong pattern (double_top), no fill
    tl.events[1].signals = [
      makeSignal({ id: 's1', pattern_type: 'double_top', side: 'short' }),
    ];
    // bar 2: failed signal only — red dot
    tl.events[2].failed_signals = [makeSignal({ id: 's2', pattern: 'h2', side: 'long' })];
    tl.events[2].background = { in_trading_range: false, allow: false, reason: 'tight TR' };
    // bar 3: only weak signals (h1) → nothing on chart
    tl.events[3].signals = [makeSignal({ id: 's3', pattern: 'h1', side: 'long' })];
    // bar 4: no events at all → nothing

    const out = buildAnnotations(tl, 4);
    expect(out.length).toBe(3);
    expect(out[0]).toMatchObject({ bar_idx: 0, kind: 'fill', text: '' });
    expect(out[1]).toMatchObject({ bar_idx: 1, kind: 'pattern', text: 'Double top' });
    expect(out[2]).toMatchObject({ bar_idx: 2, kind: 'failed_signal', text: '' });
    expect(out[2].title).toContain('tight TR');
  });

  it('hides annotations from future bars (no info leak)', () => {
    const tl = makeTimeline(5);
    tl.events[3].signals = [makeSignal({ id: 'a', pattern_type: 'wedge', side: 'long' })];
    expect(buildAnnotations(tl, 2).length).toBe(0);
    expect(buildAnnotations(tl, 3).length).toBe(1);
  });

  it('positions long fills below the bar and shorts/sells above', () => {
    const tl = makeTimeline(3);
    tl.events[1].fill = { side: 'buy', qty: 1, price: 100, reason: 'entry' };
    tl.events[2].fill = { side: 'sell', qty: 1, price: 110, reason: 'tp' };
    const out = buildAnnotations(tl, 2);
    expect(out[0].position).toBe('belowBar');
    expect(out[0].shape).toBe('arrowUp');
    expect(out[1].position).toBe('aboveBar');
    expect(out[1].shape).toBe('arrowDown');
  });

  it('drops H1/L2 shorthand spam — only fills, strong patterns, failed dots', () => {
    const tl = makeTimeline(2);
    tl.events[0].signals = [
      makeSignal({ id: 'a', pattern: 'h1', side: 'long' }),
      makeSignal({ id: 'b', pattern: 'h2', side: 'long' }),
      makeSignal({ id: 'c', pattern: 'l2', side: 'short' }),
    ];
    expect(buildAnnotations(tl, 1).length).toBe(0);
  });

  it('builds chart markers with stable IDs and sorted time', () => {
    const tl = makeTimeline(3);
    tl.events[2].signals = [
      makeSignal({ id: 'a', pattern_type: 'wedge', side: 'long' }),
    ];
    tl.events[1].signals = [
      makeSignal({ id: 'b', pattern_type: 'double_bottom', side: 'long' }),
    ];
    const markers = buildAnnotationMarkers(tl, 2);
    expect(markers.length).toBe(2);
    expect(markers[0].id).toBe('ann-1-pattern');
    expect(markers[1].id).toBe('ann-2-pattern');
    expect(markers[0].time).toBeLessThan(markers[1].time as number);
  });

  it('mounts a marker plugin, calls setMarkers on update, detaches on unmount', () => {
    const { ctx } = makeLayerCtx();
    const handle = annotationsLayer.mount(ctx);
    const tl = makeTimeline(2);
    tl.events[1].signals = [makeSignal({ id: 'a', pattern_type: 'wedge', side: 'long' })];
    handle.update(tl, 1);
    const lastMarkers = [...markersByPlugin.values()].pop() as unknown[];
    expect(lastMarkers.length).toBe(1);
    handle.unmount();
    expect(detached.length).toBeGreaterThan(0);
  });
});
