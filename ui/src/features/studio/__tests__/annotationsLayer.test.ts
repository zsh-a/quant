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
} from '../layers/annotations';
import { makeLayerCtx } from './mockChart';
import { makeDecision, makeSignal, makeTimeline } from './fixtures';

describe('annotations layer', () => {
  it('maps known patterns to Brooks shorthand', () => {
    expect(brooksShorthand('high_2')).toBe('H2');
    expect(brooksShorthand('low_1')).toBe('L1');
    expect(brooksShorthand('major_trend_reversal')).toBe('MTR');
    expect(brooksShorthand('failed_breakout')).toBe('FF');
    expect(brooksShorthand('Wedge')).toBe('Wdg');
  });

  it('falls back to first-letter compaction for unknown patterns', () => {
    expect(brooksShorthand('flag_pole_rally')).toBe('FPR');
    expect(brooksShorthand(null)).toBe('?');
  });

  it('emits at most one annotation per bar — fill > decision > signal', () => {
    const tl = makeTimeline(4);
    // bar 0: signal only → signal
    tl.events[0].signals = [makeSignal({ id: 's0', pattern: 'h2', side: 'long' })];
    // bar 1: decision + signal → decision wins
    tl.events[1].decision = makeDecision({ pattern: 'BO', side: 'long' });
    tl.events[1].signals = [makeSignal({ id: 's1', pattern: 'h1', side: 'long' })];
    // bar 2: fill + decision → fill wins
    tl.events[2].decision = makeDecision({ pattern: 'L2', side: 'short' });
    tl.events[2].fill = { side: 'buy', qty: 1, price: 101.5, reason: 'entry' };

    const out = buildAnnotations(tl, 3);
    expect(out.length).toBe(3);
    expect(out[0]).toMatchObject({ bar_idx: 0, kind: 'signal', text: 'H2' });
    expect(out[1]).toMatchObject({ bar_idx: 1, kind: 'decision', text: 'BO' });
    expect(out[2]).toMatchObject({ bar_idx: 2, kind: 'fill' });
    expect(out[2].text).toContain('101.50');
  });

  it('hides annotations from future bars (no info leak)', () => {
    const tl = makeTimeline(5);
    tl.events[3].decision = makeDecision({ pattern: 'BO' });
    expect(buildAnnotations(tl, 2).length).toBe(0);
    expect(buildAnnotations(tl, 3).length).toBe(1);
  });

  it('picks the strongest signal when a bar has many', () => {
    const tl = makeTimeline(2);
    tl.events[1].signals = [
      makeSignal({ id: 'a', pattern: 'h1', side: 'long' }),
      makeSignal({ id: 'b', pattern: 'mtr', side: 'long' }),
      makeSignal({ id: 'c', pattern: 'h2', side: 'long' }),
    ];
    const out = buildAnnotations(tl, 1);
    expect(out.length).toBe(1);
    expect(out[0].text).toBe('MTR');
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

  it('builds chart markers with stable IDs and sorted time', () => {
    const tl = makeTimeline(3);
    tl.events[2].decision = makeDecision({ pattern: 'BO' });
    tl.events[1].decision = makeDecision({ pattern: 'L2', side: 'short' });
    const markers = buildAnnotationMarkers(tl, 2);
    expect(markers.length).toBe(2);
    expect(markers[0].id).toBe('ann-1-decision');
    expect(markers[1].id).toBe('ann-2-decision');
    expect(markers[0].time).toBeLessThan(markers[1].time as number);
  });

  it('mounts a marker plugin, calls setMarkers on update, detaches on unmount', () => {
    const { ctx } = makeLayerCtx();
    const handle = annotationsLayer.mount(ctx);
    const tl = makeTimeline(2);
    tl.events[1].decision = makeDecision({ pattern: 'BO' });
    handle.update(tl, 1);
    const lastMarkers = [...markersByPlugin.values()].pop() as unknown[];
    expect(lastMarkers.length).toBe(1);
    handle.unmount();
    expect(detached.length).toBeGreaterThan(0);
  });
});
