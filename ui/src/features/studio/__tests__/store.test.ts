import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LIVE_TAIL, useStudioStore } from '../store';
import { makeBarEvent, makeTimeline } from './fixtures';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

describe('studio store', () => {
  it('starts with LIVE_TAIL and no timeline', () => {
    const s = useStudioStore.getState();
    expect(s.timeline).toBeNull();
    expect(s.currentBarIdx).toBe(LIVE_TAIL);
    expect(s.playState).toBe('paused');
    expect(s.speed).toBe(1);
  });

  it('replay sessions: initial cursor lands on the last bar (not LIVE_TAIL)', () => {
    const { actions } = useStudioStore.getState();
    const tl = { ...makeTimeline(5), session_kind: 'replay' as const };
    actions.setTimeline(tl);
    const s = useStudioStore.getState();
    expect(s.mode).toBe('replay');
    expect(s.currentBarIdx).toBe(4);
  });

  it('live sessions: cursor stays on LIVE_TAIL after timeline load', () => {
    const { actions } = useStudioStore.getState();
    const tl = { ...makeTimeline(5), session_kind: 'live' as const };
    actions.setTimeline(tl);
    const s = useStudioStore.getState();
    expect(s.mode).toBe('live');
    expect(s.currentBarIdx).toBe(LIVE_TAIL);
  });

  it('clamps setBar against timeline bounds', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    actions.setBar(2);
    expect(useStudioStore.getState().currentBarIdx).toBe(2);
    actions.setBar(99);
    expect(useStudioStore.getState().currentBarIdx).toBe(4);
    actions.setBar(-3);
    expect(useStudioStore.getState().currentBarIdx).toBe(0);
  });

  it('stepBar resolves LIVE_TAIL to last bar', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    expect(useStudioStore.getState().currentBarIdx).toBe(LIVE_TAIL);
    actions.stepBar(-1);
    expect(useStudioStore.getState().currentBarIdx).toBe(3);
    actions.stepBar(1);
    expect(useStudioStore.getState().currentBarIdx).toBe(4);
    actions.stepBar(1); // already at last bar
    expect(useStudioStore.getState().currentBarIdx).toBe(4);
  });

  it('togglePlay is a no-op without bars', () => {
    const { actions } = useStudioStore.getState();
    actions.togglePlay();
    expect(useStudioStore.getState().playState).toBe('paused');
    actions.setTimeline(makeTimeline(3));
    actions.togglePlay();
    expect(useStudioStore.getState().playState).toBe('playing');
    actions.togglePlay();
    expect(useStudioStore.getState().playState).toBe('paused');
  });

  it('cycleSpeed walks the ladder and clamps at the ends', () => {
    const { actions } = useStudioStore.getState();
    expect(useStudioStore.getState().speed).toBe(1);
    actions.cycleSpeed('up');
    expect(useStudioStore.getState().speed).toBe(2);
    actions.cycleSpeed('up');
    actions.cycleSpeed('up');
    actions.cycleSpeed('up');
    expect(useStudioStore.getState().speed).toBe(10);
    actions.cycleSpeed('up');
    expect(useStudioStore.getState().speed).toBe(10);
    actions.cycleSpeed('down');
    expect(useStudioStore.getState().speed).toBe(5);
  });

  it('jumpToLive resets currentBarIdx and pauses', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    actions.setBar(2);
    actions.setPlayState('playing');
    actions.jumpToLive();
    const s = useStudioStore.getState();
    expect(s.currentBarIdx).toBe(LIVE_TAIL);
    expect(s.playState).toBe('paused');
  });

  it('applyLiveEvent appends a new event when bar_idx is novel', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(3));
    actions.applyLiveEvent(makeBarEvent(3, { pnl_r: 0.5 }));
    const tl = useStudioStore.getState().timeline!;
    expect(tl.events.length).toBe(4);
    expect(tl.events[3].pnl_r).toBe(0.5);
  });

  it('applyLiveEvent merges into the latest event when bar_idx matches', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(3));
    actions.applyLiveEvent(makeBarEvent(2, { pnl_r: 1.2 }));
    const tl = useStudioStore.getState().timeline!;
    expect(tl.events.length).toBe(3);
    expect(tl.events[2].pnl_r).toBe(1.2);
    expect(tl.events[2].bar_idx).toBe(2);
  });
});
