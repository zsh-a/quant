import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { LIVE_TAIL, useStudioStore } from '../store';
import { makeBarEvent, makeTimeline } from './fixtures';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

describe('side panel store extensions', () => {
  it('starts with no selected signal', () => {
    expect(useStudioStore.getState().selectedSignalId).toBeNull();
  });

  it('setSelectedSignalId stores and clears the id', () => {
    const { actions } = useStudioStore.getState();
    actions.setSelectedSignalId('sig-7');
    expect(useStudioStore.getState().selectedSignalId).toBe('sig-7');
    actions.setSelectedSignalId(null);
    expect(useStudioStore.getState().selectedSignalId).toBeNull();
  });

  it('reset clears the selected signal id', () => {
    const { actions } = useStudioStore.getState();
    actions.setSelectedSignalId('sig-x');
    actions.reset();
    expect(useStudioStore.getState().selectedSignalId).toBeNull();
  });

  it('hovered/selected state is independent of currentBarIdx', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    actions.setBar(2);
    actions.setHoveredBar(4);
    actions.setSelectedSignalId('sig-3');
    const s = useStudioStore.getState();
    expect(s.currentBarIdx).toBe(2);
    expect(s.hoveredBarIdx).toBe(4);
    expect(s.selectedSignalId).toBe('sig-3');
    // Live tail still allowed alongside hover state.
    actions.jumpToLive();
    expect(useStudioStore.getState().currentBarIdx).toBe(LIVE_TAIL);
    expect(useStudioStore.getState().hoveredBarIdx).toBe(4);
  });

  it('applyLiveEvent does not clobber selectedSignalId', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(2));
    actions.setSelectedSignalId('sig-pinned');
    actions.applyLiveEvent(makeBarEvent(2, { pnl_r: 0.1 }));
    expect(useStudioStore.getState().selectedSignalId).toBe('sig-pinned');
  });
});
