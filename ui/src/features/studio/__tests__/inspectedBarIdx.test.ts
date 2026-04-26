import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { useInspectedBarIdx, useStudioStore } from '../store';
import { makeTimeline } from './fixtures';
import { renderHook } from '@testing-library/react';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

describe('useInspectedBarIdx (hover ghost selector)', () => {
  it('returns the current bar (resolved past LIVE_TAIL) when nothing is hovered', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    // LIVE_TAIL → resolves to last bar (4)
    const { result } = renderHook(() => useInspectedBarIdx());
    expect(result.current).toEqual({ barIdx: 4, preview: false });
  });

  it('returns hovered idx with preview=true when hover differs from current', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    actions.setBar(2);
    actions.setHoveredBar(0);

    const { result } = renderHook(() => useInspectedBarIdx());
    expect(result.current).toEqual({ barIdx: 0, preview: true });
  });

  it('preview is false when hover matches current bar', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    actions.setBar(2);
    actions.setHoveredBar(2);

    const { result } = renderHook(() => useInspectedBarIdx());
    expect(result.current).toEqual({ barIdx: 2, preview: false });
  });

  it('clears preview when hover returns to null', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(makeTimeline(5));
    actions.setBar(2);
    actions.setHoveredBar(0);
    actions.setHoveredBar(null);

    const { result } = renderHook(() => useInspectedBarIdx());
    expect(result.current).toEqual({ barIdx: 2, preview: false });
  });
});
