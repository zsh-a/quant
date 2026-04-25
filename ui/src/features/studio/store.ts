/**
 * Studio Zustand store — single source of truth for chart playback state.
 *
 * `currentBarIdx === -1` is the sentinel for "follow live tail": the chart
 * keeps showing the last bar and any live `applyLiveEvent` call appends a
 * new bar/event without changing user-visible state.
 */

import { create } from 'zustand';
import type { BarEvent, PlayState, SessionTimeline, StudioMode, StudioSpeed } from './types';

export const LIVE_TAIL = -1;

export interface StudioState {
  timeline: SessionTimeline | null;
  loading: boolean;
  error: string | null;

  currentBarIdx: number;
  mode: StudioMode;
  playState: PlayState;
  speed: StudioSpeed;
  hoveredBarIdx: number | null;
}

export interface StudioActions {
  setTimeline: (tl: SessionTimeline | null) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  setBar: (idx: number) => void;
  stepBar: (delta: number) => void;
  togglePlay: () => void;
  setPlayState: (state: PlayState) => void;
  setSpeed: (s: StudioSpeed) => void;
  cycleSpeed: (direction: 'up' | 'down') => void;
  setMode: (mode: StudioMode) => void;
  jumpToLive: () => void;
  setHoveredBar: (idx: number | null) => void;

  applyLiveEvent: (ev: BarEvent) => void;
  reset: () => void;
}

export type StudioStore = StudioState & { actions: StudioActions };

const INITIAL: StudioState = {
  timeline: null,
  loading: false,
  error: null,
  currentBarIdx: LIVE_TAIL,
  mode: 'live',
  playState: 'paused',
  speed: 1,
  hoveredBarIdx: null,
};

const SPEED_LADDER: StudioSpeed[] = [1, 2, 5, 10];

function clampBarIdx(idx: number, barCount: number): number {
  // No timeline yet: keep the raw index so URL-driven `?bar=N` can be
  // persisted until bars arrive (we re-clamp on timeline load).
  if (barCount <= 0) return idx;
  if (idx === LIVE_TAIL) return LIVE_TAIL;
  if (idx < 0) return 0;
  if (idx >= barCount) return barCount - 1;
  return idx;
}

export const useStudioStore = create<StudioStore>()((set, get) => ({
  ...INITIAL,

  actions: {
    setTimeline: (timeline) => {
      const { currentBarIdx } = get();
      const barCount = timeline?.bars.length ?? 0;
      // Re-clamp any URL-driven currentBarIdx now that we know the bounds.
      set({
        timeline,
        error: null,
        currentBarIdx: clampBarIdx(currentBarIdx, barCount),
      });
    },

    setLoading: (loading) => set({ loading }),
    setError: (error) => set({ error }),

    setBar: (idx) => {
      const { timeline } = get();
      const barCount = timeline?.bars.length ?? 0;
      set({ currentBarIdx: clampBarIdx(idx, barCount) });
    },

    stepBar: (delta) => {
      const { timeline, currentBarIdx } = get();
      const barCount = timeline?.bars.length ?? 0;
      if (barCount === 0) return;
      const base = currentBarIdx === LIVE_TAIL ? barCount - 1 : currentBarIdx;
      // Clamp inline — `clampBarIdx` reserves -1 for the LIVE_TAIL sentinel
      // and stepping past zero must land on the first real bar, not live.
      const next = Math.max(0, Math.min(barCount - 1, base + delta));
      set({ currentBarIdx: next });
    },

    togglePlay: () => {
      const { playState, timeline } = get();
      if (!timeline || timeline.bars.length === 0) return;
      set({ playState: playState === 'playing' ? 'paused' : 'playing' });
    },

    setPlayState: (state) => set({ playState: state }),

    setSpeed: (s) => set({ speed: s }),

    cycleSpeed: (direction) => {
      const { speed } = get();
      const idx = SPEED_LADDER.indexOf(speed);
      if (idx === -1) {
        set({ speed: 1 });
        return;
      }
      const nextIdx = direction === 'up' ? Math.min(idx + 1, SPEED_LADDER.length - 1) : Math.max(idx - 1, 0);
      set({ speed: SPEED_LADDER[nextIdx] });
    },

    setMode: (mode) => set({ mode }),

    jumpToLive: () => {
      set({ currentBarIdx: LIVE_TAIL, playState: 'paused' });
    },

    setHoveredBar: (idx) => set({ hoveredBarIdx: idx }),

    applyLiveEvent: (ev) => {
      const { timeline } = get();
      if (!timeline) return;

      const events = [...timeline.events];
      const lastEvent = events[events.length - 1];

      if (lastEvent && lastEvent.bar_idx === ev.bar_idx) {
        events[events.length - 1] = { ...lastEvent, ...ev };
      } else {
        events.push(ev);
      }

      set({
        timeline: { ...timeline, events },
      });
    },

    reset: () => set({ ...INITIAL }),
  },
}));

// Selectors
export const useStudioActions = () => useStudioStore((s) => s.actions);
export const useTimelineState = () => useStudioStore((s) => s.timeline);
export const useStudioLoading = () => useStudioStore((s) => s.loading);
export const useStudioError = () => useStudioStore((s) => s.error);
export const useCurrentBarIdx = () => useStudioStore((s) => s.currentBarIdx);
export const useStudioMode = () => useStudioStore((s) => s.mode);
export const useStudioPlayState = () => useStudioStore((s) => s.playState);
export const useStudioSpeed = () => useStudioStore((s) => s.speed);
export const useHoveredBarIdx = () => useStudioStore((s) => s.hoveredBarIdx);

/** Effective bar index resolving the `LIVE_TAIL` sentinel against the current timeline. */
export const useEffectiveBarIdx = (): number => {
  return useStudioStore((s) => {
    const barCount = s.timeline?.bars.length ?? 0;
    if (barCount === 0) return -1;
    return s.currentBarIdx === LIVE_TAIL ? barCount - 1 : s.currentBarIdx;
  });
};
