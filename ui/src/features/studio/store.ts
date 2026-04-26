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

export const LAYERS_STORAGE_KEY = 'brooks-studio-layers';

function readVisibleLayersFromStorage(): Set<string> | null {
  if (typeof window === 'undefined') return null;
  try {
    const raw = window.localStorage.getItem(LAYERS_STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return null;
    return new Set(parsed.filter((x): x is string => typeof x === 'string'));
  } catch {
    return null;
  }
}

function writeVisibleLayersToStorage(set: Set<string>): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(LAYERS_STORAGE_KEY, JSON.stringify([...set]));
  } catch {
    // storage may be full / unavailable — non-fatal
  }
}

export interface StudioState {
  timeline: SessionTimeline | null;
  loading: boolean;
  error: string | null;

  currentBarIdx: number;
  mode: StudioMode;
  playState: PlayState;
  speed: StudioSpeed;
  hoveredBarIdx: number | null;
  /** Selected signal id (e.g. clicked in the SignalSidebar) — used by layers
   * to highlight the matching marker / price lines. `null` = nothing pinned. */
  selectedSignalId: string | null;

  visibleLayers: Set<string>;
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
  setSelectedSignalId: (id: string | null) => void;

  applyLiveEvent: (ev: BarEvent) => void;
  /** Apply a coalesced batch in a single set() call. Used by useTimeline to
   *  flush WS events accumulated over a 50 ms window, so the renderer wakes
   *  up at most ~20 times per second on bursty streams. */
  applyLiveEventBatch: (events: BarEvent[]) => void;

  setVisibleLayers: (ids: Iterable<string>) => void;
  toggleLayer: (id: string) => void;

  reset: () => void;
}

export type StudioStore = StudioState & { actions: StudioActions };

function initialVisibleLayers(): Set<string> {
  return readVisibleLayersFromStorage() ?? new Set<string>();
}

const INITIAL: StudioState = {
  timeline: null,
  loading: false,
  error: null,
  currentBarIdx: LIVE_TAIL,
  mode: 'live',
  playState: 'paused',
  speed: 1,
  hoveredBarIdx: null,
  selectedSignalId: null,
  visibleLayers: initialVisibleLayers(),
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
      const kind: StudioMode = timeline?.session_kind === 'replay' ? 'replay' : 'live';

      // Re-clamp any URL-driven currentBarIdx now that we know the bounds.
      let nextBar = clampBarIdx(currentBarIdx, barCount);
      // Replay sessions have no live tail — pin the cursor inside the loaded
      // bar range so the chart shows real bars instead of an empty tail.
      if (kind === 'replay' && barCount > 0 && nextBar === LIVE_TAIL) {
        nextBar = barCount - 1;
      }

      set({
        timeline,
        error: null,
        currentBarIdx: nextBar,
        mode: kind,
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

    setSelectedSignalId: (id) => set({ selectedSignalId: id }),

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

    applyLiveEventBatch: (incoming) => {
      if (incoming.length === 0) return;
      const { timeline } = get();
      if (!timeline) return;

      const events = timeline.events.slice();
      for (const ev of incoming) {
        const lastEvent = events[events.length - 1];
        if (lastEvent && lastEvent.bar_idx === ev.bar_idx) {
          events[events.length - 1] = { ...lastEvent, ...ev };
        } else {
          events.push(ev);
        }
      }

      set({
        timeline: { ...timeline, events },
      });
    },

    setVisibleLayers: (ids) => {
      const next = new Set(ids);
      writeVisibleLayersToStorage(next);
      set({ visibleLayers: next });
    },

    toggleLayer: (id) => {
      const { visibleLayers } = get();
      const next = new Set(visibleLayers);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      writeVisibleLayersToStorage(next);
      set({ visibleLayers: next });
    },

    reset: () => set({ ...INITIAL, visibleLayers: initialVisibleLayers() }),
  },
}));

/**
 * Returns true if the studio's visibleLayers came from a fresh init (no
 * localStorage entry yet). UI code uses this to decide whether to seed the
 * store with registry-provided defaults.
 */
export function hasStoredVisibleLayers(): boolean {
  return readVisibleLayersFromStorage() !== null;
}

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
export const useSelectedSignalId = () => useStudioStore((s) => s.selectedSignalId);
export const useVisibleLayers = () => useStudioStore((s) => s.visibleLayers);

/**
 * Resolves the bar idx that side panels should display: the hovered bar if the
 * cursor is on the chart, otherwise the current bar. Returns the inspected idx
 * along with a flag so the UI can render a subtle "preview" badge.
 *
 * Implemented as two primitive selectors so the hook remains stable across
 * renders — returning a fresh object from a single selector would re-render
 * subscribers unconditionally.
 */
export const useInspectedBarIdx = (): { barIdx: number; preview: boolean } => {
  const barIdx = useStudioStore((s) => {
    const barCount = s.timeline?.bars.length ?? 0;
    const current = s.currentBarIdx === LIVE_TAIL && barCount > 0 ? barCount - 1 : s.currentBarIdx;
    return s.hoveredBarIdx !== null ? s.hoveredBarIdx : current;
  });
  const preview = useStudioStore((s) => {
    if (s.hoveredBarIdx === null) return false;
    const barCount = s.timeline?.bars.length ?? 0;
    const current = s.currentBarIdx === LIVE_TAIL && barCount > 0 ? barCount - 1 : s.currentBarIdx;
    return s.hoveredBarIdx !== current;
  });
  return { barIdx, preview };
};

/** Effective bar index resolving the `LIVE_TAIL` sentinel against the current timeline. */
export const useEffectiveBarIdx = (): number => {
  return useStudioStore((s) => {
    const barCount = s.timeline?.bars.length ?? 0;
    if (barCount === 0) return -1;
    return s.currentBarIdx === LIVE_TAIL ? barCount - 1 : s.currentBarIdx;
  });
};
