/**
 * Replay loop + keyboard shortcuts for the studio.
 *
 * Keys (disabled while a text input is focused):
 *   ←/→        step ±1
 *   Shift+←/→  step ±10
 *   Space      togglePlay
 *   [ / ]      speed × 0.5 / × 2 (cycles through STUDIO_SPEEDS ladder)
 *   End        jumpToLive
 *   Home       setBar(0)
 */

import { useEffect } from 'react';
import {
  useCurrentBarIdx,
  useStudioActions,
  useStudioPlayState,
  useStudioSpeed,
  useStudioStore,
  LIVE_TAIL,
} from '../store';

const PLAY_BASE_INTERVAL_MS = 600;

function isEditableTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return true;
  if (target.isContentEditable) return true;
  return false;
}

export function useReplay() {
  const actions = useStudioActions();
  const playState = useStudioPlayState();
  const speed = useStudioSpeed();
  const currentBarIdx = useCurrentBarIdx();

  // Playback ticker.
  useEffect(() => {
    if (playState !== 'playing') return;

    const interval = Math.max(50, Math.round(PLAY_BASE_INTERVAL_MS / speed));
    const id = window.setInterval(() => {
      const { timeline, currentBarIdx: idx, actions: a } = useStudioStore.getState();
      const barCount = timeline?.bars.length ?? 0;
      if (barCount === 0) {
        a.setPlayState('paused');
        return;
      }
      const base = idx === LIVE_TAIL ? barCount - 1 : idx;
      if (base >= barCount - 1) {
        a.setPlayState('paused');
        return;
      }
      a.setBar(base + 1);
    }, interval);

    return () => clearInterval(id);
  }, [playState, speed]);

  // Keyboard shortcuts.
  useEffect(() => {
    function onKeyDown(e: KeyboardEvent) {
      if (isEditableTarget(e.target)) return;

      switch (e.key) {
        case 'ArrowLeft':
          e.preventDefault();
          actions.stepBar(e.shiftKey ? -10 : -1);
          break;
        case 'ArrowRight':
          e.preventDefault();
          actions.stepBar(e.shiftKey ? 10 : 1);
          break;
        case ' ':
        case 'Spacebar':
          e.preventDefault();
          actions.togglePlay();
          break;
        case '[':
          e.preventDefault();
          actions.cycleSpeed('down');
          break;
        case ']':
          e.preventDefault();
          actions.cycleSpeed('up');
          break;
        case 'End':
          e.preventDefault();
          actions.jumpToLive();
          break;
        case 'Home':
          e.preventDefault();
          actions.setBar(0);
          break;
      }
    }

    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [actions]);

  return { playState, speed, currentBarIdx };
}
