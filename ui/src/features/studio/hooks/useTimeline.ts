/**
 * Loads a SessionTimeline + opens a live WS subscription.
 *
 * Live WS events go through a 50 ms coalesce buffer (`WS_BATCH_WINDOW_MS`)
 * before being applied via `applyLiveEventBatch`, so a burst of bar events
 * triggers at most ~20 store mutations / second. The chart automatically
 * follows the new tail when `currentBarIdx === LIVE_TAIL`.
 */

import { useEffect } from 'react';
import { fetchTimeline, subscribeStudio } from '../api';
import { useStudioActions } from '../store';
import type { BarEvent } from '../types';

export const WS_BATCH_WINDOW_MS = 50;

export function useTimeline(sessionId: string | null) {
  const actions = useStudioActions();

  useEffect(() => {
    if (!sessionId) {
      actions.reset();
      return;
    }

    const ac = new AbortController();
    let disposed = false;
    let wsHandle: ReturnType<typeof subscribeStudio> | null = null;

    let batch: BarEvent[] = [];
    let flushTimer: ReturnType<typeof setTimeout> | null = null;

    const flushBatch = () => {
      flushTimer = null;
      if (batch.length === 0) return;
      const drained = batch;
      batch = [];
      actions.applyLiveEventBatch(drained);
    };

    const enqueue = (ev: BarEvent) => {
      batch.push(ev);
      if (flushTimer === null) {
        flushTimer = setTimeout(flushBatch, WS_BATCH_WINDOW_MS);
      }
    };

    actions.setLoading(true);
    actions.setError(null);

    (async () => {
      try {
        const timeline = await fetchTimeline(sessionId, ac.signal);
        if (disposed) return;
        actions.setTimeline(timeline);
      } catch (e) {
        if (disposed || ac.signal.aborted) return;
        actions.setError(e instanceof Error ? e.message : 'Unknown error');
      } finally {
        if (!disposed) actions.setLoading(false);
      }

      if (disposed) return;
      wsHandle = subscribeStudio(sessionId, {
        onEvent: enqueue,
      });
    })();

    return () => {
      disposed = true;
      ac.abort();
      wsHandle?.close();
      if (flushTimer !== null) {
        clearTimeout(flushTimer);
        flushTimer = null;
      }
      // Drop any unflushed events — the next mount re-fetches the timeline
      // snapshot, which is authoritative.
      batch = [];
    };
    // `actions` is a stable Zustand selector return, but we depend on
    // sessionId to drive reload behaviour.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);
}
