/**
 * Loads a SessionTimeline + opens a live WS subscription.
 *
 * Initial fetch is paginated — the first call asks for `INITIAL_EVENT_LIMIT`
 * events for fast first paint, then any remaining events stream in via
 * `/timeline/since/{seq}` in the background. Bars and metadata always
 * come back complete on the first call so the chart can render any range
 * immediately.
 *
 * Live WS events go through a 50 ms coalesce buffer (`WS_BATCH_WINDOW_MS`)
 * before being applied via `applyLiveEventBatch`, so a burst of bar events
 * triggers at most ~20 store mutations / second. The chart automatically
 * follows the new tail when `currentBarIdx === LIVE_TAIL`.
 */

import { useEffect } from 'react';
import { fetchTimeline, fetchTimelinePage, subscribeStudio } from '../api';
import { useStudioActions } from '../store';
import type { BarEvent } from '../types';

export const WS_BATCH_WINDOW_MS = 50;
export const INITIAL_EVENT_LIMIT = 1_000;
export const PAGE_SIZE = 2_000;

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
      let kind: 'live' | 'replay' = 'live';
      let nextSeq = 0;
      let hasMore = false;
      try {
        const timeline = await fetchTimeline(sessionId, ac.signal, INITIAL_EVENT_LIMIT);
        if (disposed) return;
        kind = timeline.session_kind === 'replay' ? 'replay' : 'live';
        nextSeq = timeline.next_event_seq ?? 0;
        hasMore = Boolean(timeline.has_more_events);
        actions.setTimeline(timeline);
      } catch (e) {
        if (disposed || ac.signal.aborted) return;
        actions.setError(e instanceof Error ? e.message : 'Unknown error');
        return;
      } finally {
        if (!disposed) actions.setLoading(false);
      }

      if (disposed) return;

      // Page in remaining events in chronological order. Each batch is
      // applied via the same WS coalesce path so the renderer wakes up
      // at most ~20 times per second across both pagination and live tail.
      while (hasMore && nextSeq > 0 && !disposed && !ac.signal.aborted) {
        try {
          const page = await fetchTimelinePage(sessionId, nextSeq, PAGE_SIZE, ac.signal);
          if (disposed) return;
          if (page.events.length === 0) break;
          for (const ev of page.events) enqueue(ev);
          nextSeq = page.next_seq;
          hasMore = page.has_more;
        } catch {
          if (disposed || ac.signal.aborted) return;
          // Soft failure — stop paging but leave whatever we already have
          // visible. Next mount will retry from scratch.
          break;
        }
      }

      if (disposed) return;
      // Replay sessions are pre-computed end to end — no live tail, no WS.
      // The chart renders straight from the timeline snapshot once paging
      // completes.
      if (kind === 'replay') return;
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
