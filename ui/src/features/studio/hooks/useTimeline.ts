/**
 * Loads a SessionTimeline + opens a live WS subscription.
 *
 * Live WS events go through `applyLiveEvent`; the chart automatically
 * follows the new tail when `currentBarIdx === LIVE_TAIL`.
 */

import { useEffect } from 'react';
import { fetchTimeline, subscribeStudio } from '../api';
import { useStudioActions } from '../store';

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
        onEvent: (ev) => actions.applyLiveEvent(ev),
      });
    })();

    return () => {
      disposed = true;
      ac.abort();
      wsHandle?.close();
    };
    // `actions` is a stable Zustand selector return, but we depend on
    // sessionId to drive reload behaviour.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);
}
