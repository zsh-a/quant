import { useCallback } from 'react';
import { toast } from 'sonner';
import { apiFetch } from '../utils/api';
import {
  useSessions,
  usePrimarySession,
  useSessionActions,
} from '../store';

interface UseSessionCrudOptions {
  onSessionStarted?: (sessionId: string) => void;
  onSessionDeleted?: (id: string) => void;
  fetchSessionDataFull?: (id: string) => Promise<void>;
}

export function useSessionCrud({
  onSessionStarted,
  onSessionDeleted,
  fetchSessionDataFull,
}: UseSessionCrudOptions = {}) {
  const sessions = useSessions();
  const primarySessionId = usePrimarySession();
  const { selectSession, updateSession, removeSession, setSessions } = useSessionActions();

  const fetchSessions = useCallback(async () => {
    try {
      const resp = await apiFetch(`/sessions`);
      const data = await resp.json();
      setSessions(data);
    } catch (err) {
      console.error('Failed to fetch sessions', err);
    }
  }, [setSessions]);

  const pollTaskProgress = useCallback(
    (sessionId: string, taskId: string) => {
      const poll = async () => {
        try {
          const resp = await apiFetch(`/tasks/backtest/${taskId}`);
          const data = await resp.json();

          updateSession(sessionId, {
            progress: data.progress || 0,
            status:
              data.status === 'SUCCESS'
                ? 'completed'
                : data.status === 'FAILURE'
                  ? 'failed'
                  : 'running',
          });

          if (data.status !== 'SUCCESS' && data.status !== 'FAILURE') {
            setTimeout(poll, 2000);
          } else {
            fetchSessions();
            if (sessionId === primarySessionId) {
              fetchSessionDataFull?.(sessionId);
            }
          }
        } catch (err) {
          console.error('Task polling error:', err);
        }
      };
      poll();
    },
    [updateSession, fetchSessions, primarySessionId, fetchSessionDataFull]
  );

  const startSession = useCallback(
    async (payload: any) => {
      try {
        const useAsync = payload.async === true;
        delete payload.async;

        const endpoint = useAsync ? '/session/run_async' : '/session/run';
        const resp = await apiFetch(`${endpoint}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });
        const data = await resp.json();

        await fetchSessions();
        selectSession(data.session_id);
        toast.success('Session started');
        onSessionStarted?.(data.session_id);

        if (useAsync && data.task_id) {
          pollTaskProgress(data.session_id, data.task_id);
        }
      } catch {
        toast.error('Failed to start session');
      }
    },
    [fetchSessions, selectSession, pollTaskProgress, onSessionStarted]
  );

  const stopSession = useCallback(
    async (id: string) => {
      try {
        await apiFetch(`/session/${id}/stop`, { method: 'POST' });
        toast.success('Session stopped');
        fetchSessions();
      } catch {
        toast.error('Failed to stop session');
      }
    },
    [fetchSessions]
  );

  const deleteSession = useCallback(
    async (id: string) => {
      const session = sessions.find((s) => s.id === id);
      if (!session) return;

      const confirmed = window.confirm(
        `Delete session ${session.strategy} (${id.slice(0, 8)})? This will remove all history, trades, logs, and checkpoints.`
      );
      if (!confirmed) return;

      try {
        const resp = await apiFetch(`/session/${id}`, { method: 'DELETE' });
        if (!resp.ok) {
          const data = await resp.json().catch(() => null);
          throw new Error(data?.detail || 'Failed to delete session');
        }

        removeSession(id);
        toast.success('Session deleted');
        if (primarySessionId === id) {
          onSessionDeleted?.(id);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to delete session';
        toast.error(message);
      }
    },
    [sessions, primarySessionId, removeSession, onSessionDeleted]
  );

  return {
    startSession,
    stopSession,
    deleteSession,
    fetchSessions,
  };
}
