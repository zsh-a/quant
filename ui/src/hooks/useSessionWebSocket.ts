import { useCallback } from 'react';
import { toast } from 'sonner';
import type { EquityPoint, Trade } from '../types';
import { useWebSocket } from './useWebSocket';
import { useSessionActions, usePrimarySession } from '../store';
import { mergeEquity, mergeTrades } from './useSessionData';

interface UseSessionWebSocketOptions {
  setEquityHistory: React.Dispatch<React.SetStateAction<EquityPoint[]>>;
  setTrades: React.Dispatch<React.SetStateAction<Trade[]>>;
  lastUpdatedRef: React.MutableRefObject<string | null>;
  fetchSessionDetails: (id: string) => Promise<void>;
}

export function useSessionWebSocket({
  setEquityHistory,
  setTrades,
  lastUpdatedRef,
  fetchSessionDetails,
}: UseSessionWebSocketOptions) {
  const primarySessionId = usePrimarySession();
  const { updateSession } = useSessionActions();

  const handleMessage = useCallback(
    (message: any) => {
      switch (message.type) {
        case 'session_progress':
          updateSession(message.session_id, {
            progress: message.data.progress,
            status: message.data.status,
          });
          break;
        case 'equity_update': {
          const equity = message.data.equity;
          setEquityHistory((prev) => mergeEquity(prev, [equity]));
          lastUpdatedRef.current = equity.timestamp;
          break;
        }
        case 'equity_batch': {
          const updates = Array.isArray(message.data?.updates) ? message.data.updates : [];
          if (updates.length > 0) {
            setEquityHistory((prev) => mergeEquity(prev, updates));
            lastUpdatedRef.current = updates[updates.length - 1].timestamp;
          }
          break;
        }
        case 'trade_executed':
          setTrades((prev) => mergeTrades(prev, [message.data.trade]));
          break;
        case 'trades_batch': {
          const batchTrades = Array.isArray(message.data?.trades) ? message.data.trades : [];
          if (batchTrades.length > 0) {
            setTrades((prev) => mergeTrades(prev, batchTrades));
          }
          break;
        }
        case 'session_completed':
          updateSession(message.session_id, { status: 'completed', progress: 100 });
          if (message.session_id === primarySessionId) {
            fetchSessionDetails(message.session_id);
          }
          break;
        case 'session_failed':
          updateSession(message.session_id, { status: 'failed' });
          toast.error(message.data.error || 'Session failed');
          break;
        case 'session_stopped':
          updateSession(message.session_id, { status: 'stopped' });
          break;
        case 'error_occurred':
          toast.error(message.data.error);
          break;
      }
    },
    [primarySessionId, updateSession, setEquityHistory, setTrades, lastUpdatedRef, fetchSessionDetails]
  );

  const { isConnected, usePolling } = useWebSocket({
    sessionId: primarySessionId || '',
    enabled: !!primarySessionId,
    onMessage: handleMessage,
    fallbackToPolling: true,
    pollingInterval: 2000,
  });

  return { isConnected, usePolling };
}
