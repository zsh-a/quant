import { useState, useEffect, useRef, useCallback } from 'react';
import type { EquityPoint, Trade, Position } from '../types';
import { apiFetch } from '../utils/api';
import {
  useSessions,
  useSelectedSessions,
  usePrimarySession,
  useSessionDataCache,
  useSessionActions,
} from '../store';

function mergeEquity(prev: EquityPoint[], next: EquityPoint[]): EquityPoint[] {
  if (next.length === 0) return prev;
  const seen = new Set(prev.map((p) => p.timestamp));
  const added = next.filter((p) => !seen.has(p.timestamp));
  return added.length === 0 ? prev : [...prev, ...added];
}

function mergeTrades(prev: Trade[], next: Trade[]): Trade[] {
  if (next.length === 0) return prev;
  const key = (t: Trade) => `${t.timestamp}-${t.symbol}-${t.type}-${t.quantity}`;
  const seen = new Set(prev.map(key));
  const added = next.filter((t) => !seen.has(key(t)));
  return added.length === 0 ? prev : [...prev, ...added];
}

export { mergeEquity, mergeTrades };

export function useSessionData(activeRoute: string) {
  const sessions = useSessions();
  const selectedSessionIds = useSelectedSessions();
  const primarySessionId = usePrimarySession();
  const sessionDataCache = useSessionDataCache();
  const { updateSession, addSessionData, removeSession } = useSessionActions();

  const [equityHistory, setEquityHistory] = useState<EquityPoint[]>([]);
  const [trades, setTrades] = useState<Trade[]>([]);
  const [positions, setPositions] = useState<Record<string, Position>>({});

  const lastUpdatedRef = useRef<string | null>(null);
  const inFlightRef = useRef(false);

  const primarySession = sessions.find((s) => s.id === primarySessionId);

  const fetchSessionDataFull = useCallback(
    async (id: string) => {
      const session = sessions.find((s) => s.id === id);
      if (session?.status === 'completed' && sessionDataCache[id]) return;

      try {
        const fetchPaged = async (path: string, limit = 2000) => {
          let offset = 0;
          let hasMore = true;
          const items: any[] = [];
          while (hasMore) {
            const resp = await apiFetch(`${path}?limit=${limit}&offset=${offset}`);
            if (resp.status === 404) {
              removeSession(id);
              return [];
            }
            if (!resp.ok) throw new Error(`Failed to fetch ${path}`);
            const data = await resp.json();
            const pageItems = Array.isArray(data.items) ? data.items : [];
            items.push(...pageItems);
            hasMore = Boolean(data.has_more);
            if (pageItems.length === 0) hasMore = false;
            else offset += pageItems.length;
          }
          return items;
        };

        const [equity, tradeList] = await Promise.all([
          fetchPaged(`/sessions/${id}/equity`),
          fetchPaged(`/sessions/${id}/trades`),
        ]);

        const pos = equity.length > 0 ? equity[equity.length - 1]?.positions || {} : {};
        addSessionData(id, { equity, trades: tradeList, positions: pos });
      } catch (err) {
        console.error('Error fetching full session data', err);
      }
    },
    [sessions, sessionDataCache, addSessionData, removeSession]
  );

  const fetchSessionDetails = useCallback(
    async (id: string) => {
      if (inFlightRef.current) return;
      try {
        inFlightRef.current = true;
        const since = lastUpdatedRef.current;
        const path = since
          ? `/session/${id}/status?since=${encodeURIComponent(since)}`
          : `/session/${id}/status`;
        const resp = await apiFetch(path);

        if (resp.status === 404) {
          removeSession(id);
          return;
        }

        const data = await resp.json();
        const equityList = data.equity_history || [];
        const tradeList = data.trades || [];

        updateSession(id, {
          status: data.status,
          progress: data.progress,
          end_date: data.end_date,
        });

        if (!since) {
          setEquityHistory(equityList);
          setTrades(tradeList);
        } else {
          setEquityHistory((prev) => mergeEquity(prev, equityList));
          setTrades((prev) => mergeTrades(prev, tradeList));
        }

        if (equityList.length > 0) {
          lastUpdatedRef.current = equityList[equityList.length - 1].timestamp;
        }
        setPositions(data.positions || {});
      } catch (err) {
        console.error('Fetch details error', err);
      } finally {
        inFlightRef.current = false;
      }
    },
    [updateSession, removeSession]
  );

  // 主 session 切换时清空并重新获取
  const isSessionRoute = activeRoute.startsWith('/session');
  useEffect(() => {
    if (isSessionRoute && primarySessionId) {
      setEquityHistory([]);
      setTrades([]);
      setPositions({});
      lastUpdatedRef.current = null;
      fetchSessionDetails(primarySessionId);
    }

    let detailInterval: number | null = null;
    if (isSessionRoute && primarySessionId) {
      const interval = primarySession?.status === 'running' ? 2500 : 8000;
      detailInterval = window.setInterval(() => {
        fetchSessionDetails(primarySessionId);
      }, interval);
    }

    return () => {
      if (detailInterval) clearInterval(detailInterval);
    };
  }, [isSessionRoute, primarySessionId, primarySession?.run_id, primarySession?.status]);

  // 加载选中的对比 session 数据
  useEffect(() => {
    selectedSessionIds.forEach((id) => {
      if (!sessionDataCache[id]) {
        fetchSessionDataFull(id);
      }
    });
  }, [selectedSessionIds]);

  return {
    equityHistory,
    setEquityHistory,
    trades,
    setTrades,
    positions,
    setPositions,
    lastUpdatedRef,
    fetchSessionDetails,
    fetchSessionDataFull,
    mergeEquity,
    mergeTrades,
  };
}
