/**
 * useBrooksLive — React hook wrapping the BrooksLive WebSocket channel.
 *
 * Opens a WS to ``/ws/brooks/{sessionId}`` and fans incoming events out
 * to typed React state: bars, regime, signals, equity curve, trades.
 * Falls back to the main session polling pipeline isn't needed here —
 * the live panel is useless without a live feed, so we just report the
 * connection status and let the user retry.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { WS_BASE, apiFetch, getToken } from '../utils/api';

export interface BarOHLCV {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface RegimePayload {
  regime: string;
  confidence: number;
  reasons: string[];
  consecutive_trend_bars?: number;
  ema_atr_distance?: number;
  swing_range_atr?: number;
  bar_overlap_ratio?: number;
}

export interface DecisionPayload {
  side: 'long' | 'short';
  entry_px: number;
  stop_px: number;
  target_px: number | null;
  quantity: number;
  probability: number;
  expected_r: number;
  regime: string;
  htf_aligned: boolean;
  pattern: string;
  source: string;
  reasoning: string;
  order_id?: string;
  order_type?: string;
  [key: string]: unknown;
}

export interface PositionPayload {
  side: string;
  entry_px: number;
  stop_px: number;
  qty_open: number;
  one_r: number;
  ladder_stage: number;
}

export interface EquityPoint {
  timestamp: string;
  total_equity: number;
  cash: number;
  positions?: Record<string, unknown>;
}

export interface TradeRecord {
  timestamp: string;
  symbol: string;
  type: string;
  price: number;
  quantity: number;
  commission?: number;
  [key: string]: unknown;
}

export interface BrooksLiveState {
  connected: boolean;
  sessionId: string | null;
  lastRegime: RegimePayload | null;
  lastDecision: DecisionPayload | null;
  recentSignals: DecisionPayload[];
  bars: BarOHLCV[];
  equityCurve: EquityPoint[];
  trades: TradeRecord[];
  positions: Record<string, PositionPayload>;
  analyst: string;
  symbol: string;
  lastEventAt: string | null;
  error: string | null;
}

const INITIAL_STATE: BrooksLiveState = {
  connected: false,
  sessionId: null,
  lastRegime: null,
  lastDecision: null,
  recentSignals: [],
  bars: [],
  equityCurve: [],
  trades: [],
  positions: {},
  analyst: 'rule',
  symbol: '',
  lastEventAt: null,
  error: null,
};

const MAX_BARS = 400;
const MAX_SIGNALS = 50;
const MAX_TRADES = 200;
const MAX_EQUITY = 2000;

export function useBrooksLive(sessionId: string | null) {
  const [state, setState] = useState<BrooksLiveState>(INITIAL_STATE);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleEvent = useCallback((raw: { type: string; data: Record<string, unknown>; timestamp?: string }) => {
    const d = raw.data ?? {};
    setState((prev) => {
      const next = { ...prev, lastEventAt: raw.timestamp ?? new Date().toISOString() };
      if (raw.type === 'strategy_step') {
        const event = d.event as string | undefined;
        if (event === 'bar_closed') {
          const ohlcv = d.ohlcv as BarOHLCV | undefined;
          const timestamp = (d.timestamp as string) ?? next.lastEventAt;
          if (ohlcv) {
            next.bars = [...prev.bars, { ...ohlcv, timestamp }].slice(-MAX_BARS);
            next.symbol = (d.symbol as string) ?? prev.symbol;
          }
          const regime = d.regime as RegimePayload | null | undefined;
          if (regime) next.lastRegime = regime;
          const decision = d.decision as DecisionPayload | null | undefined;
          if (decision) {
            next.lastDecision = decision;
            next.recentSignals = [decision, ...prev.recentSignals].slice(0, MAX_SIGNALS);
          }
          const analyst = d.analyst as string | undefined;
          if (analyst) next.analyst = analyst;
          const positions = d.positions as Record<string, PositionPayload> | undefined;
          if (positions) next.positions = positions;
        } else if (event === 'analyst_switched') {
          next.analyst = (d.analyst as string) ?? prev.analyst;
        }
      } else if (raw.type === 'equity_update') {
        const eq = d.equity as EquityPoint | undefined;
        if (eq) {
          next.equityCurve = [...prev.equityCurve, eq].slice(-MAX_EQUITY);
        }
      } else if (raw.type === 'equity_batch') {
        const updates = (d.updates as EquityPoint[] | undefined) ?? [];
        if (updates.length) {
          next.equityCurve = [...prev.equityCurve, ...updates].slice(-MAX_EQUITY);
        }
      } else if (raw.type === 'trade_executed') {
        const trade = d.trade as TradeRecord | undefined;
        if (trade) {
          next.trades = [trade, ...prev.trades].slice(0, MAX_TRADES);
        }
      } else if (raw.type === 'trades_batch') {
        const trades = (d.trades as TradeRecord[] | undefined) ?? [];
        if (trades.length) {
          next.trades = [...trades, ...prev.trades].slice(0, MAX_TRADES);
        }
      } else if (raw.type === 'session_failed' || raw.type === 'error_occurred') {
        next.error = (d.error as string) ?? 'unknown error';
      }
      return next;
    });
  }, []);

  useEffect(() => {
    if (!sessionId) {
      setState(INITIAL_STATE);
      return;
    }
    setState((prev) => ({ ...prev, sessionId }));

    let disposed = false;

    const connect = () => {
      const token = getToken();
      const url = token
        ? `${WS_BASE}/ws/brooks/${sessionId}?token=${encodeURIComponent(token)}`
        : `${WS_BASE}/ws/brooks/${sessionId}`;
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        if (disposed) {
          ws.close();
          return;
        }
        setState((prev) => ({ ...prev, connected: true, error: null }));
        ws.send(JSON.stringify({ type: 'subscribe', session_id: sessionId }));
      };
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === 'ping') {
            ws.send(JSON.stringify({ type: 'pong' }));
            return;
          }
          handleEvent(data);
        } catch (e) {
          console.error('[BrooksLive WS] parse error', e);
        }
      };
      ws.onerror = () => {
        setState((prev) => ({ ...prev, connected: false }));
      };
      ws.onclose = () => {
        setState((prev) => ({ ...prev, connected: false }));
        if (!disposed) {
          reconnectRef.current = setTimeout(connect, 2500);
        }
      };
    };

    // Bootstrap state from /brooks-live/{id}/state so UI isn't blank on reconnect.
    (async () => {
      try {
        const resp = await apiFetch(`/brooks-live/${sessionId}/state`);
        if (!resp.ok) return;
        const snap = await resp.json();
        setState((prev) => ({
          ...prev,
          lastRegime: snap.last_regime ?? prev.lastRegime,
          lastDecision: snap.last_decision ?? prev.lastDecision,
          recentSignals: (snap.recent_signals || prev.recentSignals) as DecisionPayload[],
          equityCurve: (snap.equity_points || prev.equityCurve) as EquityPoint[],
          trades: (snap.trades || prev.trades) as TradeRecord[],
          analyst: snap.analyst ?? prev.analyst,
          symbol: snap.config?.symbol ?? prev.symbol,
        }));
      } catch {
        /* ignore bootstrap failure — the WS stream will populate state */
      }
    })();

    connect();

    return () => {
      disposed = true;
      if (reconnectRef.current) clearTimeout(reconnectRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [sessionId, handleEvent]);

  const switchAnalyst = useCallback(async (analyst: string) => {
    if (!sessionId) return;
    const resp = await apiFetch(`/brooks-live/${sessionId}/switch`, {
      method: 'POST',
      body: JSON.stringify({ analyst }),
    });
    if (!resp.ok) throw new Error(`switch failed: ${resp.status}`);
    setState((prev) => ({ ...prev, analyst }));
  }, [sessionId]);

  const stopSession = useCallback(async () => {
    if (!sessionId) return;
    const resp = await apiFetch(`/brooks-live/${sessionId}/stop`, { method: 'POST' });
    if (!resp.ok) throw new Error(`stop failed: ${resp.status}`);
  }, [sessionId]);

  return { state, switchAnalyst, stopSession };
}
