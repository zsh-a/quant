/**
 * WebSocket hook for real-time session updates
 * Provides auto-reconnect and fallback to HTTP polling
 */

import { useEffect, useRef, useState, useCallback } from 'react';
import { API_BASE, WS_BASE, getToken, apiFetch } from '../utils/api';

interface WebSocketMessage {
    type: string;
    session_id: string;
    timestamp: string;
    data: any;
}

interface UseWebSocketOptions {
    sessionId: string;
    onMessage?: (message: WebSocketMessage) => void;
    onConnect?: () => void;
    onDisconnect?: () => void;
    onError?: (error: Event) => void;
    enabled?: boolean;
    fallbackToPolling?: boolean;
    pollingInterval?: number;
}

export const useWebSocket = ({
    sessionId,
    onMessage,
    onConnect,
    onDisconnect,
    onError,
    enabled = true,
    fallbackToPolling = true,
    pollingInterval = 2000
}: UseWebSocketOptions) => {
    const [isConnected, setIsConnected] = useState(false);
    const [connectionAttempts, setConnectionAttempts] = useState(0);
    const [usePolling, setUsePolling] = useState(false);

    const wsRef = useRef<WebSocket | null>(null);
    const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const pollingIntervalRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const lastMessageTimeRef = useRef<string>('');

    // Refs for callbacks so connect() identity is stable and we don't reconnect on every render
    const onMessageRef = useRef(onMessage);
    const onConnectRef = useRef(onConnect);
    const onDisconnectRef = useRef(onDisconnect);
    const onErrorRef = useRef(onError);
    onMessageRef.current = onMessage;
    onConnectRef.current = onConnect;
    onDisconnectRef.current = onDisconnect;
    onErrorRef.current = onError;

    // Cleanup function
    const cleanup = useCallback(() => {
        if (wsRef.current) {
            wsRef.current.close();
            wsRef.current = null;
        }
        if (reconnectTimeoutRef.current) {
            clearTimeout(reconnectTimeoutRef.current);
            reconnectTimeoutRef.current = null;
        }
        if (pollingIntervalRef.current) {
            clearInterval(pollingIntervalRef.current);
            pollingIntervalRef.current = null;
        }
    }, []);

    // Connect to WebSocket - deps only sessionId/enabled/connectionAttempts so we don't reconnect on every parent re-render
    const connect = useCallback(() => {
        if (!enabled || !sessionId) return;

        try {
            const token = getToken();
            const wsUrl = token
                ? `${WS_BASE}/ws/${sessionId}?token=${encodeURIComponent(token)}`
                : `${WS_BASE}/ws/${sessionId}`;
            const ws = new WebSocket(wsUrl);
            wsRef.current = ws;

            ws.onopen = () => {
                setIsConnected(true);
                setConnectionAttempts(0);
                setUsePolling(false);

                ws.send(JSON.stringify({
                    type: 'subscribe',
                    session_id: sessionId
                }));

                onConnectRef.current?.();
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);

                    if (message.type === 'ping') {
                        ws.send(JSON.stringify({ type: 'pong' }));
                        return;
                    }

                    lastMessageTimeRef.current = message.timestamp;
                    onMessageRef.current?.(message);
                } catch (error) {
                    console.error('[WebSocket] Failed to parse message:', error);
                }
            };

            ws.onerror = (err) => {
                onErrorRef.current?.(err);
            };

            ws.onclose = () => {
                setIsConnected(false);
                wsRef.current = null;
                onDisconnectRef.current?.();

                if (enabled && connectionAttempts < 5) {
                    const delay = Math.min(1000 * Math.pow(2, connectionAttempts), 30000);
                    reconnectTimeoutRef.current = setTimeout(() => {
                        setConnectionAttempts(prev => prev + 1);
                        connect();
                    }, delay);
                } else if (fallbackToPolling) {
                    setUsePolling(true);
                }
            };

        } catch (error) {
            console.error('[WebSocket] Connection failed:', error);
            if (fallbackToPolling) {
                setUsePolling(true);
            }
        }
    }, [enabled, sessionId, connectionAttempts, fallbackToPolling]);

    // HTTP polling fallback - no onMessage in deps, use ref
    useEffect(() => {
        if (!usePolling || !enabled || !sessionId) return;

        const poll = async () => {
            try {
                const path = lastMessageTimeRef.current
                    ? `/session/${sessionId}/status?since=${encodeURIComponent(lastMessageTimeRef.current)}`
                    : `/session/${sessionId}/status`;

                const response = await apiFetch(path);
                if (response.status === 404) {
                    console.warn(`[Polling] Session ${sessionId} not found, stopping polling`);
                    setUsePolling(false);
                    return;
                }
                const data = await response.json();
                const equityHistory = data.equity_history || [];
                const trades = data.trades || [];

                const latestEquityTs = equityHistory.length > 0
                    ? equityHistory[equityHistory.length - 1].timestamp
                    : '';
                const latestTradeTs = trades.length > 0
                    ? trades[trades.length - 1].timestamp
                    : '';
                const latestTimestamp = [latestEquityTs, latestTradeTs]
                    .filter(Boolean)
                    .sort()
                    .slice(-1)[0];

                if (latestTimestamp) {
                    lastMessageTimeRef.current = latestTimestamp;
                }

                onMessageRef.current?.({
                    type: 'session_progress',
                    session_id: sessionId,
                    timestamp: new Date().toISOString(),
                    data: {
                        progress: data.progress || 0,
                        status: data.status || 'unknown'
                    }
                });

                if (equityHistory.length > 0) {
                    onMessageRef.current?.({
                        type: 'equity_batch',
                        session_id: sessionId,
                        timestamp: latestEquityTs || new Date().toISOString(),
                        data: {
                            updates: equityHistory,
                        }
                    });
                }

                if (trades.length > 0) {
                    onMessageRef.current?.({
                        type: 'trades_batch',
                        session_id: sessionId,
                        timestamp: latestTradeTs || new Date().toISOString(),
                        data: {
                            trades,
                        }
                    });
                }
            } catch (error) {
                console.error('[Polling] Error:', error);
            }
        };

        poll();
        pollingIntervalRef.current = setInterval(poll, pollingInterval);

        return () => {
            if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current);
                pollingIntervalRef.current = null;
            }
        };
    }, [usePolling, enabled, sessionId, pollingInterval]);

    // Connect on mount or when sessionId/enabled/usePolling change only
    useEffect(() => {
        if (enabled && sessionId && !usePolling) {
            connect();
        }
        return cleanup;
    }, [enabled, sessionId, usePolling, connect, cleanup]);

    // Send message
    const sendMessage = useCallback((message: any) => {
        if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify(message));
            return true;
        }
        return false;
    }, []);

    // Manual reconnect
    const reconnect = useCallback(() => {
        cleanup();
        setConnectionAttempts(0);
        setUsePolling(false);
        connect();
    }, [cleanup, connect]);

    return {
        isConnected,
        usePolling,
        sendMessage,
        reconnect,
        connectionAttempts
    };
};
