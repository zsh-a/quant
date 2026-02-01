/**
 * WebSocket hook for real-time session updates
 * Provides auto-reconnect and fallback to HTTP polling
 */

import { useEffect, useRef, useState, useCallback } from 'react';

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

const WS_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'ws://localhost:8000'
    : `ws://${window.location.hostname}:8000`;

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
    const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    const pollingIntervalRef = useRef<NodeJS.Timeout | null>(null);
    const lastMessageTimeRef = useRef<string>('');

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

    // Connect to WebSocket
    const connect = useCallback(() => {
        if (!enabled || !sessionId) return;

        try {
            const ws = new WebSocket(`${WS_BASE}/ws/${sessionId}`);
            wsRef.current = ws;

            ws.onopen = () => {
                console.log(`[WebSocket] Connected to session: ${sessionId}`);
                setIsConnected(true);
                setConnectionAttempts(0);
                setUsePolling(false);

                // Send subscribe message
                ws.send(JSON.stringify({
                    type: 'subscribe',
                    session_id: sessionId
                }));

                onConnect?.();
            };

            ws.onmessage = (event) => {
                try {
                    const message: WebSocketMessage = JSON.parse(event.data);

                    // Handle ping/pong
                    if (message.type === 'ping') {
                        ws.send(JSON.stringify({ type: 'pong' }));
                        return;
                    }

                    // Update last message time for polling fallback
                    lastMessageTimeRef.current = message.timestamp;

                    onMessage?.(message);
                } catch (error) {
                    console.error('[WebSocket] Failed to parse message:', error);
                }
            };

            ws.onerror = (error) => {
                console.error('[WebSocket] Error:', error);
                onError?.(error);
            };

            ws.onclose = () => {
                console.log('[WebSocket] Disconnected');
                setIsConnected(false);
                wsRef.current = null;
                onDisconnect?.();

                // Attempt to reconnect
                if (enabled && connectionAttempts < 5) {
                    const delay = Math.min(1000 * Math.pow(2, connectionAttempts), 30000);
                    console.log(`[WebSocket] Reconnecting in ${delay}ms...`);

                    reconnectTimeoutRef.current = setTimeout(() => {
                        setConnectionAttempts(prev => prev + 1);
                        connect();
                    }, delay);
                } else if (fallbackToPolling) {
                    // Fall back to HTTP polling after max reconnect attempts
                    console.log('[WebSocket] Max reconnect attempts reached, falling back to polling');
                    setUsePolling(true);
                }
            };

        } catch (error) {
            console.error('[WebSocket] Connection failed:', error);
            if (fallbackToPolling) {
                setUsePolling(true);
            }
        }
    }, [enabled, sessionId, connectionAttempts, fallbackToPolling, onConnect, onMessage, onDisconnect, onError]);

    // HTTP polling fallback
    useEffect(() => {
        if (!usePolling || !enabled || !sessionId) return;

        console.log('[Polling] Starting HTTP polling fallback');

        const poll = async () => {
            try {
                const url = lastMessageTimeRef.current
                    ? `http://localhost:8000/session/${sessionId}/status?since=${encodeURIComponent(lastMessageTimeRef.current)}`
                    : `http://localhost:8000/session/${sessionId}/status`;

                const response = await fetch(url);
                const data = await response.json();

                // Simulate WebSocket message format
                if (data.equity_history && data.equity_history.length > 0) {
                    const lastEquity = data.equity_history[data.equity_history.length - 1];
                    lastMessageTimeRef.current = lastEquity.timestamp;

                    onMessage?.({
                        type: 'session_progress',
                        session_id: sessionId,
                        timestamp: new Date().toISOString(),
                        data: {
                            progress: data.progress || 0,
                            status: data.status || 'unknown'
                        }
                    });
                }
            } catch (error) {
                console.error('[Polling] Error:', error);
            }
        };

        // Initial poll
        poll();

        // Set up polling interval
        pollingIntervalRef.current = setInterval(poll, pollingInterval);

        return () => {
            if (pollingIntervalRef.current) {
                clearInterval(pollingIntervalRef.current);
            }
        };
    }, [usePolling, enabled, sessionId, pollingInterval, onMessage]);

    // Connect on mount or when sessionId changes
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
