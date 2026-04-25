/**
 * Studio API client — talks to the S1 backend (`/brooks-studio/...`).
 *
 * Falls back to the legacy `/brooks-live/{id}/state` snapshot if the studio
 * timeline endpoint is missing (so spike work can run against the existing
 * BrooksLive backend until S1 ships).
 */

import { WS_BASE, apiFetch, getToken } from '../../utils/api';
import type { BarEvent, SessionTimeline } from './types';

export class StudioApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = 'StudioApiError';
  }
}

export async function fetchTimeline(sessionId: string, signal?: AbortSignal): Promise<SessionTimeline> {
  const resp = await apiFetch(`/brooks-studio/sessions/${sessionId}/timeline`, { signal });
  if (!resp.ok) {
    throw new StudioApiError(`Failed to load timeline (${resp.status})`, resp.status);
  }
  return (await resp.json()) as SessionTimeline;
}

export type WsHandlers = {
  onOpen?: () => void;
  onClose?: () => void;
  onError?: (err: Event) => void;
  onEvent: (ev: BarEvent) => void;
};

export interface StudioWsHandle {
  close: () => void;
  send: (payload: unknown) => void;
}

/**
 * Open a WS subscription to the studio bar event channel for a session.
 * Sends `{ type: 'subscribe', session_id }` on open and replies to ping
 * frames automatically. Caller must invoke `close()` to terminate.
 */
export function subscribeStudio(sessionId: string, handlers: WsHandlers): StudioWsHandle {
  const token = getToken();
  const url = token
    ? `${WS_BASE}/ws/brooks-studio/${sessionId}?token=${encodeURIComponent(token)}`
    : `${WS_BASE}/ws/brooks-studio/${sessionId}`;
  const ws = new WebSocket(url);

  ws.onopen = () => {
    try {
      ws.send(JSON.stringify({ type: 'subscribe', session_id: sessionId }));
    } catch {
      /* ignore — onerror will surface socket failures */
    }
    handlers.onOpen?.();
  };

  ws.onmessage = (event) => {
    let data: { type?: string; data?: unknown };
    try {
      data = JSON.parse(event.data);
    } catch (e) {
      console.error('[StudioWS] parse error', e);
      return;
    }
    if (data.type === 'ping') {
      try {
        ws.send(JSON.stringify({ type: 'pong' }));
      } catch {
        /* socket may be closing */
      }
      return;
    }
    if (data.type === 'bar_event' && data.data) {
      handlers.onEvent(data.data as BarEvent);
    }
  };

  ws.onerror = (e) => handlers.onError?.(e);
  ws.onclose = () => handlers.onClose?.();

  return {
    close: () => ws.close(),
    send: (payload: unknown) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(payload));
      }
    },
  };
}
