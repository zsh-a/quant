/**
 * Bidirectional sync between query params and the studio store.
 *
 *   ?bar=N          ↔ currentBarIdx (omitted when LIVE_TAIL)
 *   ?mode=live|replay ↔ mode
 *   ?speed=N        ↔ speed (omitted when 1)
 *
 * URL → store on mount and whenever the search string changes externally.
 * Store → URL writes via `replaceState` so we never push history entries.
 */

import { useEffect, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import {
  LIVE_TAIL,
  useCurrentBarIdx,
  useStudioActions,
  useStudioMode,
  useStudioSpeed,
} from '../store';
import { STUDIO_SPEEDS, type StudioMode, type StudioSpeed } from '../types';

function parseBar(raw: string | null): number | null {
  if (raw === null || raw === '') return null;
  const n = Number(raw);
  if (!Number.isFinite(n) || !Number.isInteger(n) || n < 0) return null;
  return n;
}

function parseMode(raw: string | null): StudioMode | null {
  if (raw === 'live' || raw === 'replay') return raw;
  return null;
}

function parseSpeed(raw: string | null): StudioSpeed | null {
  if (raw === null || raw === '') return null;
  const n = Number(raw);
  return (STUDIO_SPEEDS as readonly number[]).includes(n) ? (n as StudioSpeed) : null;
}

export function useUrlState() {
  const location = useLocation();
  const navigate = useNavigate();
  const actions = useStudioActions();

  const currentBarIdx = useCurrentBarIdx();
  const mode = useStudioMode();
  const speed = useStudioSpeed();

  // Track last URL we wrote, so we can ignore the round-trip echo.
  const lastWrittenSearchRef = useRef<string | null>(null);

  // URL → store
  useEffect(() => {
    if (lastWrittenSearchRef.current === location.search) {
      return;
    }
    const params = new URLSearchParams(location.search);
    const bar = parseBar(params.get('bar'));
    const m = parseMode(params.get('mode'));
    const s = parseSpeed(params.get('speed'));

    if (bar !== null) actions.setBar(bar);
    if (m !== null) actions.setMode(m);
    if (s !== null) actions.setSpeed(s);
  }, [location.search, actions]);

  // store → URL
  useEffect(() => {
    const params = new URLSearchParams(location.search);

    if (currentBarIdx === LIVE_TAIL) {
      params.delete('bar');
    } else {
      params.set('bar', String(currentBarIdx));
    }

    if (mode === 'live') {
      params.delete('mode');
    } else {
      params.set('mode', mode);
    }

    if (speed === 1) {
      params.delete('speed');
    } else {
      params.set('speed', String(speed));
    }

    const next = params.toString();
    const nextSearch = next ? `?${next}` : '';
    if (nextSearch === location.search) return;

    lastWrittenSearchRef.current = nextSearch;
    navigate({ pathname: location.pathname, search: nextSearch }, { replace: true });
    // location.search is intentionally captured to detect external edits via
    // the URL→store effect above; we rely on `lastWrittenSearchRef` to avoid
    // ping-pong on round trips.
  }, [currentBarIdx, mode, speed, navigate, location.pathname, location.search]);
}
