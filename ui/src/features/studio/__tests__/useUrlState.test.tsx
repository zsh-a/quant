import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, renderHook } from '@testing-library/react';
import { MemoryRouter, Route, Routes, useLocation } from 'react-router-dom';
import { useStudioStore } from '../store';
import { useUrlState } from '../hooks/useUrlState';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

function makeWrapper(initialPath: string) {
  const Wrapper: React.FC<{ children: React.ReactNode }> = ({ children }) => (
    <MemoryRouter initialEntries={[initialPath]}>
      <Routes>
        <Route path="/studio/:sessionId" element={<>{children}</>} />
      </Routes>
    </MemoryRouter>
  );
  return Wrapper;
}

function useUrlStateWithLocation() {
  useUrlState();
  return useLocation();
}

describe('useUrlState', () => {
  it('reads bar / mode / speed from URL on mount', () => {
    const Wrapper = makeWrapper('/studio/abc?bar=42&mode=replay&speed=5');
    renderHook(() => useUrlStateWithLocation(), { wrapper: Wrapper });

    const s = useStudioStore.getState();
    expect(s.currentBarIdx).toBe(42);
    expect(s.mode).toBe('replay');
    expect(s.speed).toBe(5);
  });

  it('writes store changes back to the URL via replaceState', () => {
    const Wrapper = makeWrapper('/studio/abc');
    const { result } = renderHook(() => useUrlStateWithLocation(), { wrapper: Wrapper });

    expect(result.current.search).toBe('');

    act(() => {
      useStudioStore.getState().actions.setTimeline({
        session_id: 'abc',
        symbol: 'X',
        base_interval: '5m',
        htf_intervals: [],
        bars: Array.from({ length: 50 }, (_, i) => ({
          timestamp_ns: i * 1_000_000_000,
          open: 1,
          high: 1,
          low: 1,
          close: 1,
          volume: 0,
        })),
        htf_bars: {},
        events: [],
        pnl_curve: [],
        config: {},
        created_at: 'x',
      });
      useStudioStore.getState().actions.setBar(7);
      useStudioStore.getState().actions.setSpeed(2);
    });

    expect(result.current.search).toContain('bar=7');
    expect(result.current.search).toContain('speed=2');
    expect(result.current.search).not.toContain('mode=');
  });

  it('omits ?bar from URL when currentBarIdx is LIVE_TAIL', () => {
    const Wrapper = makeWrapper('/studio/abc?bar=3');
    const { result } = renderHook(() => useUrlStateWithLocation(), { wrapper: Wrapper });
    expect(useStudioStore.getState().currentBarIdx).toBe(3);

    act(() => {
      useStudioStore.getState().actions.jumpToLive();
    });
    expect(result.current.search).not.toContain('bar=');
  });

  it('ignores invalid query values (non-numeric bar, unknown mode, off-ladder speed)', () => {
    const Wrapper = makeWrapper('/studio/abc?bar=oops&mode=hyperdrive&speed=99');
    renderHook(() => useUrlStateWithLocation(), { wrapper: Wrapper });
    const s = useStudioStore.getState();
    // Defaults preserved.
    expect(s.currentBarIdx).toBe(-1);
    expect(s.mode).toBe('live');
    expect(s.speed).toBe(1);
  });
});
