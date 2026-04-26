import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { useStudioStore } from '../store';
import {
  buildRegimeSegments,
  RegimeTimeline,
} from '../components/side/RegimeTimeline';
import { makeRegime, makeTimeline } from './fixtures';
import type { SessionTimeline } from '../types';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

function withRegimes(): SessionTimeline {
  const tl = makeTimeline(6);
  tl.events[0].regime = makeRegime('strong_bull_trend');
  tl.events[1].regime = makeRegime('strong_bull_trend');
  tl.events[2].regime = makeRegime('strong_bull_trend');
  tl.events[3].regime = makeRegime('tight_tr');
  tl.events[4].regime = makeRegime('strong_bear_trend');
  tl.events[5].regime = makeRegime('strong_bear_trend');
  return tl;
}

describe('buildRegimeSegments', () => {
  it('groups consecutive bars with the same regime', () => {
    const segs = buildRegimeSegments(withRegimes());
    expect(segs.length).toBe(3);
    expect(segs[0]).toMatchObject({ name: 'strong_bull_trend', start: 0, end: 2 });
    expect(segs[1]).toMatchObject({ name: 'tight_tr', start: 3, end: 3 });
    expect(segs[2]).toMatchObject({ name: 'strong_bear_trend', start: 4, end: 5 });
  });

  it('treats bars without a regime as the synthetic "unknown" segment', () => {
    const tl = makeTimeline(3);
    tl.events[0].regime = makeRegime('strong_bull_trend');
    // events[1], events[2] have no regime set
    const segs = buildRegimeSegments(tl);
    expect(segs.map((s) => s.name)).toEqual(['strong_bull_trend', 'unknown']);
  });

  it('returns one segment when all bars share a regime', () => {
    const tl = makeTimeline(4);
    for (const ev of tl.events) ev.regime = makeRegime('channel');
    const segs = buildRegimeSegments(tl);
    expect(segs.length).toBe(1);
    expect(segs[0]).toMatchObject({ start: 0, end: 3, name: 'channel' });
  });
});

describe('RegimeTimeline component', () => {
  it('renders empty state when no timeline is loaded', () => {
    render(<RegimeTimeline />);
    expect(screen.getByTestId('regime-timeline-empty')).toBeInTheDocument();
  });

  it('renders one button per segment with regime data attribute', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(withRegimes());
    render(<RegimeTimeline />);
    expect(screen.getByTestId('regime-segment-0')).toHaveAttribute(
      'data-regime-name',
      'strong_bull_trend',
    );
    expect(screen.getByTestId('regime-segment-1')).toHaveAttribute('data-regime-name', 'tight_tr');
    expect(screen.getByTestId('regime-segment-2')).toHaveAttribute(
      'data-regime-name',
      'strong_bear_trend',
    );
  });

  it('clicking a segment jumps to its starting bar', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(withRegimes());
    render(<RegimeTimeline />);
    act(() => {
      fireEvent.click(screen.getByTestId('regime-segment-2'));
    });
    expect(useStudioStore.getState().currentBarIdx).toBe(4);
  });

  it('hovering a segment surfaces stats in the tooltip', () => {
    const { actions } = useStudioStore.getState();
    actions.setTimeline(withRegimes());
    render(<RegimeTimeline />);
    expect(screen.getByTestId('regime-tooltip-empty')).toBeInTheDocument();
    act(() => {
      fireEvent.mouseEnter(screen.getByTestId('regime-segment-0'));
    });
    const tip = screen.getByTestId('regime-tooltip');
    expect(tip).toHaveTextContent('strong_bull_trend');
    expect(tip).toHaveTextContent('bars 0-2');
  });
});
