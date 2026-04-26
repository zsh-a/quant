import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { useStudioStore } from '../store';
import {
  collectSidebarSignals,
  DEFAULT_FILTER,
  SignalSidebar,
  sourceKind,
} from '../components/side/SignalSidebar';
import { makeSignal, makeTimeline } from './fixtures';
import type { Signal } from '../types';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

describe('sourceKind helper', () => {
  it('extracts the prefix before colon', () => {
    expect(sourceKind('rule:h2')).toBe('rule');
    expect(sourceKind('llm:claude-opus-4-7')).toBe('llm');
    expect(sourceKind('vlm:vision-x')).toBe('vlm');
  });

  it('returns null for unknown / non-string sources', () => {
    expect(sourceKind('mystery:thing')).toBeNull();
    expect(sourceKind(undefined)).toBeNull();
    expect(sourceKind(42)).toBeNull();
  });
});

describe('collectSidebarSignals', () => {
  function buildTimelineWithSignals() {
    const tl = makeTimeline(5);
    tl.events[1].signals = [
      makeSignal({ id: 's1', side: 'long', source: 'rule:h2', probability: 0.6 }),
    ];
    tl.events[2].signals = [
      makeSignal({ id: 's2', side: 'short', source: 'llm:claude', probability: 0.4 }),
      makeSignal({ id: 's3', side: 'long', source: 'rule:wedge', probability: 0.85 }),
    ];
    tl.events[4].signals = [
      makeSignal({ id: 's4', side: 'long', source: 'vlm:eye', probability: 0.5 }),
    ];
    return tl;
  }

  it('hides signals from future bars (no info leak)', () => {
    const tl = buildTimelineWithSignals();
    expect(collectSidebarSignals(tl, 1, DEFAULT_FILTER).map((r) => r.id)).toEqual(['s1']);
    expect(collectSidebarSignals(tl, 2, DEFAULT_FILTER).map((r) => r.id).sort()).toEqual([
      's1',
      's2',
      's3',
    ]);
  });

  it('returns most-recent first', () => {
    const tl = buildTimelineWithSignals();
    const rows = collectSidebarSignals(tl, 4, DEFAULT_FILTER);
    expect(rows[0].id).toBe('s4');
    expect(rows[rows.length - 1].id).toBe('s1');
  });

  it('filters by source kind', () => {
    const tl = buildTimelineWithSignals();
    const rule = collectSidebarSignals(tl, 4, { ...DEFAULT_FILTER, source: 'rule' });
    expect(rule.map((r) => r.id).sort()).toEqual(['s1', 's3']);

    const llm = collectSidebarSignals(tl, 4, { ...DEFAULT_FILTER, source: 'llm' });
    expect(llm.map((r) => r.id)).toEqual(['s2']);
  });

  it('filters by side', () => {
    const tl = buildTimelineWithSignals();
    const longs = collectSidebarSignals(tl, 4, { ...DEFAULT_FILTER, side: 'long' });
    expect(longs.map((r) => r.id).sort()).toEqual(['s1', 's3', 's4']);

    const shorts = collectSidebarSignals(tl, 4, { ...DEFAULT_FILTER, side: 'short' });
    expect(shorts.map((r) => r.id)).toEqual(['s2']);
  });

  it('drops signals below minimum probability', () => {
    const tl = buildTimelineWithSignals();
    const high = collectSidebarSignals(tl, 4, { ...DEFAULT_FILTER, minProbability: 0.7 });
    expect(high.map((r) => r.id)).toEqual(['s3']);
  });

  it('falls back to event bar_idx when signal lacks bar idx fields', () => {
    const tl = makeTimeline(3);
    tl.events[2].signals = [{ id: 'noidx', side: 'long' } as Signal];
    expect(collectSidebarSignals(tl, 2, DEFAULT_FILTER)[0].bar_idx).toBe(2);
  });

  it('honors signal_bar_idx override when present', () => {
    const tl = makeTimeline(3);
    tl.events[2].signals = [
      makeSignal({ id: 'override', side: 'long', signal_bar_idx: 1 }),
    ];
    const rows = collectSidebarSignals(tl, 2, DEFAULT_FILTER);
    expect(rows[0].bar_idx).toBe(1);
  });
});

describe('SignalSidebar component', () => {
  function loadFixture() {
    const tl = makeTimeline(5);
    tl.events[1].signals = [
      makeSignal({ id: 's1', side: 'long', source: 'rule:h2', probability: 0.6, pattern: 'h2' }),
    ];
    tl.events[2].signals = [
      makeSignal({ id: 's2', side: 'short', source: 'llm:claude', probability: 0.4, pattern: 'wedge' }),
    ];
    useStudioStore.getState().actions.setTimeline(tl);
  }

  it('renders rows for past signals and labels them', () => {
    loadFixture();
    render(<SignalSidebar />);
    expect(screen.getByTestId('signal-row-s1')).toBeInTheDocument();
    expect(screen.getByTestId('signal-row-s2')).toBeInTheDocument();
    expect(screen.getByText('h2')).toBeInTheDocument();
    expect(screen.getByText('wedge')).toBeInTheDocument();
  });

  it('clicking a row jumps to its bar and pins the selection', () => {
    loadFixture();
    render(<SignalSidebar />);
    act(() => {
      fireEvent.click(screen.getByTestId('signal-row-s1'));
    });
    const s = useStudioStore.getState();
    expect(s.currentBarIdx).toBe(1);
    expect(s.selectedSignalId).toBe('s1');
  });

  it('shows the preview badge when chart hover differs from current bar', () => {
    loadFixture();
    const { actions } = useStudioStore.getState();
    actions.setBar(4);
    actions.setHoveredBar(1);
    render(<SignalSidebar />);
    expect(screen.getByText('preview')).toBeInTheDocument();
  });

  it('clicking a side filter excludes the opposite side', () => {
    loadFixture();
    render(<SignalSidebar />);
    act(() => {
      fireEvent.click(screen.getByTestId('filter-side-long'));
    });
    expect(screen.getByTestId('signal-row-s1')).toBeInTheDocument();
    expect(screen.queryByTestId('signal-row-s2')).toBeNull();
  });
});
