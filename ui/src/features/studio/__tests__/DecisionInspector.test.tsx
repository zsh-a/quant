import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen } from '@testing-library/react';
import { useStudioStore } from '../store';
import {
  DecisionInspector,
  decisionSupportingSignals,
  expectedRTone,
  probabilityBucket,
} from '../components/side/DecisionInspector';
import { makeDecision, makeSignal, makeTimeline } from './fixtures';
import type { Decision } from '../types';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

describe('probabilityBucket', () => {
  it('partitions probabilities into low / mid / high / top', () => {
    expect(probabilityBucket(0.3).tone).toBe('low');
    expect(probabilityBucket(0.55).tone).toBe('mid');
    expect(probabilityBucket(0.7).tone).toBe('high');
    expect(probabilityBucket(0.92).tone).toBe('top');
  });
});

describe('expectedRTone', () => {
  it('returns negative tone for losing E[R]', () => {
    expect(expectedRTone(-0.5)).toContain('rose');
  });
  it('returns positive tone for winning E[R]', () => {
    expect(expectedRTone(2.0)).toContain('emerald');
  });
});

describe('decisionSupportingSignals', () => {
  it('returns the signals array when present', () => {
    const d: Decision = makeDecision();
    (d as Record<string, unknown>).signals = [makeSignal({ id: 'a' }), makeSignal({ id: 'b' })];
    expect(decisionSupportingSignals(d).length).toBe(2);
  });
  it('returns an empty array when missing or wrong type', () => {
    expect(decisionSupportingSignals(makeDecision())).toEqual([]);
  });
});

describe('DecisionInspector component', () => {
  function loadFixture(decision: Decision | null = makeDecision()) {
    const tl = makeTimeline(5);
    if (decision) {
      tl.events[2].decision = decision;
    }
    const { actions } = useStudioStore.getState();
    actions.setTimeline(tl);
    actions.setBar(2);
  }

  it('renders empty placeholder when no decision exists', () => {
    loadFixture(null);
    render(<DecisionInspector />);
    expect(screen.getByTestId('decision-inspector-empty')).toBeInTheDocument();
  });

  it('shows side, pattern, regime and htf-aligned tag', () => {
    loadFixture(makeDecision({ side: 'long', pattern: 'h2', regime: 'strong_bull_trend', htf_aligned: true }));
    render(<DecisionInspector />);
    expect(screen.getByTestId('decision-side-badge')).toHaveTextContent('long');
    expect(screen.getByTestId('decision-regime')).toHaveTextContent('strong_bull_trend');
    expect(screen.getByTestId('decision-htf-aligned')).toBeInTheDocument();
    expect(screen.queryByTestId('decision-htf-misaligned')).toBeNull();
  });

  it('renders the misaligned tag when htf_aligned is false', () => {
    loadFixture(makeDecision({ htf_aligned: false }));
    render(<DecisionInspector />);
    expect(screen.getByTestId('decision-htf-misaligned')).toBeInTheDocument();
  });

  it('renders entry / stop / target cells with distance to current price', () => {
    const tl = makeTimeline(5);
    tl.bars[2] = { ...tl.bars[2], close: 100 };
    tl.events[2].decision = makeDecision({ entry_px: 105, stop_px: 95, target_px: 120 });
    const { actions } = useStudioStore.getState();
    actions.setTimeline(tl);
    actions.setBar(2);

    render(<DecisionInspector />);
    const levels = screen.getByTestId('decision-price-levels');
    expect(levels).toHaveTextContent('entry');
    expect(levels).toHaveTextContent('stop');
    expect(levels).toHaveTextContent('target');
    // Distance text appears in parentheses with sign and percentage.
    expect(levels.textContent).toMatch(/\+5/);
    expect(levels.textContent).toMatch(/-5/);
  });

  it('renders reasoning markdown including code blocks', () => {
    loadFixture(
      makeDecision({
        reasoning: '## Why\nbreakout above **EMA20**\n\n```\npx > ema20\n```',
      }),
    );
    render(<DecisionInspector />);
    const block = screen.getByTestId('decision-reasoning');
    expect(block).toBeInTheDocument();
    expect(block.querySelector('h2')).toBeTruthy();
    expect(block.querySelector('strong')).toBeTruthy();
    expect(block.querySelector('pre code')).toBeTruthy();
  });

  it('falls back to current bar when hovered idx clears', () => {
    loadFixture(makeDecision({ pattern: 'h2' }));
    const { actions } = useStudioStore.getState();
    actions.setHoveredBar(0); // events[0].decision is null in fixture
    const { rerender } = render(<DecisionInspector />);
    expect(screen.getByTestId('decision-inspector-empty')).toBeInTheDocument();

    act(() => {
      actions.setHoveredBar(null);
    });
    rerender(<DecisionInspector />);
    expect(screen.queryByTestId('decision-inspector-empty')).toBeNull();
    expect(screen.getByTestId('decision-side-badge')).toBeInTheDocument();
  });

  it('copy json button emits decision JSON', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    });

    loadFixture(makeDecision({ pattern: 'h2', probability: 0.7 }));
    render(<DecisionInspector />);
    fireEvent.click(screen.getByTestId('decision-copy-json'));
    expect(writeText).toHaveBeenCalled();
    const arg = writeText.mock.calls[0][0] as string;
    expect(arg).toContain('"pattern": "h2"');
  });
});
