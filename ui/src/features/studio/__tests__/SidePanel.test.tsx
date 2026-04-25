import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { useStudioStore } from '../store';
import { SidePanel } from '../components/side/SidePanel';
import { makeDecision, makeRegime, makeSignal, makeTimeline } from './fixtures';

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

function loadFixture() {
  const tl = makeTimeline(4);
  tl.events[0].regime = makeRegime('strong_bull_trend');
  tl.events[1].regime = makeRegime('tight_tr');
  tl.events[2].regime = makeRegime('strong_bear_trend');
  tl.events[1].signals = [makeSignal({ id: 's1', side: 'long', source: 'rule:h2' })];
  tl.events[2].decision = makeDecision({ pattern: 'wedge', side: 'short' });
  const { actions } = useStudioStore.getState();
  actions.setTimeline(tl);
  actions.setBar(2);
}

describe('SidePanel tab container', () => {
  it('renders all three tab triggers', () => {
    loadFixture();
    render(<SidePanel />);
    expect(screen.getByTestId('tab-signals')).toBeInTheDocument();
    expect(screen.getByTestId('tab-decision')).toBeInTheDocument();
    expect(screen.getByTestId('tab-regime')).toBeInTheDocument();
  });

  it('defaults to the decision tab', () => {
    loadFixture();
    render(<SidePanel />);
    expect(screen.getByTestId('decision-inspector')).toBeInTheDocument();
  });

  it('switching to the signals tab reveals the sidebar', async () => {
    loadFixture();
    const user = userEvent.setup();
    render(<SidePanel />);
    await user.click(screen.getByTestId('tab-signals'));
    expect(screen.getByTestId('signal-sidebar')).toBeInTheDocument();
  });

  it('switching to the regime tab reveals the timeline', async () => {
    loadFixture();
    const user = userEvent.setup();
    render(<SidePanel />);
    await user.click(screen.getByTestId('tab-regime'));
    expect(screen.getByTestId('regime-timeline')).toBeInTheDocument();
  });
});
