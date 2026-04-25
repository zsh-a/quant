import { afterEach, beforeEach, describe, expect, it } from 'vitest';
import { act, render } from '@testing-library/react';
import { useReplay } from '../hooks/useReplay';
import { LIVE_TAIL, useStudioStore } from '../store';
import { makeTimeline } from './fixtures';

function Harness() {
  useReplay();
  return <input data-testid="input" />;
}

const reset = () => useStudioStore.getState().actions.reset();

beforeEach(reset);
afterEach(reset);

function press(key: string, init: KeyboardEventInit = {}) {
  act(() => {
    window.dispatchEvent(new KeyboardEvent('keydown', { key, ...init }));
  });
}

describe('useReplay keyboard shortcuts', () => {
  it('arrow keys step the bar index', () => {
    useStudioStore.getState().actions.setTimeline(makeTimeline(10));
    render(<Harness />);

    press('ArrowLeft');
    expect(useStudioStore.getState().currentBarIdx).toBe(8); // started at LIVE_TAIL → resolves to 9, then -1
    press('ArrowRight');
    expect(useStudioStore.getState().currentBarIdx).toBe(9);
    press('ArrowLeft', { shiftKey: true });
    expect(useStudioStore.getState().currentBarIdx).toBe(0); // -10 from 9 clamps to 0
    press('ArrowRight', { shiftKey: true });
    expect(useStudioStore.getState().currentBarIdx).toBe(9); // +10 from 0 clamps to 9
  });

  it('Space toggles play state when bars exist', () => {
    useStudioStore.getState().actions.setTimeline(makeTimeline(3));
    render(<Harness />);

    press(' ');
    expect(useStudioStore.getState().playState).toBe('playing');
    press(' ');
    expect(useStudioStore.getState().playState).toBe('paused');
  });

  it('[ and ] cycle the speed ladder', () => {
    render(<Harness />);
    expect(useStudioStore.getState().speed).toBe(1);
    press(']');
    expect(useStudioStore.getState().speed).toBe(2);
    press(']');
    press(']');
    press(']'); // clamps at 10
    expect(useStudioStore.getState().speed).toBe(10);
    press('[');
    expect(useStudioStore.getState().speed).toBe(5);
  });

  it('End jumps to live, Home jumps to bar 0', () => {
    useStudioStore.getState().actions.setTimeline(makeTimeline(5));
    useStudioStore.getState().actions.setBar(3);
    render(<Harness />);

    press('End');
    expect(useStudioStore.getState().currentBarIdx).toBe(LIVE_TAIL);
    press('Home');
    expect(useStudioStore.getState().currentBarIdx).toBe(0);
  });

  it('does not handle keys while a text input is focused', () => {
    useStudioStore.getState().actions.setTimeline(makeTimeline(5));
    const { getByTestId } = render(<Harness />);
    const input = getByTestId('input') as HTMLInputElement;
    input.focus();

    act(() => {
      input.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
    });
    expect(useStudioStore.getState().currentBarIdx).toBe(LIVE_TAIL);
  });
});
