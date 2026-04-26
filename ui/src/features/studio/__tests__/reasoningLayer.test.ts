import { describe, expect, it, vi } from 'vitest';

const markerCalls: { setMarkers: ReturnType<typeof vi.fn>; markers: unknown[]; detach: ReturnType<typeof vi.fn> }[] = [];

vi.mock('lightweight-charts', () => ({
  createSeriesMarkers: vi.fn(() => {
    const handle: any = {
      markers: [] as unknown[],
      setMarkers: vi.fn(function (this: any, m: unknown[]) {
        this.markers = m;
      }),
      detach: vi.fn(),
    };
    handle.setMarkers = vi.fn((m: unknown[]) => {
      handle.markers = m;
    });
    markerCalls.push(handle);
    return handle;
  }),
}));

import {
  buildReasoningMarkers,
  reasoningLayer,
  reasoningSnippet,
} from '../layers/reasoning';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';

describe('reasoning layer', () => {
  it('extracts a trimmed snippet from decision reasoning first', () => {
    const snippet = reasoningSnippet({
      bar_idx: 1,
      timestamp_ns: 0,
      signals: [{ reasoning: 'fallback' } as any],
      decision: { reasoning: '  multi   line\n reasoning ' } as any,
    });
    expect(snippet?.startsWith('multi line reasoning')).toBe(true);
  });

  it('falls back to signal reasoning when decision is missing', () => {
    expect(
      reasoningSnippet({ bar_idx: 0, timestamp_ns: 0, signals: [{ reasoning: 'sig' } as any] }),
    ).toBe('sig');
  });

  it('truncates very long reasoning with an ellipsis', () => {
    const long = 'x'.repeat(200);
    const snippet = reasoningSnippet({
      bar_idx: 0,
      timestamp_ns: 0,
      signals: [],
      decision: { reasoning: long } as any,
    });
    expect(snippet?.length).toBeLessThanOrEqual(60);
    expect(snippet?.endsWith('…')).toBe(true);
  });

  it('drops markers for bars beyond currentBarIdx', () => {
    const tl = makeTimeline(5);
    tl.events[1].decision = { reasoning: 'A' } as any;
    tl.events[3].decision = { reasoning: 'B' } as any;
    expect(buildReasoningMarkers(tl, 2).length).toBe(1);
    expect(buildReasoningMarkers(tl, 4).length).toBe(2);
  });

  it('mounts a markers plugin and sets markers on update', () => {
    const before = markerCalls.length;
    const { ctx } = makeLayerCtx();
    const handle = reasoningLayer.mount(ctx);
    expect(markerCalls.length).toBe(before + 1);
    const recorded = markerCalls[markerCalls.length - 1];

    const tl = makeTimeline(3);
    tl.events[1].decision = { reasoning: 'hi' } as any;
    handle.update(tl, 2);
    expect((recorded.markers as unknown[]).length).toBe(1);

    handle.unmount();
  });
});
