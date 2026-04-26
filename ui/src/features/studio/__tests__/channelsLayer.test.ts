import { describe, expect, it, vi } from 'vitest';

vi.mock('lightweight-charts', () => ({
  LineSeries: { type: 'Line' },
  LineStyle: { Solid: 0, Dotted: 1, Dashed: 2 },
}));

import {
  buildChannelLineData,
  buildChannelMidlineData,
  channelsLayer,
  latestStructure,
} from '../layers/channels';
import { makeLayerCtx } from './mockChart';
import { makeTimeline } from './fixtures';

describe('channels layer', () => {
  it('returns the latest structure at-or-before the cursor', () => {
    const tl = makeTimeline(5);
    tl.events[2].structure = {
      always_in: 'long',
      confirmed_swings: [],
      micro_channel_top: { slope: 0.5, intercept: 100, start: 1, end: 4 },
      micro_channel_bot: null,
    };
    expect(latestStructure(tl, 4)?.micro_channel_top?.slope).toBe(0.5);
    expect(latestStructure(tl, 1)).toBeNull();
  });

  it('builds two endpoints from the channel fit clamped to the cursor', () => {
    const tl = makeTimeline(5);
    const channel = { slope: 1, intercept: 100, start: 1, end: 4 };
    const data = buildChannelLineData(tl, channel, 3);
    expect(data.length).toBe(2);
    // y = 1*x + 100; clamped end = 3
    expect(data[0].value).toBe(101); // x=1
    expect(data[1].value).toBe(103); // x=3
  });

  it('returns empty data when no channel or end <= start', () => {
    const tl = makeTimeline(5);
    expect(buildChannelLineData(tl, null, 3).length).toBe(0);
    expect(buildChannelLineData(tl, undefined, 3).length).toBe(0);
    expect(
      buildChannelLineData(tl, { slope: 1, intercept: 0, start: 4, end: 2 }, 3).length,
    ).toBe(0);
  });

  it('builds a midline averaging top + bot, only when both exist', () => {
    const tl = makeTimeline(5);
    const top = { slope: 1, intercept: 100, start: 0, end: 4 };
    const bot = { slope: -1, intercept: 100, start: 0, end: 4 };
    const data = buildChannelMidlineData(tl, top, bot, 3);
    expect(data.length).toBe(2);
    // y_mid = ((1*x + 100) + (-1*x + 100)) / 2 = 100 at every x
    expect(data[0].value).toBe(100);
    expect(data[1].value).toBe(100);

    expect(buildChannelMidlineData(tl, top, null, 3).length).toBe(0);
    expect(buildChannelMidlineData(tl, null, bot, 3).length).toBe(0);
  });

  it('mounts top + bot + mid line series and pushes data on update', () => {
    const { ctx, chart } = makeLayerCtx();
    const handle = channelsLayer.mount(ctx);
    expect(chart.added.length).toBe(3);

    const tl = makeTimeline(5);
    tl.events[1].structure = {
      always_in: 'neutral',
      confirmed_swings: [],
      micro_channel_top: { slope: 0.5, intercept: 100, start: 0, end: 4 },
      micro_channel_bot: { slope: -0.2, intercept: 102, start: 0, end: 4 },
    };
    handle.update(tl, 4);

    expect(chart.added[0].data.length).toBe(2); // top
    expect(chart.added[1].data.length).toBe(2); // bot
    expect(chart.added[2].data.length).toBe(2); // mid

    handle.unmount();
    expect(chart.removed.length).toBe(3);
  });
});
