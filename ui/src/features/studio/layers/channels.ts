/**
 * Channels layer — micro_channel_top / micro_channel_bot fits drawn as a
 * cohesive "channel skeleton": solid bold boundary lines plus a thin dashed
 * midline. The visual goal is to make the channel feel like one structure
 * (the way Al Brooks draws them on his charts) rather than two faint dashed
 * lines that compete with everything else for the eye.
 *
 * Reads the latest event at-or-before `currentBarIdx` (the same convention
 * the swings layer uses, since structure is accumulated). For each channel
 * fit (line of the form `y = slope * x + intercept`, valid on the bar range
 * `[start, end]`) we sample two endpoints and render with a dedicated
 * `LineSeries`. Sampling outside the fit range is intentionally avoided so
 * the line doesn't extrapolate past the data the model fit on.
 *
 * Lightweight-charts does not support polygon fills between two arbitrary
 * sloped lines, so the "channel zone" is communicated by line weight,
 * solidness, and the midline rather than a translucent fill.
 */

import {
  LineSeries,
  LineStyle,
  type ISeriesApi,
  type LineData,
  type UTCTimestamp,
} from 'lightweight-charts';
import type { ChartLayer, LayerCtx, LayerHandle } from './types';
import type { ChannelLine, SessionTimeline, StructureView } from '../types';

const TOP_COLOR = 'rgba(239, 83, 80, 0.85)';
const BOT_COLOR = 'rgba(38, 166, 154, 0.85)';
const MID_COLOR = 'rgba(180, 180, 180, 0.55)';

function timeForBar(timeline: SessionTimeline, idx: number): UTCTimestamp | null {
  const bar = timeline.bars[idx];
  if (!bar) return null;
  return Math.floor(bar.timestamp_ns / 1_000_000_000) as UTCTimestamp;
}

export function latestStructure(
  timeline: SessionTimeline,
  currentBarIdx: number,
): StructureView | null {
  for (let i = Math.min(currentBarIdx, timeline.events.length - 1); i >= 0; i--) {
    const ev = timeline.events[i];
    if (ev?.structure) return ev.structure;
  }
  return null;
}

export function buildChannelLineData(
  timeline: SessionTimeline,
  channel: ChannelLine | null | undefined,
  currentBarIdx: number,
): LineData<UTCTimestamp>[] {
  if (!channel) return [];
  const start = Math.max(0, Math.floor(channel.start));
  const end = Math.min(currentBarIdx, Math.floor(channel.end));
  if (end <= start) return [];

  const startT = timeForBar(timeline, start);
  const endT = timeForBar(timeline, end);
  if (startT === null || endT === null) return [];

  // y = slope * x + intercept; x is bar index.
  return [
    { time: startT, value: channel.slope * start + channel.intercept },
    { time: endT, value: channel.slope * end + channel.intercept },
  ];
}

/**
 * Midline = average(top, bot) at the same endpoints. Returns empty if either
 * boundary is missing — a one-sided channel has no center to draw.
 */
export function buildChannelMidlineData(
  timeline: SessionTimeline,
  top: ChannelLine | null | undefined,
  bot: ChannelLine | null | undefined,
  currentBarIdx: number,
): LineData<UTCTimestamp>[] {
  if (!top || !bot) return [];
  const start = Math.max(0, Math.floor(Math.max(top.start, bot.start)));
  const end = Math.min(currentBarIdx, Math.floor(Math.min(top.end, bot.end)));
  if (end <= start) return [];
  const startT = timeForBar(timeline, start);
  const endT = timeForBar(timeline, end);
  if (startT === null || endT === null) return [];
  const avg = (ch: ChannelLine, x: number) => ch.slope * x + ch.intercept;
  return [
    { time: startT, value: (avg(top, start) + avg(bot, start)) / 2 },
    { time: endT, value: (avg(top, end) + avg(bot, end)) / 2 },
  ];
}

export const channelsLayer: ChartLayer = {
  id: 'channels',
  name: 'Micro channels',
  swatch: TOP_COLOR,
  defaultVisible: true,

  mount(ctx: LayerCtx): LayerHandle {
    const { chart } = ctx;
    let topSeries: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      color: TOP_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    let botSeries: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      color: BOT_COLOR,
      lineWidth: 2,
      lineStyle: LineStyle.Solid,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });
    let midSeries: ISeriesApi<'Line'> | null = chart.addSeries(LineSeries, {
      color: MID_COLOR,
      lineWidth: 1,
      lineStyle: LineStyle.Dashed,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    });

    return {
      update(timeline, currentBarIdx) {
        if (!topSeries || !botSeries || !midSeries) return;
        const struct = latestStructure(timeline, currentBarIdx);
        topSeries.setData(buildChannelLineData(timeline, struct?.micro_channel_top, currentBarIdx));
        botSeries.setData(buildChannelLineData(timeline, struct?.micro_channel_bot, currentBarIdx));
        midSeries.setData(
          buildChannelMidlineData(
            timeline,
            struct?.micro_channel_top,
            struct?.micro_channel_bot,
            currentBarIdx,
          ),
        );
      },
      unmount() {
        try {
          if (topSeries) chart.removeSeries(topSeries);
          if (botSeries) chart.removeSeries(botSeries);
          if (midSeries) chart.removeSeries(midSeries);
        } catch {
          // chart already disposed
        }
        topSeries = null;
        botSeries = null;
        midSeries = null;
      },
    };
  },
};
