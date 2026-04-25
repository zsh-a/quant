/**
 * RegimeTimeline — horizontal band that segments the timeline by regime so
 * users can scan transitions and jump to a regime's first bar.
 *
 * Aligns with the bottom Scrubber: x-axis covers `bars[0..barCount-1]`, each
 * segment's width is proportional to its bar count.
 */

import { useMemo, useState } from 'react';
import { cn } from '../../../../lib/utils';
import { regimeColorOf } from '../../layers/regime';
import {
  useEffectiveBarIdx,
  useStudioActions,
  useTimelineState,
} from '../../store';
import type { SessionTimeline } from '../../types';

export interface RegimeSegment {
  name: string;
  start: number; // first bar idx (inclusive)
  end: number; // last bar idx (inclusive)
  startTimeNs: number;
  endTimeNs: number;
}

/**
 * Group consecutive bars with the same regime name into segments. Bars without
 * a regime fall into a synthetic `"unknown"` segment so the band is contiguous.
 */
export function buildRegimeSegments(timeline: SessionTimeline): RegimeSegment[] {
  const eventByBar = new Map<number, string | null | undefined>();
  for (const ev of timeline.events) {
    eventByBar.set(ev.bar_idx, ev.regime?.name);
  }
  const out: RegimeSegment[] = [];
  let cur: RegimeSegment | null = null;
  for (let i = 0; i < timeline.bars.length; i++) {
    const name = eventByBar.get(i) ?? 'unknown';
    const bar = timeline.bars[i];
    if (cur && cur.name === name) {
      cur.end = i;
      cur.endTimeNs = bar.timestamp_ns;
    } else {
      if (cur) out.push(cur);
      cur = {
        name,
        start: i,
        end: i,
        startTimeNs: bar.timestamp_ns,
        endTimeNs: bar.timestamp_ns,
      };
    }
  }
  if (cur) out.push(cur);
  return out;
}

function formatNs(ns: number): string {
  try {
    const d = new Date(ns / 1_000_000);
    return d.toISOString().replace('T', ' ').replace(/\.\d+Z$/, 'Z');
  } catch {
    return '?';
  }
}

export function RegimeTimeline() {
  const timeline = useTimelineState();
  const barCount = timeline?.bars.length ?? 0;
  const cursorBar = useEffectiveBarIdx();
  const { setBar } = useStudioActions();
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const segments = useMemo(
    () => (timeline ? buildRegimeSegments(timeline) : []),
    [timeline],
  );

  if (!timeline || barCount === 0) {
    return (
      <div className="px-3 py-2 text-xs text-muted-foreground" data-testid="regime-timeline-empty">
        No timeline.
      </div>
    );
  }

  const cursorPct = barCount > 1 ? (cursorBar / (barCount - 1)) * 100 : 0;
  const hovered = hoveredIdx !== null ? segments[hoveredIdx] : null;

  return (
    <div className="flex flex-col gap-2 px-2 py-2" data-testid="regime-timeline">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">
        Regime timeline
      </div>
      <div
        className="relative h-10 w-full overflow-hidden rounded border border-border/60 bg-muted/30"
        role="list"
      >
        <div className="absolute inset-0 flex">
          {segments.map((seg, i) => {
            const span = Math.max(seg.end - seg.start + 1, 1);
            const flexBasis = (span / barCount) * 100;
            const isHovered = hoveredIdx === i;
            return (
              <button
                key={`${seg.start}-${seg.name}`}
                type="button"
                role="listitem"
                style={{ flexBasis: `${flexBasis}%`, backgroundColor: regimeColorOf(seg.name) }}
                onClick={() => setBar(seg.start)}
                onMouseEnter={() => setHoveredIdx(i)}
                onMouseLeave={() => setHoveredIdx((cur) => (cur === i ? null : cur))}
                title={`${seg.name} · bars ${seg.start}-${seg.end}`}
                className={cn(
                  'group relative flex items-center justify-center overflow-hidden border-r border-border/30 px-1 text-[10px] font-medium uppercase text-foreground/80 last:border-r-0 transition-shadow',
                  isHovered && 'ring-1 ring-primary/60',
                )}
                data-testid={`regime-segment-${i}`}
                data-regime-name={seg.name}
              >
                <span className="truncate">{flexBasis > 6 ? seg.name : ''}</span>
              </button>
            );
          })}
        </div>
        <div
          aria-hidden
          className="pointer-events-none absolute inset-y-0 w-px bg-primary/80 shadow-[0_0_4px_rgba(99,102,241,0.7)]"
          style={{ left: `${cursorPct}%` }}
          data-testid="regime-cursor"
        />
      </div>
      <SegmentTooltip segment={hovered} />
    </div>
  );
}

function SegmentTooltip({ segment }: { segment: RegimeSegment | null }) {
  if (!segment) {
    return (
      <div className="text-[11px] text-muted-foreground" data-testid="regime-tooltip-empty">
        Hover a segment for details.
      </div>
    );
  }
  const span = segment.end - segment.start + 1;
  return (
    <div
      className="rounded-md border border-border/60 bg-card/60 px-2 py-1.5 text-[11px]"
      data-testid="regime-tooltip"
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-medium text-foreground">{segment.name}</span>
        <span className="tabular-nums text-muted-foreground">
          bars {segment.start}-{segment.end} ({span})
        </span>
      </div>
      <div className="mt-0.5 text-[10.5px] tabular-nums text-muted-foreground">
        {formatNs(segment.startTimeNs)} → {formatNs(segment.endTimeNs)}
      </div>
    </div>
  );
}
