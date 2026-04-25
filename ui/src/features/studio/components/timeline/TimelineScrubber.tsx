/**
 * Bar-level scrubber. Drag throttled to 50 ms via requestAnimationFrame so
 * the chart redraw stays under one frame even for 10K-bar timelines.
 */

import { useCallback, useEffect, useRef } from 'react';
import { Slider } from '../../../../components/ui/slider';
import {
  LIVE_TAIL,
  useCurrentBarIdx,
  useStudioActions,
  useTimelineState,
} from '../../store';

const THROTTLE_MS = 50;

export function TimelineScrubber() {
  const timeline = useTimelineState();
  const currentBarIdx = useCurrentBarIdx();
  const { setBar } = useStudioActions();

  const lastEmitRef = useRef(0);
  const pendingRef = useRef<number | null>(null);
  const rafRef = useRef<number | null>(null);

  const flush = useCallback(() => {
    rafRef.current = null;
    if (pendingRef.current === null) return;
    setBar(pendingRef.current);
    pendingRef.current = null;
    lastEmitRef.current = performance.now();
  }, [setBar]);

  const throttledSet = useCallback(
    (idx: number) => {
      const now = performance.now();
      if (now - lastEmitRef.current >= THROTTLE_MS) {
        lastEmitRef.current = now;
        setBar(idx);
        pendingRef.current = null;
        return;
      }
      pendingRef.current = idx;
      if (rafRef.current === null) {
        rafRef.current = window.setTimeout(flush, THROTTLE_MS - (now - lastEmitRef.current));
      }
    },
    [flush, setBar],
  );

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) clearTimeout(rafRef.current);
    };
  }, []);

  const barCount = timeline?.bars.length ?? 0;
  const max = Math.max(0, barCount - 1);
  const value = currentBarIdx === LIVE_TAIL ? max : Math.min(currentBarIdx, max);

  if (barCount === 0) {
    return (
      <div className="flex h-12 items-center px-3 text-xs text-muted-foreground">
        No bars loaded
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1.5 px-3 py-2">
      <Slider
        value={[value]}
        min={0}
        max={max}
        step={1}
        onValueChange={(vals) => {
          const v = vals[0];
          if (typeof v === 'number') throttledSet(v);
        }}
        aria-label="Timeline scrubber"
        data-testid="timeline-scrubber"
      />
      {/* S3: regime band thumbnail will mount into this 5px-tall slot. */}
      <div className="h-[5px] rounded bg-muted/50" aria-hidden />
      <div className="flex items-center justify-between text-[11px] text-muted-foreground tabular-nums">
        <span>bar {value} / {max}</span>
        {currentBarIdx === LIVE_TAIL && <span className="text-emerald-400">● LIVE</span>}
      </div>
    </div>
  );
}
