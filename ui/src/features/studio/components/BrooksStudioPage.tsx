/**
 * BrooksStudioPage — top-level route for `/studio/:sessionId`.
 *
 * S2 ships the skeleton: chart + scrubber + playback + URL/keyboard
 * bindings. S3 fills in the side panels & layer toggle into the placeholders
 * outlined below.
 */

import { useParams } from 'react-router-dom';
import { useTimeline } from '../hooks/useTimeline';
import { useReplay } from '../hooks/useReplay';
import { useUrlState } from '../hooks/useUrlState';
import {
  useStudioError,
  useStudioLoading,
  useTimelineState,
} from '../store';
import { ChartCanvas } from './chart/ChartCanvas';
import { TimelineScrubber } from './timeline/TimelineScrubber';
import { PlaybackControls } from './timeline/PlaybackControls';
import { LayerToggle } from './timeline/LayerToggle';

export default function BrooksStudioPage() {
  const { sessionId } = useParams<{ sessionId: string }>();

  useTimeline(sessionId ?? null);
  useUrlState();
  useReplay();

  const timeline = useTimelineState();
  const loading = useStudioLoading();
  const error = useStudioError();

  if (!sessionId) {
    return (
      <div className="flex h-[calc(100vh-120px)] items-center justify-center text-sm text-muted-foreground">
        Missing session id. Open <code className="rounded bg-muted px-1.5">/studio/&lt;session_id&gt;</code>.
      </div>
    );
  }

  return (
    <div className="flex h-[calc(100vh-120px)] flex-col gap-3 px-4 pb-4">
      <header className="flex items-center justify-between gap-3 text-sm">
        <div className="flex items-center gap-3">
          <span className="font-medium">Brooks Studio</span>
          {timeline && (
            <span className="text-muted-foreground tabular-nums">
              {timeline.symbol} · {timeline.base_interval} · {timeline.bars.length} bars
            </span>
          )}
        </div>
        <code className="text-[11px] text-muted-foreground">{sessionId}</code>
      </header>

      <div className="relative min-h-0 flex-1 overflow-hidden rounded-xl border border-border/70 bg-[#0e1116]">
        <ChartCanvas />
        <div className="pointer-events-auto absolute right-3 top-3 z-10">
          <LayerToggle />
        </div>
        {loading && !timeline && (
          <div className="absolute inset-0 flex items-center justify-center bg-black/30 text-xs text-muted-foreground">
            Loading timeline…
          </div>
        )}
        {error && (
          <div className="absolute left-3 top-3 max-w-md rounded-md border border-destructive/60 bg-destructive/15 px-3 py-1.5 text-xs text-destructive-foreground">
            {error}
          </div>
        )}
      </div>

      <div className="rounded-xl border border-border/70 bg-card">
        <TimelineScrubber />
        <div className="border-t border-border/60">
          <PlaybackControls />
        </div>
      </div>
    </div>
  );
}
