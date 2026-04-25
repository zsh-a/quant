import { ChevronFirst, ChevronLast, Pause, Play, Radio, SkipBack, SkipForward } from 'lucide-react';
import { Button } from '../../../../components/ui/button';
import {
  useStudioActions,
  useStudioPlayState,
  useStudioSpeed,
} from '../../store';
import { STUDIO_SPEEDS, type StudioSpeed } from '../../types';

export function PlaybackControls() {
  const playState = useStudioPlayState();
  const speed = useStudioSpeed();
  const { stepBar, togglePlay, setSpeed, setBar, jumpToLive } = useStudioActions();

  return (
    <div className="flex items-center gap-1 px-3 py-2">
      <Button
        variant="ghost"
        size="icon"
        onClick={() => setBar(0)}
        title="Jump to start (Home)"
        aria-label="Jump to start"
      >
        <ChevronFirst className="size-4" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => stepBar(-1)}
        title="Step back (←)"
        aria-label="Step back"
      >
        <SkipBack className="size-4" />
      </Button>
      <Button
        variant="default"
        size="icon"
        onClick={togglePlay}
        title="Play / Pause (Space)"
        aria-label={playState === 'playing' ? 'Pause' : 'Play'}
      >
        {playState === 'playing' ? <Pause className="size-4" /> : <Play className="size-4" />}
      </Button>
      <Button
        variant="ghost"
        size="icon"
        onClick={() => stepBar(1)}
        title="Step forward (→)"
        aria-label="Step forward"
      >
        <SkipForward className="size-4" />
      </Button>
      <Button
        variant="ghost"
        size="icon"
        onClick={jumpToLive}
        title="Jump to live tail (End)"
        aria-label="Jump to live"
      >
        <ChevronLast className="size-4" />
      </Button>

      <div className="mx-2 h-5 w-px bg-border" aria-hidden />

      <div
        className="inline-flex items-center gap-1 rounded-md border border-border/70 bg-secondary/40 p-0.5 text-xs"
        role="group"
        aria-label="Speed"
      >
        {STUDIO_SPEEDS.map((s) => (
          <button
            key={s}
            type="button"
            onClick={() => setSpeed(s as StudioSpeed)}
            className={`tabular-nums px-2 py-0.5 rounded ${
              speed === s
                ? 'bg-primary text-primary-foreground'
                : 'text-muted-foreground hover:text-foreground'
            }`}
            aria-pressed={speed === s}
            data-testid={`speed-${s}`}
          >
            {s}×
          </button>
        ))}
      </div>

      <div className="ml-auto flex items-center gap-1.5 text-[11px] text-muted-foreground">
        <Radio className="size-3" />
        <span>{playState === 'playing' ? 'playing' : 'paused'}</span>
      </div>
    </div>
  );
}
