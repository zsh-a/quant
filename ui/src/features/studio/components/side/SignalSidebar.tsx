/**
 * SignalSidebar — list of detector signals up to (and including) the
 * currentBarIdx. Click a row to jump the chart to that signal's bar and pin
 * it as the selected layer; chart-driven hover updates the inspected
 * `currentBarIdx` cutoff so users see "what was visible when X happened".
 *
 * Virtualization kicks in past `VIRTUALIZE_THRESHOLD` rows (react-window).
 */

import { memo, useMemo, useState } from 'react';
import { FixedSizeList, type ListChildComponentProps } from 'react-window';
import AutoSizer from 'react-virtualized-auto-sizer';
import { ArrowDown, ArrowUp, Filter } from 'lucide-react';
import { cn } from '../../../../lib/utils';
import {
  useInspectedBarIdx,
  useSelectedSignalId,
  useStudioActions,
  useTimelineState,
} from '../../store';
import type { BarEvent, SessionTimeline, Signal } from '../../types';

export const SOURCE_KINDS = ['rule', 'llm', 'vlm'] as const;
export type SourceKind = (typeof SOURCE_KINDS)[number];
export type SourceFilter = SourceKind | 'all';
export type SideFilter = 'long' | 'short' | 'all';

// Phase S6: lowered from 100 → 50 so mid-sized lists also benefit from
// react-window. Below 50 rows the FixedSizeList overhead isn't worth it
// (DOM is small enough to layout in a few ms).
const VIRTUALIZE_THRESHOLD = 50;
const ROW_HEIGHT = 56;

export interface SignalRow {
  id: string;
  bar_idx: number;
  signal: Signal;
}

/** Read the high-level bucket from a signal `source` like "rule:h2" → "rule". */
export function sourceKind(source: unknown): SourceKind | null {
  if (typeof source !== 'string') return null;
  const head = source.split(':')[0]?.toLowerCase();
  if (head === 'rule' || head === 'llm' || head === 'vlm') return head;
  return null;
}

function readNumber(obj: Record<string, unknown>, key: string): number | null {
  const v = obj[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function signalProbability(sig: Signal): number | null {
  return readNumber(sig as Record<string, unknown>, 'probability');
}

function signalBarIdx(sig: Signal, fallback: number): number {
  const sb = readNumber(sig as Record<string, unknown>, 'signal_bar_idx');
  if (sb !== null) return sb;
  if (typeof sig.bar_idx === 'number') return sig.bar_idx;
  return fallback;
}

export interface SignalFilter {
  source: SourceFilter;
  side: SideFilter;
  minProbability: number;
}

export const DEFAULT_FILTER: SignalFilter = {
  source: 'all',
  side: 'all',
  minProbability: 0,
};

export function collectSidebarSignals(
  timeline: SessionTimeline,
  cutoffBarIdx: number,
  filter: SignalFilter,
): SignalRow[] {
  const out: SignalRow[] = [];
  const events: BarEvent[] = timeline.events;
  for (const ev of events) {
    if (ev.bar_idx > cutoffBarIdx) continue;
    if (!ev.signals?.length) continue;
    for (let j = 0; j < ev.signals.length; j++) {
      const sig = ev.signals[j];
      if (filter.side !== 'all' && sig.side !== filter.side) continue;
      if (filter.source !== 'all') {
        if (sourceKind(sig.source) !== filter.source) continue;
      }
      const prob = signalProbability(sig);
      if (filter.minProbability > 0 && (prob === null || prob < filter.minProbability)) continue;
      const id = sig.id ?? `sig-${ev.bar_idx}-${j}`;
      out.push({ id, bar_idx: signalBarIdx(sig, ev.bar_idx), signal: sig });
    }
  }
  // Most-recent first.
  out.sort((a, b) => b.bar_idx - a.bar_idx);
  return out;
}

interface SignalRowViewProps {
  row: SignalRow;
  selected: boolean;
  onClick: () => void;
}

const SignalRowView = memo(function SignalRowView({
  row,
  selected,
  onClick,
}: SignalRowViewProps) {
  const { signal, bar_idx } = row;
  const side = signal.side;
  const prob = signalProbability(signal);
  const source = typeof signal.source === 'string' ? signal.source : '—';
  const pattern = signal.pattern ?? '?';

  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'flex w-full items-center gap-2 rounded-md border border-transparent px-2 py-1.5 text-left text-xs transition-colors',
        'hover:border-border/60 hover:bg-accent/40',
        selected && 'border-primary/60 bg-primary/10',
      )}
      data-testid={`signal-row-${row.id}`}
      data-selected={selected ? 'true' : 'false'}
    >
      <span
        aria-hidden
        className={cn(
          'mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-sm',
          side === 'long' && 'bg-emerald-500/20 text-emerald-400',
          side === 'short' && 'bg-rose-500/20 text-rose-400',
          !side && 'bg-muted text-muted-foreground',
        )}
      >
        {side === 'short' ? <ArrowDown className="size-3" /> : <ArrowUp className="size-3" />}
      </span>
      <span className="flex min-w-0 flex-1 flex-col">
        <span className="flex items-center justify-between gap-1.5">
          <span className="truncate font-medium text-foreground">{pattern}</span>
          <span className="shrink-0 tabular-nums text-muted-foreground">#{bar_idx}</span>
        </span>
        <span className="flex items-center justify-between gap-1.5 text-[10.5px] text-muted-foreground">
          <span className="truncate">{source}</span>
          <span className="shrink-0 tabular-nums">
            {prob !== null ? `${(prob * 100).toFixed(0)}%` : '—'}
          </span>
        </span>
      </span>
    </button>
  );
});

export function SignalSidebar() {
  const timeline = useTimelineState();
  const { barIdx, preview } = useInspectedBarIdx();
  const selectedSignalId = useSelectedSignalId();
  const { setBar, setSelectedSignalId } = useStudioActions();

  const [filter, setFilter] = useState<SignalFilter>(DEFAULT_FILTER);

  const rows = useMemo<SignalRow[]>(() => {
    if (!timeline) return [];
    return collectSidebarSignals(timeline, barIdx, filter);
  }, [timeline, barIdx, filter]);

  if (!timeline) {
    return (
      <div className="px-2 py-3 text-xs text-muted-foreground">No timeline loaded.</div>
    );
  }

  const handleClickRow = (row: SignalRow) => {
    setSelectedSignalId(row.id);
    setBar(row.bar_idx);
  };

  return (
    <div className="flex h-full min-h-0 flex-col gap-2" data-testid="signal-sidebar">
      <FilterBar filter={filter} onChange={setFilter} />
      <div className="px-2 text-[11px] text-muted-foreground">
        {rows.length} signal{rows.length === 1 ? '' : 's'} ≤ bar {barIdx}
        {preview && (
          <span className="ml-2 rounded-sm bg-amber-500/15 px-1.5 py-0.5 text-amber-400">
            preview
          </span>
        )}
      </div>
      <div className="min-h-0 flex-1 px-1.5 pb-1.5">
        {rows.length === 0 ? (
          <div className="px-1 py-3 text-xs text-muted-foreground">No signals match.</div>
        ) : rows.length > VIRTUALIZE_THRESHOLD ? (
          <AutoSizer>
            {({ height, width }: { height: number; width: number }) => (
              <FixedSizeList
                height={height}
                width={width}
                itemCount={rows.length}
                itemSize={ROW_HEIGHT}
                overscanCount={4}
                itemData={{ rows, selectedSignalId, handleClickRow }}
                data-testid="signal-list-virtual"
              >
                {VirtualRow}
              </FixedSizeList>
            )}
          </AutoSizer>
        ) : (
          <div className="flex flex-col gap-1 overflow-y-auto pr-1" data-testid="signal-list-plain">
            {rows.map((row) => (
              <SignalRowView
                key={row.id}
                row={row}
                selected={row.id === selectedSignalId}
                onClick={() => handleClickRow(row)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

interface VirtualItemData {
  rows: SignalRow[];
  selectedSignalId: string | null;
  handleClickRow: (row: SignalRow) => void;
}

function VirtualRow({ index, style, data }: ListChildComponentProps<VirtualItemData>) {
  const row = data.rows[index];
  return (
    <div style={style} className="px-0.5">
      <SignalRowView
        row={row}
        selected={row.id === data.selectedSignalId}
        onClick={() => data.handleClickRow(row)}
      />
    </div>
  );
}

const PROBABILITY_OPTIONS = [0, 0.5, 0.7, 0.9] as const;

function FilterBar({
  filter,
  onChange,
}: {
  filter: SignalFilter;
  onChange: (next: SignalFilter) => void;
}) {
  return (
    <div
      className="flex flex-wrap items-center gap-1.5 px-2 pt-2 text-[11px]"
      data-testid="signal-filter-bar"
    >
      <Filter className="size-3 text-muted-foreground" aria-hidden />

      <Pill
        active={filter.source === 'all'}
        onClick={() => onChange({ ...filter, source: 'all' })}
        label="all"
      />
      {SOURCE_KINDS.map((s) => (
        <Pill
          key={s}
          active={filter.source === s}
          onClick={() => onChange({ ...filter, source: s })}
          label={s}
          testId={`filter-source-${s}`}
        />
      ))}

      <span className="mx-1 h-3 w-px bg-border" aria-hidden />

      <Pill
        active={filter.side === 'all'}
        onClick={() => onChange({ ...filter, side: 'all' })}
        label="all"
      />
      <Pill
        active={filter.side === 'long'}
        onClick={() => onChange({ ...filter, side: 'long' })}
        label="long"
        tone="bull"
        testId="filter-side-long"
      />
      <Pill
        active={filter.side === 'short'}
        onClick={() => onChange({ ...filter, side: 'short' })}
        label="short"
        tone="bear"
        testId="filter-side-short"
      />

      <span className="mx-1 h-3 w-px bg-border" aria-hidden />

      <select
        value={filter.minProbability}
        onChange={(e) => onChange({ ...filter, minProbability: Number(e.target.value) })}
        className="rounded border border-border/60 bg-background px-1 py-0.5 text-[11px] text-foreground"
        aria-label="Minimum probability"
        data-testid="filter-min-probability"
      >
        {PROBABILITY_OPTIONS.map((p) => (
          <option key={p} value={p}>
            p ≥ {p === 0 ? 'any' : `${(p * 100).toFixed(0)}%`}
          </option>
        ))}
      </select>
    </div>
  );
}

function Pill({
  label,
  active,
  onClick,
  tone,
  testId,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
  tone?: 'bull' | 'bear';
  testId?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'rounded-full border px-2 py-0.5 transition-colors',
        active
          ? 'border-primary/60 bg-primary/15 text-foreground'
          : 'border-border/60 bg-muted/30 text-muted-foreground hover:text-foreground',
        active && tone === 'bull' && 'border-emerald-500/60 bg-emerald-500/15 text-emerald-400',
        active && tone === 'bear' && 'border-rose-500/60 bg-rose-500/15 text-rose-400',
      )}
      aria-pressed={active}
      data-testid={testId}
    >
      {label}
    </button>
  );
}
