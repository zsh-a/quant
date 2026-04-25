/**
 * MultiAnalystCompare — side panel that fans the same bar out to N analysts
 * and shows their results side-by-side.
 *
 * The user picks a set of analysts (rule / llm:claude / vlm:gemini /
 * ensemble.critic …) and presses *Run*; we POST `/replay-bar` with the
 * analyst list and current bar index, then render a table where each column
 * is one analyst and rows show the comparable fields (side, primary
 * pattern, probability, expected R, signal entry/stop, decision target).
 *
 * Caching: results are keyed by `${sessionId}|${barIdx}|${analyst}` so that
 * re-running the same combination is an O(1) lookup. Concurrency: all
 * analysts in one request fan out on the backend (`asyncio.gather`), so
 * walltime is bounded by the slowest single analyst.
 */

import { useEffect, useMemo, useRef, useState } from 'react';
import { useParams } from 'react-router-dom';
import { Loader2, Play } from 'lucide-react';
import { Button } from '../../../../components/ui/button';
import { replayBar, type ReplayBarAnalystResult } from '../../api';
import { useEffectiveBarIdx } from '../../store';
import type { Decision, Signal } from '../../types';

const DEFAULT_ANALYSTS = [
  'rule',
  'llm',
  'vlm',
  'ensemble.critic',
] as const;

type CacheKey = string;

function cacheKey(sessionId: string, barIdx: number, analyst: string): CacheKey {
  return `${sessionId}|${barIdx}|${analyst}`;
}

/** Bucket-style probability (0…1 → "low/med/high") for visual diff. */
export function probabilityBucket(p: number | null | undefined): string {
  if (p == null || !Number.isFinite(p)) return '—';
  if (p < 0.4) return 'low';
  if (p < 0.6) return 'med';
  return 'high';
}

function topSignal(signals: Signal[]): Signal | null {
  if (signals.length === 0) return null;
  return signals.reduce((best, cur) => {
    const a = (cur.probability as number | undefined) ?? 0;
    const b = (best.probability as number | undefined) ?? 0;
    return a > b ? cur : best;
  });
}

interface Row {
  label: string;
  pick: (r: ReplayBarAnalystResult) => string;
}

const ROWS: Row[] = [
  {
    label: '#signals',
    pick: (r) => String(r.signals.length),
  },
  {
    label: 'top side',
    pick: (r) => topSignal(r.signals)?.side ?? '—',
  },
  {
    label: 'top pattern',
    pick: (r) => topSignal(r.signals)?.pattern ?? '—',
  },
  {
    label: 'top prob',
    pick: (r) => {
      const t = topSignal(r.signals);
      const p = (t?.probability as number | undefined) ?? null;
      return p == null ? '—' : `${(p * 100).toFixed(0)}% (${probabilityBucket(p)})`;
    },
  },
  {
    label: 'top entry',
    pick: (r) => {
      const t = topSignal(r.signals);
      const v = (t?.entry_px as number | undefined) ?? (t?.['entry'] as number | undefined);
      return typeof v === 'number' ? v.toFixed(4) : '—';
    },
  },
  {
    label: 'top stop',
    pick: (r) => {
      const t = topSignal(r.signals);
      const v = (t?.stop_px as number | undefined) ?? (t?.['stop'] as number | undefined);
      return typeof v === 'number' ? v.toFixed(4) : '—';
    },
  },
  {
    label: 'decision',
    pick: (r) => decisionLabel(r.decision),
  },
  {
    label: 'expected R',
    pick: (r) => (r.decision ? r.decision.expected_r.toFixed(2) : '—'),
  },
  {
    label: 'error',
    pick: (r) => r.error ?? '—',
  },
];

function decisionLabel(d: Decision | null): string {
  if (!d) return '—';
  const tgt = d.target_px != null ? ` t=${d.target_px.toFixed(2)}` : '';
  return `${d.side} e=${d.entry_px.toFixed(2)} s=${d.stop_px.toFixed(2)}${tgt}`;
}

/** Diff highlight: cell value differs from the *first* (reference) result. */
function diffClass(values: string[], idx: number): string {
  if (idx === 0) return '';
  return values[idx] !== values[0] ? 'bg-amber-500/15 text-amber-200' : '';
}

export function MultiAnalystCompare() {
  const { sessionId } = useParams<{ sessionId: string }>();
  const currentBarIdx = useEffectiveBarIdx();
  const [selected, setSelected] = useState<string[]>([...DEFAULT_ANALYSTS]);
  const [draft, setDraft] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<ReplayBarAnalystResult[]>([]);
  const cacheRef = useRef<Map<CacheKey, ReplayBarAnalystResult>>(new Map());
  const abortRef = useRef<AbortController | null>(null);

  // Drop pending request on unmount.
  useEffect(() => () => abortRef.current?.abort(), []);

  const cachedView = useMemo(() => {
    if (!sessionId) return null;
    const out: ReplayBarAnalystResult[] = [];
    for (const a of selected) {
      const cached = cacheRef.current.get(cacheKey(sessionId, currentBarIdx, a));
      if (cached) out.push({ ...cached, cached: true });
    }
    return out.length === selected.length && selected.length > 0 ? out : null;
  }, [sessionId, currentBarIdx, selected]);

  useEffect(() => {
    if (cachedView) setResults(cachedView);
  }, [cachedView]);

  const onRun = async () => {
    if (!sessionId || selected.length === 0 || currentBarIdx < 0) return;
    abortRef.current?.abort();
    const ctl = new AbortController();
    abortRef.current = ctl;
    setLoading(true);
    setError(null);
    try {
      // Skip analysts already cached for this bar.
      const fresh = selected.filter(
        (a) => !cacheRef.current.has(cacheKey(sessionId, currentBarIdx, a)),
      );
      let next: ReplayBarAnalystResult[];
      if (fresh.length === 0) {
        next = selected.map((a) => ({
          ...(cacheRef.current.get(cacheKey(sessionId, currentBarIdx, a)) as ReplayBarAnalystResult),
          cached: true,
        }));
      } else {
        const resp = await replayBar(sessionId, currentBarIdx, fresh, ctl.signal);
        for (const r of resp.results) {
          cacheRef.current.set(cacheKey(sessionId, currentBarIdx, r.analyst), r);
        }
        next = selected.map((a) => {
          const r = cacheRef.current.get(cacheKey(sessionId, currentBarIdx, a));
          return r ? { ...r, cached: !fresh.includes(a) } : (
            { analyst: a, bar_idx: currentBarIdx, signals: [], decision: null, error: 'missing', cached: false }
          );
        });
      }
      setResults(next);
    } catch (e) {
      if ((e as Error).name === 'AbortError') return;
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  };

  const removeAnalyst = (a: string) => {
    setSelected((cur) => cur.filter((x) => x !== a));
  };

  const addAnalyst = () => {
    const trimmed = draft.trim();
    if (!trimmed || selected.includes(trimmed)) return;
    setSelected((cur) => [...cur, trimmed]);
    setDraft('');
  };

  // Render rows: each cell value gathered first so we can diff.
  const cellValues: string[][] = ROWS.map((row) => results.map(row.pick));

  return (
    <div className="flex flex-col gap-3 p-3 text-xs" data-testid="multi-analyst-compare">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-wide text-muted-foreground">
          Compare on bar {currentBarIdx >= 0 ? currentBarIdx : '—'}
        </span>
        <div className="ml-auto flex items-center gap-1">
          <input
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                addAnalyst();
              }
            }}
            placeholder="add analyst…"
            className="rounded border border-border/70 bg-secondary/30 px-2 py-1 text-xs outline-none focus:border-primary"
            data-testid="multi-analyst-add-input"
          />
          <Button size="sm" variant="secondary" onClick={addAnalyst} data-testid="multi-analyst-add">
            +
          </Button>
        </div>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {selected.map((a) => (
          <button
            key={a}
            type="button"
            onClick={() => removeAnalyst(a)}
            className="rounded-full bg-secondary/50 px-2 py-0.5 text-[11px] text-foreground hover:bg-destructive/40"
            title={`Remove ${a}`}
            data-testid={`multi-analyst-chip-${a}`}
          >
            {a} ×
          </button>
        ))}
      </div>

      <div className="flex items-center gap-2">
        <Button
          size="sm"
          onClick={onRun}
          disabled={loading || selected.length === 0 || currentBarIdx < 0}
          data-testid="multi-analyst-run"
        >
          {loading ? <Loader2 className="size-3.5 animate-spin" /> : <Play className="size-3.5" />}
          <span className="ml-1.5">{loading ? 'Running…' : 'Run'}</span>
        </Button>
        {error && (
          <span className="text-[11px] text-destructive" data-testid="multi-analyst-error">
            {error}
          </span>
        )}
      </div>

      {results.length > 0 && (
        <div className="overflow-x-auto rounded border border-border/60">
          <table className="min-w-full text-xs">
            <thead>
              <tr className="bg-secondary/40">
                <th className="px-2 py-1 text-left font-medium text-muted-foreground">field</th>
                {results.map((r) => (
                  <th
                    key={r.analyst}
                    className="px-2 py-1 text-left font-medium tabular-nums"
                    data-testid={`multi-analyst-col-${r.analyst}`}
                  >
                    {r.analyst}
                    {r.cached && <span className="ml-1 text-[10px] text-muted-foreground">(cached)</span>}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {ROWS.map((row, ri) => (
                <tr key={row.label} className="border-t border-border/50">
                  <td className="px-2 py-1 text-muted-foreground">{row.label}</td>
                  {results.map((_, ci) => {
                    const value = cellValues[ri][ci];
                    return (
                      <td
                        key={ci}
                        className={`px-2 py-1 tabular-nums ${diffClass(cellValues[ri], ci)}`}
                        data-testid={`multi-analyst-cell-${row.label}-${ci}`}
                      >
                        {value}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
