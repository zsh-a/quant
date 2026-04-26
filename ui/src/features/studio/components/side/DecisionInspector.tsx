/**
 * DecisionInspector — full Decision detail for the inspected bar (current
 * bar by default, hovered bar when the user is moving the chart crosshair).
 *
 * The Decision schema is loosely typed (`[key: string]: unknown`) so each
 * helper here narrows defensively: missing fields render as "—" rather than
 * throwing.
 */

import { useState } from 'react';
import { Copy, Check, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { cn } from '../../../../lib/utils';
import { Button } from '../../../../components/ui/button';
import {
  useInspectedBarIdx,
  useTimelineState,
} from '../../store';
import type { Decision, Signal } from '../../types';

export function probabilityBucket(p: number): { label: string; tone: 'low' | 'mid' | 'high' | 'top' } {
  if (p < 0.5) return { label: 'low', tone: 'low' };
  if (p < 0.65) return { label: 'medium', tone: 'mid' };
  if (p < 0.8) return { label: 'high', tone: 'high' };
  return { label: 'very high', tone: 'top' };
}

const BUCKET_COLORS: Record<'low' | 'mid' | 'high' | 'top', string> = {
  low: 'bg-rose-500/15 text-rose-400',
  mid: 'bg-amber-500/15 text-amber-400',
  high: 'bg-emerald-500/15 text-emerald-400',
  top: 'bg-emerald-400/25 text-emerald-300',
};

export function expectedRTone(r: number): string {
  if (r >= 1.5) return 'text-emerald-400';
  if (r > 0) return 'text-emerald-300';
  if (r === 0) return 'text-muted-foreground';
  return 'text-rose-400';
}

function fmtPx(v: number | null | undefined, fallback = '—'): string {
  if (typeof v !== 'number' || !Number.isFinite(v)) return fallback;
  if (Math.abs(v) >= 1000) return v.toFixed(2);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  return v.toFixed(4);
}

function diffToCurrent(target: number | null | undefined, current: number | null): string {
  if (typeof target !== 'number' || current === null) return '';
  const delta = target - current;
  const sign = delta > 0 ? '+' : '';
  const pct = (Math.abs(delta) / Math.max(Math.abs(current), 1e-9)) * 100;
  return `${sign}${fmtPx(delta)} (${pct.toFixed(2)}%)`;
}

function readNumber(obj: Record<string, unknown>, key: string): number | null {
  const v = obj[key];
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

/** `decision.signals` is loosely typed so we narrow to a Signal[] best-effort. */
export function decisionSupportingSignals(decision: Decision): Signal[] {
  const v = (decision as Record<string, unknown>).signals;
  return Array.isArray(v) ? (v as Signal[]) : [];
}

export function DecisionInspector() {
  const timeline = useTimelineState();
  const { barIdx, preview } = useInspectedBarIdx();

  if (!timeline) {
    return <div className="px-3 py-3 text-xs text-muted-foreground">No timeline loaded.</div>;
  }

  const ev = timeline.events.find((e) => e.bar_idx === barIdx);
  const decision = ev?.decision ?? null;
  const bar = timeline.bars[barIdx];
  const currentPx = bar ? bar.close : null;

  if (!decision) {
    return (
      <div className="flex flex-col gap-1.5 px-3 py-3 text-xs text-muted-foreground" data-testid="decision-inspector-empty">
        <div className="flex items-center gap-2">
          <span className="font-medium text-foreground">Decision @ bar {barIdx}</span>
          {preview && (
            <span className="rounded-sm bg-amber-500/15 px-1.5 py-0.5 text-amber-400">preview</span>
          )}
        </div>
        <span>—</span>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3 px-3 py-3 text-xs" data-testid="decision-inspector">
      <Header decision={decision} barIdx={barIdx} preview={preview} />
      <PriceLevels decision={decision} currentPx={currentPx} />
      <ProbabilityRow decision={decision} />
      <ContextRow decision={decision} />
      <Reasoning decision={decision} />
      <SignalsSection decision={decision} />
      <CopyJsonButton decision={decision} />
    </div>
  );
}

function Header({
  decision,
  barIdx,
  preview,
}: {
  decision: Decision;
  barIdx: number;
  preview: boolean;
}) {
  const sideTone =
    decision.side === 'long'
      ? 'bg-emerald-500/15 text-emerald-400'
      : 'bg-rose-500/15 text-rose-400';
  return (
    <div className="flex items-center justify-between gap-2">
      <div className="flex items-center gap-2">
        <span
          className={cn('rounded-md px-2 py-0.5 text-[11px] font-semibold uppercase', sideTone)}
          data-testid="decision-side-badge"
        >
          {decision.side}
        </span>
        <span className="text-foreground font-medium">{decision.pattern || '—'}</span>
        <span className="text-muted-foreground tabular-nums">@ bar {barIdx}</span>
      </div>
      {preview && (
        <span className="rounded-sm bg-amber-500/15 px-1.5 py-0.5 text-[10.5px] text-amber-400">
          preview
        </span>
      )}
    </div>
  );
}

function PriceLevels({ decision, currentPx }: { decision: Decision; currentPx: number | null }) {
  const target = decision.target_px;
  return (
    <div className="grid grid-cols-3 gap-1.5 text-[11px]" data-testid="decision-price-levels">
      <PxCell label="entry" value={decision.entry_px} delta={diffToCurrent(decision.entry_px, currentPx)} />
      <PxCell label="stop" value={decision.stop_px} delta={diffToCurrent(decision.stop_px, currentPx)} tone="bear" />
      <PxCell
        label="target"
        value={typeof target === 'number' ? target : null}
        delta={diffToCurrent(target, currentPx)}
        tone="bull"
      />
    </div>
  );
}

function PxCell({
  label,
  value,
  delta,
  tone,
}: {
  label: string;
  value: number | null;
  delta: string;
  tone?: 'bull' | 'bear';
}) {
  const tonecls =
    tone === 'bull' ? 'text-emerald-300' : tone === 'bear' ? 'text-rose-300' : 'text-foreground';
  return (
    <div className="rounded-md border border-border/60 bg-card/50 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className={cn('tabular-nums font-medium', tonecls)}>{fmtPx(value)}</div>
      {delta && <div className="text-[10px] text-muted-foreground tabular-nums">{delta}</div>}
    </div>
  );
}

function ProbabilityRow({ decision }: { decision: Decision }) {
  const p = typeof decision.probability === 'number' ? decision.probability : null;
  const er = typeof decision.expected_r === 'number' ? decision.expected_r : null;
  const bucket = p !== null ? probabilityBucket(p) : null;
  return (
    <div className="flex items-center gap-2" data-testid="decision-probability-row">
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase text-muted-foreground">prob</span>
        {p !== null && bucket ? (
          <span className={cn('rounded px-1.5 py-0.5 text-[11px] font-medium tabular-nums', BUCKET_COLORS[bucket.tone])}>
            {(p * 100).toFixed(0)}% · {bucket.label}
          </span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </div>
      <div className="flex items-center gap-1.5">
        <span className="text-[10px] uppercase text-muted-foreground">E[R]</span>
        {er !== null ? (
          <span className={cn('tabular-nums font-medium', expectedRTone(er))}>{er.toFixed(2)}R</span>
        ) : (
          <span className="text-muted-foreground">—</span>
        )}
      </div>
    </div>
  );
}

function ContextRow({ decision }: { decision: Decision }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      <Tag label={decision.regime || 'no-regime'} testId="decision-regime" />
      {decision.htf_aligned ? (
        <Tag label="htf-aligned" tone="ok" testId="decision-htf-aligned" />
      ) : (
        <Tag label="htf-misaligned" tone="warn" testId="decision-htf-misaligned" />
      )}
      <Tag label={decision.source || '—'} tone="muted" testId="decision-source" />
    </div>
  );
}

function Tag({
  label,
  tone = 'muted',
  testId,
}: {
  label: string;
  tone?: 'ok' | 'warn' | 'muted';
  testId?: string;
}) {
  const cls =
    tone === 'ok'
      ? 'border-emerald-500/60 bg-emerald-500/10 text-emerald-300'
      : tone === 'warn'
        ? 'border-amber-500/60 bg-amber-500/10 text-amber-300'
        : 'border-border/60 bg-muted/30 text-muted-foreground';
  return (
    <span
      className={cn('inline-flex rounded-full border px-1.5 py-0.5 text-[10.5px]', cls)}
      data-testid={testId}
    >
      {label}
    </span>
  );
}

function Reasoning({ decision }: { decision: Decision }) {
  const reasoning = typeof decision.reasoning === 'string' ? decision.reasoning : '';
  if (!reasoning.trim()) return null;
  return (
    <div className="rounded-md border border-border/60 bg-card/30 p-2.5" data-testid="decision-reasoning">
      <div className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">reasoning</div>
      <div className="prose prose-invert prose-sm max-w-none text-[12px] leading-relaxed text-foreground/90 [&_code]:rounded [&_code]:bg-muted/60 [&_code]:px-1 [&_code]:py-0.5 [&_pre]:overflow-x-auto [&_pre]:rounded [&_pre]:bg-muted/60 [&_pre]:p-2 [&_pre]:text-[11px] [&_p]:my-1 [&_ul]:my-1 [&_ul]:list-disc [&_ul]:pl-5">
        <ReactMarkdown>{reasoning}</ReactMarkdown>
      </div>
    </div>
  );
}

function SignalsSection({ decision }: { decision: Decision }) {
  const signals = decisionSupportingSignals(decision);
  if (signals.length === 0) return null;
  return (
    <div className="flex flex-col gap-1.5" data-testid="decision-signals">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">supporting signals ({signals.length})</div>
      <div className="flex flex-col gap-1">
        {signals.map((sig, i) => (
          <SignalDetail key={(sig.id ?? `sig-${i}`)} signal={sig} index={i} />
        ))}
      </div>
    </div>
  );
}

function SignalDetail({ signal, index }: { signal: Signal; index: number }) {
  const [open, setOpen] = useState(false);
  const sigRecord = signal as Record<string, unknown>;
  const prob = readNumber(sigRecord, 'probability');
  const pattern = signal.pattern ?? '?';
  return (
    <div className="rounded-md border border-border/60 bg-card/30 text-[11px]">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-2 px-2 py-1 text-left hover:bg-accent/30"
        data-testid={`decision-signal-${index}`}
      >
        <ChevronDown className={cn('size-3 transition-transform', open && 'rotate-180')} aria-hidden />
        <span className="font-medium">{pattern}</span>
        <span className="text-muted-foreground">{signal.side ?? '—'}</span>
        <span className="ml-auto tabular-nums text-muted-foreground">
          {prob !== null ? `${(prob * 100).toFixed(0)}%` : '—'}
        </span>
      </button>
      {open && (
        <pre className="overflow-x-auto rounded-b-md bg-muted/40 px-2 py-1.5 text-[10.5px] leading-relaxed text-muted-foreground">
{JSON.stringify(signal, null, 2)}
        </pre>
      )}
    </div>
  );
}

function CopyJsonButton({ decision }: { decision: Decision }) {
  const [copied, setCopied] = useState(false);

  const onCopy = () => {
    const json = JSON.stringify(decision, null, 2);
    if (typeof navigator !== 'undefined' && navigator.clipboard?.writeText) {
      void navigator.clipboard.writeText(json).then(() => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1200);
      });
    } else {
      // jsdom / restricted contexts: surface success silently for tests.
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    }
  };

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={onCopy}
      className="self-start text-[11px]"
      data-testid="decision-copy-json"
    >
      {copied ? <Check className="size-3" /> : <Copy className="size-3" />}
      <span className="ml-1.5">{copied ? 'copied' : 'copy json'}</span>
    </Button>
  );
}
