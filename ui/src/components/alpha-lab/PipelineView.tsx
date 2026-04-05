/**
 * Visual pipeline stage progression for alpha search.
 *
 * Renders a horizontal flow showing how formulas pass through each stage:
 *   Generate(12) -> Screen(12->8) -> Evaluate(8) -> Archive(+3)
 */
import React from 'react'
import type { AlphaRoundRecord, AlphaStageRecord } from '../../types'
import { fmtDur } from './shared'

function stageColor(stage: AlphaStageRecord): string {
  if (stage.kind === 'generate') return 'border-blue-500/40 bg-blue-500/10 text-blue-300'
  const passRate = stage.input > 0 ? stage.output / stage.input : 1
  if (passRate >= 0.7) return 'border-emerald-500/40 bg-emerald-500/10 text-emerald-300'
  if (passRate >= 0.3) return 'border-amber-500/40 bg-amber-500/10 text-amber-300'
  return 'border-rose-500/40 bg-rose-500/10 text-rose-300'
}

function stageLabel(kind: string): string {
  switch (kind) {
    case 'generate': return 'Generate'
    case 'quick_screen': return 'Screen'
    case 'evaluate': return 'Evaluate'
    case 'fitness': return 'Fitness'
    case 'archive': return 'Archive'
    default: return kind
  }
}

function StageBox({ stage }: { stage: AlphaStageRecord }) {
  const arrow = stage.kind === 'generate'
    ? `${stage.output}`
    : `${stage.input} \u2192 ${stage.output}`

  return (
    <div className={`flex flex-col items-center rounded-xl border px-3 py-2 min-w-[90px] ${stageColor(stage)}`}>
      <div className="text-[10px] font-semibold uppercase tracking-wider">{stageLabel(stage.kind)}</div>
      <div className="mt-1 text-sm font-semibold font-mono">{arrow}</div>
      <div className="mt-0.5 text-[10px] opacity-70">{fmtDur(stage.duration_ms)}</div>
    </div>
  )
}

function RoundPipeline({ round }: { round: AlphaRoundRecord }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-semibold text-muted-foreground">R{round.round}</span>
        <span className="text-[10px] text-muted-foreground">{round.strategies.join(', ')}</span>
        <span className="ml-auto text-[10px] text-muted-foreground">{fmtDur(round.duration_ms)}</span>
      </div>
      <div className="flex items-center gap-2 overflow-x-auto">
        {round.stages.map((stage, i) => (
          <React.Fragment key={`${stage.kind}-${stage.strategy}-${i}`}>
            {i > 0 && <span className="text-muted-foreground/50 text-sm shrink-0">&rarr;</span>}
            <StageBox stage={stage} />
          </React.Fragment>
        ))}
        {round.archive_size > 0 && (
          <>
            <span className="text-muted-foreground/50 text-sm shrink-0">&rarr;</span>
            <div className="flex flex-col items-center rounded-xl border border-violet-500/40 bg-violet-500/10 text-violet-300 px-3 py-2 min-w-[90px]">
              <div className="text-[10px] font-semibold uppercase tracking-wider">Archive</div>
              <div className="mt-1 text-sm font-semibold font-mono">{round.archive_size}</div>
              <div className="mt-0.5 text-[10px] opacity-70">best {round.best_fitness.toFixed(3)}</div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}

interface PipelineViewProps {
  rounds: AlphaRoundRecord[]
  compact?: boolean
}

export const PipelineView: React.FC<PipelineViewProps> = ({ rounds, compact = false }) => {
  if (rounds.length === 0) return null
  const display = compact ? rounds.slice(-3) : rounds
  return (
    <div className="space-y-4">
      {display.map(round => (
        <RoundPipeline key={round.round} round={round} />
      ))}
    </div>
  )
}
