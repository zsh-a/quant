/**
 * LLM Analysis component.
 *
 * Sends pipeline state to LLM for analysis and displays results.
 * Users can provide custom instructions to guide the analysis.
 */
import React, { useCallback, useState } from 'react'
import { Loader2, Sparkles } from 'lucide-react'
import { alphaApi } from '../../utils/alphaApi'
import { Button } from '../ui/button'
import { Input } from '../ui/input'
import { SectionCard } from '../layout/SectionCard'

interface LLMAnalysisProps {
  jobId: string
  onApplySeeds?: (seeds: string[]) => void
}

export const LLMAnalysis: React.FC<LLMAnalysisProps> = ({ jobId, onApplySeeds }) => {
  const [instruction, setInstruction] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<{
    summary: string
    analysis: string
    suggested_seeds?: string[]
    suggested_operators?: string[]
    identified_weakness?: string
  } | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleAnalyze = useCallback(async () => {
    try {
      setLoading(true)
      setError(null)
      const r = await alphaApi.analyzeSearch(jobId, instruction || 'Analyze the search results and suggest improvements')
      setResult(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Analysis failed')
    } finally {
      setLoading(false)
    }
  }, [jobId, instruction])

  // Extract formula suggestions from analysis text
  const extractFormulas = useCallback((text: string): string[] => {
    const matches = text.match(/`(cs_rank\([^`]+\)|ts_\w+\([^`]+\))`/g)
    return matches ? matches.map(m => m.replace(/`/g, '')) : []
  }, [])

  return (
    <SectionCard title="LLM Analysis" description="Send pipeline state to LLM for diagnosis and suggestions">
      <div className="flex items-center gap-3">
        <Input
          value={instruction}
          onChange={e => setInstruction(e.target.value)}
          placeholder="Ask about the search results... (e.g., 'Why is IC low?')"
          className="flex-1"
        />
        <Button onClick={() => void handleAnalyze()} disabled={loading}>
          {loading ? <Loader2 className="animate-spin" /> : <Sparkles className="size-4" />}
          {loading ? 'Analyzing...' : 'Analyze'}
        </Button>
      </div>

      {error && (
        <div className="rounded-xl border border-rose-500/20 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </div>
      )}

      {result && (
        <div className="space-y-4">
          {/* Summary section */}
          <details className="group">
            <summary className="cursor-pointer text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition">
              Pipeline Summary (click to expand)
            </summary>
            <pre className="mt-2 whitespace-pre-wrap rounded-xl border border-border/50 bg-card/70 p-4 text-xs font-mono text-muted-foreground max-h-64 overflow-y-auto">
              {result.summary}
            </pre>
          </details>

          {/* Analysis */}
          <div className="rounded-xl border border-border/50 bg-card/70 p-4">
            <div className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-2">Analysis</div>
            <div className="prose prose-invert prose-sm max-w-none whitespace-pre-wrap text-sm text-foreground/90">
              {result.analysis}
            </div>
          </div>

          {/* Structured sections */}
          {result.identified_weakness && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-3">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-amber-400/90 mb-1">
                Identified Weakness
              </div>
              <div className="text-xs text-amber-100/90 whitespace-pre-wrap">{result.identified_weakness}</div>
            </div>
          )}

          {result.suggested_operators && result.suggested_operators.length > 0 && (
            <div className="rounded-xl border border-border/40 bg-card/70 p-3">
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Suggested Operators
              </div>
              <div className="flex flex-wrap gap-1.5">
                {result.suggested_operators.map(op => (
                  <span key={op} className="rounded-md bg-secondary/60 px-2 py-0.5 text-[11px] font-mono">{op}</span>
                ))}
              </div>
            </div>
          )}

          {/* Apply suggestions */}
          {onApplySeeds && (() => {
            const structured = result.suggested_seeds ?? []
            const fallback = structured.length ? [] : extractFormulas(result.analysis)
            const seeds = structured.length ? structured : fallback
            if (seeds.length === 0) return null
            return (
              <div className="rounded-xl border border-border/40 bg-card/70 p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground">
                    Suggested Seeds ({seeds.length})
                  </div>
                  <Button
                    variant="secondary"
                    size="sm"
                    onClick={() => onApplySeeds(seeds)}
                  >
                    Use as Seeds for Next Run
                  </Button>
                </div>
                <ul className="space-y-0.5">
                  {seeds.map(s => (
                    <li key={s} className="truncate font-mono text-[11px] text-foreground/80" title={s}>{s}</li>
                  ))}
                </ul>
              </div>
            )
          })()}
        </div>
      )}
    </SectionCard>
  )
}
