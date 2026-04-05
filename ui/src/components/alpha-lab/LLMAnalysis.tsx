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
  const [result, setResult] = useState<{ summary: string; analysis: string } | null>(null)
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

          {/* Apply suggestions */}
          {onApplySeeds && (() => {
            const formulas = extractFormulas(result.analysis)
            if (formulas.length === 0) return null
            return (
              <div className="flex items-center gap-2">
                <span className="text-xs text-muted-foreground">{formulas.length} formulas found in analysis</span>
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => onApplySeeds(formulas)}
                >
                  Apply as Seeds
                </Button>
              </div>
            )
          })()}
        </div>
      )}
    </SectionCard>
  )
}
