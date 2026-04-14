import React, { useState } from 'react'
import { Loader2, Play, Factory, CheckCircle2, XCircle } from 'lucide-react'
import { Button } from '../ui/button'
import { SectionCard } from '../layout/SectionCard'
import { MetricCard } from '../layout/MetricCard'
import { Badge } from '../ui/badge'
import { Switch } from '../ui/switch'
import { alphaApi } from '../../utils/alphaApi'
import { toast } from 'sonner'

export function FactoryTab() {
  const [markets, setMarkets] = useState<string[]>(['crypto'])
  const [generations, setGenerations] = useState(5)
  const [topK, setTopK] = useState(30)
  const [running, setRunning] = useState(false)
  const [result, setResult] = useState<Record<string, unknown> | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)

  const toggleMarket = (m: string) => {
    setMarkets((prev) => (prev.includes(m) ? prev.filter((x) => x !== m) : [...prev, m]))
  }

  const handleRun = async () => {
    setRunning(true)
    setResult(null)
    try {
      const resp = await alphaApi.runFactory({ markets, generations, top_k: topK })
      setJobId(resp.job_id)
      toast.success(`Factory job started: ${resp.job_id}`)
      // Poll for completion
      const poll = async () => {
        try {
          const jobs = await alphaApi.listSearchJobs()
          const job = (jobs as any[]).find((j: any) => j.job_id === resp.job_id)
          if (job?.status === 'completed') {
            setResult(job.result)
            setRunning(false)
            toast.success('Factor factory completed')
          } else if (job?.status === 'failed') {
            setRunning(false)
            toast.error(`Factory failed: ${job.error}`)
          } else {
            setTimeout(poll, 3000)
          }
        } catch {
          setTimeout(poll, 5000)
        }
      }
      setTimeout(poll, 3000)
    } catch (err) {
      setRunning(false)
      toast.error('Failed to start factory')
    }
  }

  return (
    <div className="space-y-5">
      <SectionCard title="Factor Factory" description="全自动 搜索 → 评估 → 入库 → 组合 流水线">
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-3">
            <div className="text-xs font-medium text-muted-foreground uppercase tracking-wider">Markets</div>
            <div className="flex gap-2">
              {['crypto', 'a_share'].map((m) => (
                <label key={m} className="flex items-center gap-2 cursor-pointer select-none">
                  <Switch checked={markets.includes(m)} onCheckedChange={() => toggleMarket(m)} />
                  <span className="text-sm">{m === 'crypto' ? 'Crypto' : 'A-Share'}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-muted-foreground">Generations</label>
              <input
                type="number"
                className="glass-input mt-1"
                value={generations}
                onChange={(e) => setGenerations(Number(e.target.value))}
                min={1}
                max={50}
              />
            </div>
            <div>
              <label className="text-xs text-muted-foreground">Top K</label>
              <input
                type="number"
                className="glass-input mt-1"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                min={5}
                max={200}
              />
            </div>
          </div>
        </div>
        <div className="mt-4">
          <Button onClick={handleRun} disabled={running || markets.length === 0} className="gap-2">
            {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
            {running ? 'Running...' : 'Run Factory Pipeline'}
          </Button>
          {jobId && <span className="ml-3 text-xs text-muted-foreground">Job: {jobId}</span>}
        </div>
      </SectionCard>

      {result && (
        <SectionCard title="Pipeline Results">
          <div className="grid gap-3 sm:grid-cols-4">
            <MetricCard label="Zoo Size" value={String((result as any).zoo_count ?? '--')} />
            <MetricCard
              label="Combination"
              value={(result as any).combination?.status === 'completed' ? 'OK' : (result as any).combination?.status ?? '--'}
            />
            <MetricCard label="Decaying" value={String(((result as any).decaying_factors ?? []).length)} hint="衰减因子" />
            <MetricCard label="Total Time" value={`${((result as any).timing?.total ?? 0).toFixed(1)}s`} />
          </div>
          {Object.entries((result as any).search_results ?? {}).map(([market, r]: [string, any]) => (
            <div key={market} className="mt-3 flex items-center gap-2 text-sm">
              {r.status === 'completed' ? (
                <CheckCircle2 size={14} className="text-emerald-500" />
              ) : (
                <XCircle size={14} className="text-red-500" />
              )}
              <span className="font-medium">{market}</span>
              <span className="text-muted-foreground">
                {r.status === 'completed'
                  ? `${r.archive_size} factors, ${r.total_evaluated} evaluated`
                  : r.error ?? r.status}
              </span>
            </div>
          ))}
        </SectionCard>
      )}
    </div>
  )
}
