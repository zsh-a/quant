/**
 * LineageView — visualise the provenance DAG rooted at a given node.
 *
 * Rendered as a sankey diagram: left-to-right flow from the earliest
 * ancestor (search_job / data_update_run) through zoo_factor nodes into
 * simulation_job and simulation_run leaves.  Edge labels carry the
 * relation kind (produced / derived_from / promoted_to / backtests /
 * similar_to).
 */
import { useEffect, useMemo, useState } from 'react'
import ReactEChartsCore from 'echarts-for-react/lib/core'
import * as echarts from 'echarts/core'
import { SankeyChart } from 'echarts/charts'
import { TooltipComponent } from 'echarts/components'
import { CanvasRenderer } from 'echarts/renderers'
import { Loader2 } from 'lucide-react'
import { alphaApi } from '../../utils/alphaApi'

echarts.use([SankeyChart, TooltipComponent, CanvasRenderer])

const KIND_COLOR: Record<string, string> = {
  search_job: '#6366f1',
  zoo_factor: '#10b981',
  simulation_job: '#f59e0b',
  simulation_run: '#ef4444',
  data_update_run: '#94a3b8',
}

const RELATION_COLOR: Record<string, string> = {
  produced: 'rgba(99,102,241,0.55)',
  derived_from: 'rgba(16,185,129,0.55)',
  promoted_to: 'rgba(245,158,11,0.55)',
  backtests: 'rgba(239,68,68,0.55)',
  similar_to: 'rgba(148,163,184,0.45)',
  triggered_by: 'rgba(56,189,248,0.55)',
}

function shortId(kind: string, id: string): string {
  if (!id) return kind
  if (id.length <= 10) return `${kind}:${id}`
  return `${kind}:${id.slice(0, 8)}…`
}

interface Props {
  kind: string
  nodeId: string
  maxDepth?: number
  height?: number
  onSelect?: (kind: string, id: string) => void
}

export function LineageView({ kind, nodeId, maxDepth = 4, height = 360, onSelect }: Props) {
  const [loading, setLoading] = useState(false)
  const [err, setErr] = useState<string | null>(null)
  const [graph, setGraph] = useState<Awaited<ReturnType<typeof alphaApi.getLineage>> | null>(null)

  useEffect(() => {
    let cancelled = false
    if (!nodeId) return
    setLoading(true)
    setErr(null)
    alphaApi.getLineage(kind, nodeId, maxDepth)
      .then(g => { if (!cancelled) setGraph(g) })
      .catch(e => { if (!cancelled) setErr(e instanceof Error ? e.message : 'lineage load failed') })
      .finally(() => { if (!cancelled) setLoading(false) })
    return () => { cancelled = true }
  }, [kind, nodeId, maxDepth])

  const option = useMemo(() => {
    if (!graph) return null
    const nodes = graph.nodes.map(n => ({
      name: `${n.kind}:${n.id}`,
      itemStyle: { color: KIND_COLOR[n.kind] ?? '#64748b' },
      label: { formatter: shortId(n.kind, n.id), color: '#94a3b8', fontSize: 10 },
    }))
    const links = graph.edges.map(e => ({
      source: `${e.parent_kind}:${e.parent_id}`,
      target: `${e.child_kind}:${e.child_id}`,
      value: 1,
      lineStyle: { color: RELATION_COLOR[e.relation] ?? 'rgba(148,163,184,0.35)' },
      label: { show: true, formatter: e.relation, fontSize: 9, color: '#64748b' },
    }))
    return {
      backgroundColor: 'transparent',
      tooltip: {
        trigger: 'item',
        formatter: (params: any) => {
          if (params.dataType === 'edge') {
            return `${params.data.source}<br/><b>${params.data.label?.formatter}</b><br/>${params.data.target}`
          }
          return params.name
        },
      },
      series: [
        {
          type: 'sankey',
          emphasis: { focus: 'adjacency' },
          nodeAlign: 'justify',
          nodeGap: 12,
          nodeWidth: 14,
          data: nodes,
          links,
          lineStyle: { curveness: 0.5 },
        },
      ],
    }
  }, [graph])

  const onEvents = useMemo(() => ({
    click: (params: any) => {
      if (!onSelect || params.dataType !== 'node') return
      const [k, ...rest] = String(params.name).split(':')
      onSelect(k, rest.join(':'))
    },
  }), [onSelect])

  if (loading) {
    return (
      <div className="flex items-center gap-2 py-6 text-xs text-muted-foreground">
        <Loader2 className="size-3 animate-spin" /> loading lineage…
      </div>
    )
  }
  if (err) {
    return <div className="py-3 text-xs text-red-400">{err}</div>
  }
  if (!graph || graph.nodes.length === 0) {
    return <div className="py-3 text-xs text-muted-foreground">No lineage recorded for this node.</div>
  }
  if (!option) return null

  return (
    <div className="rounded-xl border border-border/40 bg-card/60 p-2">
      <ReactEChartsCore
        echarts={echarts}
        option={option}
        onEvents={onEvents}
        style={{ height, width: '100%' }}
        opts={{ renderer: 'canvas' }}
      />
      <div className="mt-2 flex flex-wrap gap-2 text-[10px] text-muted-foreground">
        {Object.entries(KIND_COLOR).map(([k, c]) => (
          <span key={k} className="inline-flex items-center gap-1">
            <span className="inline-block size-2 rounded" style={{ background: c }} />
            {k}
          </span>
        ))}
      </div>
    </div>
  )
}
