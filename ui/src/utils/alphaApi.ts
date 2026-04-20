/**
 * Typed API client for Alpha Lab endpoints.
 * Replaces inline fetch() calls in AlphaLabWorkspace.
 */
import { API_BASE, authHeaders } from './api'
import type {
  AlphaLabCombineResult,
  AlphaLabEvaluationSummary,
  AlphaLabRunDetail,
  AlphaLabSearchJob,
  AlphaLabTrainingHistory,
  AlphaLabValidationReport,
  AlphaLabWorkspace,
  AlphaLabZooEntry,
  AlphaPipelineRecord,
  CheckpointEntry,
  EventBacktestParams,
  EventBacktestResponse,
  FactorCatalogResponse,
  StrategyStateResponse,
} from '../types'

class AlphaApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'AlphaApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const headers = authHeaders(init?.headers as Record<string, string> | undefined)
  const r = await fetch(`${API_BASE}${path}`, { ...init, headers })
  const body = await r.json().catch(() => null)
  if (!r.ok) throw new AlphaApiError(r.status, body?.detail ?? 'request failed')
  return body as T
}

function post<T>(path: string, body: unknown): Promise<T> {
  return request<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
}

export const alphaApi = {
  // Workspace
  getWorkspace: () => request<AlphaLabWorkspace>('/alpha-lab/workspace'),

  // Formula
  validate: (formula: string) =>
    post<AlphaLabValidationReport>('/alpha-lab/validate', { formula }),

  evaluateDb: (params: {
    formula: string; symbols: string[]; start_time: string; end_time: string;
    interval?: string; min_quote_volume?: number; blocked_utc_hours?: number[];
    summary_only?: boolean; market?: string; universe?: string; exclude_st?: boolean;
  }) => post<AlphaLabEvaluationSummary>('/alpha-lab/evaluate-db', params),

  // Search
  submitSearch: (params: Record<string, unknown>) =>
    post<{ job_id: string; status: string; request_id?: string }>('/alpha-lab/search-db', params),

  getSearchJob: (jobId: string) =>
    request<AlphaLabSearchJob>(`/alpha-lab/search-jobs/${jobId}`),

  listSearchJobs: () =>
    request<{ jobs: AlphaLabSearchJob[] }>('/alpha-lab/search-jobs'),

  cancelSearchJob: (jobId: string) =>
    post<{ job_id: string; status: string; already_settled?: boolean }>(
      `/alpha-lab/search-jobs/${jobId}/cancel`, {},
    ),

  getSearchPipeline: (jobId: string) =>
    request<{ pipeline: AlphaPipelineRecord | null }>(`/alpha-lab/search-jobs/${jobId}/pipeline`),

  analyzeSearch: (jobId: string, instruction: string) =>
    post<{
      summary: string
      analysis: string
      prompt_length: number
      suggested_seeds?: string[]
      suggested_operators?: string[]
      identified_weakness?: string
    }>(`/alpha-lab/search-jobs/${jobId}/analyze`, { instruction }),

  // Zoo
  listZoo: (limit = 50) => request<{ entries: AlphaLabZooEntry[] }>(`/alpha-lab/zoo?limit=${limit}`),
  saveToZoo: (params: Record<string, unknown>) => post<AlphaLabZooEntry>('/alpha-lab/zoo', params),

  listSearchPresets: () =>
    request<{
      presets: Array<{
        key: string; label: string; hint: string; strategy: string;
        params: Record<string, number>
        budget?: { max_wall_time_sec?: number; max_full_eval?: number; max_llm_tokens?: number }
      }>
      auto_archive?: { top_k: number }
    }>('/alpha-lab/search-presets'),

  promoteZooToSimulation: (factorId: string, params: Record<string, unknown>) =>
    post<{ job_id: string; source_zoo_factor_id: string; job: Record<string, unknown> }>(
      `/alpha-lab/zoo/${encodeURIComponent(factorId)}/promote-to-simulation`, params,
    ),

  // Lineage
  getLineage: (kind: string, nodeId: string, maxDepth = 4) =>
    request<{
      root: { kind: string; id: string }
      nodes: { kind: string; id: string }[]
      edges: {
        parent_kind: string; parent_id: string;
        child_kind: string; child_id: string;
        relation: string; meta: Record<string, unknown>
      }[]
    }>(`/alpha-lab/lineage/${encodeURIComponent(kind)}/${encodeURIComponent(nodeId)}?max_depth=${maxDepth}`),

  // Runs
  listRuns: (limit = 20) => request<{ runs: unknown[] }>(`/alpha-lab/runs?limit=${limit}`),
  getRun: (runId: string) => request<AlphaLabRunDetail>(`/alpha-lab/runs/${runId}`),

  // Combine
  combineZoo: (params: Record<string, unknown>) =>
    post<AlphaLabCombineResult>('/alpha-lab/combine-zoo', params),

  // Tracing
  getTracingSummary: () => request<Record<string, unknown>>('/alpha-lab/tracing/summary'),
  getTracingSpans: (limit = 50) =>
    request<{ spans: unknown[]; total: number }>(`/alpha-lab/tracing/spans?limit=${limit}`),
  getTracingByRequest: (requestId: string) =>
    request<{ request_id: string; spans: Record<string, unknown>[]; total: number; enabled: boolean }>(
      `/alpha-lab/tracing/request/${encodeURIComponent(requestId)}`,
    ),
  listTracingRequests: (limit = 50) =>
    request<{ requests: string[]; total: number; enabled: boolean }>(
      `/alpha-lab/tracing/requests?limit=${limit}`,
    ),

  // Neural
  getNeuralHistory: () => request<AlphaLabTrainingHistory>('/alpha-lab/neural/history'),

  // Strategy State Management
  getStrategyState: () => request<StrategyStateResponse>('/alpha-lab/strategy-state'),
  listCheckpoints: () => request<{ checkpoints: CheckpointEntry[] }>('/alpha-lab/checkpoints'),
  getJobCheckpoints: (jobId: string) =>
    request<{ job_id: string; checkpoints: CheckpointEntry[] }>(`/alpha-lab/checkpoints/${jobId}`),
  getFactorCatalog: (params?: {
    strategy?: string; round_idx?: number; min_ic?: number;
    evaluated_only?: boolean; limit?: number;
  }) => {
    const qs = new URLSearchParams()
    if (params?.strategy) qs.set('strategy', params.strategy)
    if (params?.round_idx != null) qs.set('round_idx', String(params.round_idx))
    if (params?.min_ic != null) qs.set('min_ic', String(params.min_ic))
    if (params?.evaluated_only) qs.set('evaluated_only', 'true')
    if (params?.limit) qs.set('limit', String(params.limit))
    const q = qs.toString()
    return request<FactorCatalogResponse>(`/alpha-lab/factor-catalog${q ? '?' + q : ''}`)
  },
  getFactorCatalogStats: () =>
    request<{ strategies: Record<string, unknown>; total_factors: number; job_id?: string }>('/alpha-lab/factor-catalog/stats'),

  // Event Engine Backtest
  runEventBacktest: (params: EventBacktestParams) =>
    post<EventBacktestResponse>('/alpha-lab/event-backtest', params),

  // Factor Factory
  runFactory: (params: Record<string, unknown>) =>
    post<{ job_id: string; status: string }>('/alpha-lab/factory/run', params),

  // Cross-market Migration
  migrateFormula: (params: { formula: string; source_market: string; target_market: string }) =>
    post<{
      original_formula: string
      migrated_formula: string | null
      source_market: string
      target_market: string
      field_mappings: Record<string, string>
      unmappable_fields: string[]
      is_viable: boolean
    }>('/alpha-lab/migrate', params),
}
