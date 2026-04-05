/**
 * Typed API client for Alpha Lab endpoints.
 * Replaces inline fetch() calls in AlphaLabWorkspace.
 */
import { API_BASE } from './api'
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
} from '../types'

class AlphaApiError extends Error {
  constructor(public status: number, message: string) {
    super(message)
    this.name = 'AlphaApiError'
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${API_BASE}${path}`, init)
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
    summary_only?: boolean;
  }) => post<AlphaLabEvaluationSummary>('/alpha-lab/evaluate-db', params),

  // Search
  submitSearch: (params: Record<string, unknown>) =>
    post<{ job_id: string; status: string }>('/alpha-lab/search-db', params),

  getSearchJob: (jobId: string) =>
    request<AlphaLabSearchJob>(`/alpha-lab/search-jobs/${jobId}`),

  getSearchPipeline: (jobId: string) =>
    request<{ pipeline: AlphaPipelineRecord | null }>(`/alpha-lab/search-jobs/${jobId}/pipeline`),

  analyzeSearch: (jobId: string, instruction: string) =>
    post<{ summary: string; analysis: string; prompt_length: number }>(
      `/alpha-lab/search-jobs/${jobId}/analyze`, { instruction },
    ),

  // Zoo
  listZoo: (limit = 50) => request<{ entries: AlphaLabZooEntry[] }>(`/alpha-lab/zoo?limit=${limit}`),
  saveToZoo: (params: Record<string, unknown>) => post<AlphaLabZooEntry>('/alpha-lab/zoo', params),

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

  // Neural
  getNeuralHistory: () => request<AlphaLabTrainingHistory>('/alpha-lab/neural/history'),
}
