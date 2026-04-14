/**
 * Shared constants, helpers and types for Alpha Lab tab components.
 */
import { formatPercent, formatPrice } from '../../utils/format'

import { API_BASE, apiFetch, getToken } from '../../utils/api'

export const METRIC_KEYS = ['sharpe', 'rank_ic', 'ic_ir', 'total_return', 'max_drawdown', 'avg_turnover', 'signal_coverage', 'pnl_per_turnover'] as const
export const IC_DETAIL_KEYS = ['rank_ic_1d', 'rank_ic_5d', 'rank_ic_10d', 'ic_decay', 'ic_std', 'turnover_proxy'] as const
export const PCT_METRICS = new Set(['total_return', 'max_drawdown', 'volatility', 'signal_coverage', 'turnover_proxy'])
export const LABELS: Record<string, string> = {
  sharpe: 'Sharpe', rank_ic: 'Rank IC', ic_ir: 'IC IR', ic_std: 'IC Std', ic_decay: 'IC Decay',
  pnl_per_turnover: 'PnL/Turnover', total_return: 'Total Return', max_drawdown: 'Max DD',
  avg_turnover: 'Avg Turnover', signal_coverage: 'Coverage', turnover_proxy: 'Turnover Proxy',
  rank_ic_1d: 'IC 1D', rank_ic_5d: 'IC 5D', rank_ic_10d: 'IC 10D', pnl_efficiency_score: 'PnL Eff',
}
export const SEARCH_POLL_MS = 3000
export const TRACING_POLL_MS = 10_000

export const pad = (n: number) => String(n).padStart(2, '0')
export const dtLocal = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`
export const dtDate = (d: Date) => `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`
export const toISO = (v: string) => { const d = new Date(v); return Number.isNaN(d.getTime()) ? v : d.toISOString() }

export const DEFAULT_LOOKBACK_YEARS = 3

export const RANGE_PRESETS = [
  { label: '1M', days: 30 },
  { label: '3M', days: 90 },
  { label: '1Y', days: 365 },
  { label: '2Y', days: 730 },
  { label: '3Y', days: 1095 },
  { label: '5Y', days: 1825 },
] as const

export function fmt(key: string, v?: number | null) {
  const n = Number(v)
  if (!Number.isFinite(n)) return '--'
  if (PCT_METRICS.has(key)) return formatPercent(n, 2)
  return Math.abs(n) >= 10 ? formatPrice(n, 2) : formatPrice(n, 4)
}

export function fmtTime(v?: string | null) {
  if (!v) return '--'
  const d = new Date(v)
  return Number.isNaN(d.getTime()) ? v : d.toLocaleString('zh-CN', { hour12: false })
}

export function fmtDur(ms: number) {
  if (ms < 1000) return `${Math.round(ms)}ms`
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`
  return `${(ms / 60_000).toFixed(1)}m`
}

export function fmtTokens(n: number) {
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}K`
  return String(n)
}

export { API_BASE, apiFetch, getToken }

/** Shared context shape for tab components */
export interface AlphaLabContext {
  formula: string
  setFormula: (f: string) => void
  interval: string
  symbols: string
  startTime: string
  endTime: string
  symList: () => string[]
  loadWorkspace: () => Promise<void>
  loading: boolean
  handleLoadFormula: (f: string) => void
  setErr: (e: string | null) => void
}
