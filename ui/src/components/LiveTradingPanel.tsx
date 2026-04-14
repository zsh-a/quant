/**
 * 实盘交易面板 — 仓位、订单状态、P&L、紧急控制
 */
import React, { useCallback, useEffect, useState } from 'react'
import {
  AlertTriangle, DollarSign, PauseCircle, PlayCircle,
  RefreshCw, ShieldOff, TrendingUp,
} from 'lucide-react'
import { toast } from 'sonner'

import { apiFetch } from '../utils/api'
import { Button } from './ui/button'
import { Badge } from './ui/badge'
import { SectionCard } from './layout/SectionCard'
import { MetricCard } from './layout/MetricCard'

interface Props {
  sessionId: string
}

interface AccountInfo {
  cash: number
  total_equity: number
  positions: Record<string, number>
  detailed_positions: Record<string, {
    qty: number; name: string; price: number; value: number
    avg_cost: number; unrealized_pnl: number; pnl_pct: number
  }>
  pending_orders: any[]
}

export function LiveTradingPanel({ sessionId }: Props) {
  const [account, setAccount] = useState<AccountInfo | null>(null)
  const [loading, setLoading] = useState(false)

  const fetchAccount = useCallback(async () => {
    setLoading(true)
    try {
      const resp = await apiFetch(`/session/${sessionId}/status`)
      if (resp.ok) {
        const data = await resp.json()
        if (data.account_info) setAccount(data.account_info)
      }
    } catch { /* ignore */ }
    setLoading(false)
  }, [sessionId])

  useEffect(() => {
    fetchAccount()
    const timer = setInterval(fetchAccount, 5000)
    return () => clearInterval(timer)
  }, [fetchAccount])

  const handleStop = async () => {
    if (!confirm('确定要停止实盘交易？未平仓位将保留。')) return
    try {
      await apiFetch(`/session/${sessionId}/stop`, { method: 'POST' })
      toast.success('交易已停止')
    } catch {
      toast.error('停止失败')
    }
  }

  const positions = account?.detailed_positions ?? {}
  const posEntries = Object.entries(positions)
  const totalPnl = posEntries.reduce((sum, [, p]) => sum + p.unrealized_pnl, 0)

  return (
    <div className="space-y-5">
      {/* Metrics Row */}
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Cash"
          value={account ? `¥${account.cash.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '--'}
          hint="可用资金"
        />
        <MetricCard
          label="Total Equity"
          value={account ? `¥${account.total_equity.toLocaleString(undefined, { maximumFractionDigits: 0 })}` : '--'}
          hint="总权益 (现金 + 持仓市值)"
        />
        <MetricCard
          label="Unrealized P&L"
          value={
            account ? (
              <span className={totalPnl >= 0 ? 'text-emerald-500' : 'text-red-500'}>
                ¥{totalPnl.toLocaleString(undefined, { maximumFractionDigits: 0 })}
              </span>
            ) : '--'
          }
          hint="未实现盈亏"
        />
        <MetricCard
          label="Positions"
          value={posEntries.length}
          hint="当前持仓数"
        />
      </div>

      {/* Positions Table */}
      <SectionCard
        title="Positions"
        description="当前持仓"
        action={
          <Button variant="ghost" size="sm" onClick={fetchAccount} disabled={loading}>
            <RefreshCw size={13} className={loading ? 'animate-spin' : ''} />
          </Button>
        }
      >
        {posEntries.length === 0 ? (
          <div className="py-6 text-center text-sm text-muted-foreground">无持仓</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border/50 text-xs text-muted-foreground">
                  <th className="pb-2 text-left font-medium">标的</th>
                  <th className="pb-2 text-right font-medium">数量</th>
                  <th className="pb-2 text-right font-medium">现价</th>
                  <th className="pb-2 text-right font-medium">成本</th>
                  <th className="pb-2 text-right font-medium">市值</th>
                  <th className="pb-2 text-right font-medium">盈亏</th>
                  <th className="pb-2 text-right font-medium">盈亏%</th>
                </tr>
              </thead>
              <tbody>
                {posEntries.map(([code, p]) => (
                  <tr key={code} className="border-b border-border/30">
                    <td className="py-2">
                      <div className="font-medium">{p.name}</div>
                      <div className="text-xs text-muted-foreground">{code}</div>
                    </td>
                    <td className="py-2 text-right tabular-nums">{p.qty}</td>
                    <td className="py-2 text-right tabular-nums">{p.price.toFixed(2)}</td>
                    <td className="py-2 text-right tabular-nums">{p.avg_cost.toFixed(2)}</td>
                    <td className="py-2 text-right tabular-nums">{p.value.toLocaleString()}</td>
                    <td className={`py-2 text-right tabular-nums ${p.unrealized_pnl >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                      {p.unrealized_pnl >= 0 ? '+' : ''}{p.unrealized_pnl.toFixed(0)}
                    </td>
                    <td className={`py-2 text-right tabular-nums ${p.pnl_pct >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
                      {(p.pnl_pct * 100).toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </SectionCard>

      {/* Pending Orders */}
      {(account?.pending_orders?.length ?? 0) > 0 && (
        <SectionCard title="Pending Orders" description="等待成交的订单">
          <div className="space-y-2">
            {account!.pending_orders.map((o: any, i: number) => (
              <div key={i} className="flex items-center justify-between rounded-lg bg-secondary/30 px-3 py-2 text-sm">
                <span>
                  <Badge variant="outline" className="mr-2">{o.type}</Badge>
                  {o.symbol} × {o.quantity}
                </span>
                <span className="text-xs text-muted-foreground">{o.status}</span>
              </div>
            ))}
          </div>
        </SectionCard>
      )}

      {/* Emergency Controls */}
      <SectionCard title="Controls" description="交易控制">
        <div className="flex gap-3">
          <Button variant="destructive" size="sm" className="gap-1.5" onClick={handleStop}>
            <PauseCircle size={14} />
            停止交易
          </Button>
        </div>
      </SectionCard>
    </div>
  )
}
