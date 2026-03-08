import type { ReactNode } from "react"

import { cn } from "../../lib/utils"

interface MetricCardProps {
  label: string
  value: ReactNode
  hint?: ReactNode
  trend?: ReactNode
  className?: string
}

export function MetricCard({ label, value, hint, trend, className }: MetricCardProps) {
  return (
    <div
      className={cn(
        "group relative overflow-hidden rounded-3xl border border-border/70 bg-card/80 p-5 shadow-[0_20px_60px_-40px_rgba(15,23,42,0.85)] transition-transform duration-200 hover:-translate-y-1",
        className,
      )}
    >
      <div className="absolute inset-x-5 top-0 h-px bg-gradient-to-r from-transparent via-primary/50 to-transparent" />
      <div className="space-y-3.5">
        <p className="text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">
          {label}
        </p>
        <div className="text-[1.9rem] font-semibold tracking-[-0.03em] text-foreground">{value}</div>
        {trend ? <div className="text-sm font-semibold tracking-[-0.012em] text-foreground/92">{trend}</div> : null}
        {hint ? <div className="text-[13px] leading-6 text-muted-foreground">{hint}</div> : null}
      </div>
    </div>
  )
}
