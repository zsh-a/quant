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
        "group relative rounded-lg border border-border bg-card p-4 transition-colors hover:border-border/80 hover:bg-accent/30",
        className,
      )}
    >
      {/* Accent top line */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/30 to-transparent opacity-0 transition-opacity group-hover:opacity-100" />

      <div className="space-y-2">
        <p className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground">
          {label}
        </p>
        <div className="text-2xl font-semibold tracking-tight text-foreground" style={{ fontVariantNumeric: 'tabular-nums' }}>
          {value}
        </div>
        {trend && <div className="text-sm font-medium text-foreground/80">{trend}</div>}
        {hint && <div className="text-[12px] text-muted-foreground">{hint}</div>}
      </div>
    </div>
  )
}
