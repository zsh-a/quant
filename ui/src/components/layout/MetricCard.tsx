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
        "group relative overflow-hidden rounded-xl border border-border/60 bg-card p-5 transition-all duration-300 hover:border-border hover:bg-accent/20",
        className,
      )}
    >
      {/* Accent top line on hover */}
      <div className="absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/40 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />

      <div className="space-y-2.5">
        <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
          {label}
        </p>
        <div className="text-2xl font-semibold tracking-tight text-foreground" style={{ fontVariantNumeric: 'tabular-nums' }}>
          {value}
        </div>
        {trend && <div className="text-sm font-medium text-foreground/80">{trend}</div>}
        {hint && <div className="text-xs text-muted-foreground">{hint}</div>}
      </div>
    </div>
  )
}
