import type { ReactNode } from "react"

import { cn } from "../../lib/utils"

interface PageHeaderProps {
  eyebrow?: string
  title: string
  description?: string
  actions?: ReactNode
  className?: string
}

export function PageHeader({ eyebrow, title, description, actions, className }: PageHeaderProps) {
  return (
    <div className={cn("flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between", className)}>
      <div className="space-y-1.5">
        {eyebrow && (
          <p className="text-[11px] font-semibold uppercase tracking-widest text-primary/80">
            {eyebrow}
          </p>
        )}
        <h1 className="text-lg font-semibold tracking-tight text-foreground">
          {title}
        </h1>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      {actions && <div className="flex items-center gap-2.5">{actions}</div>}
    </div>
  )
}
