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
    <div className={cn("flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between", className)}>
      <div className="space-y-2.5">
        {eyebrow ? (
          <p className="text-[11px] font-semibold uppercase tracking-[0.24em] text-primary/85">
            {eyebrow}
          </p>
        ) : null}
        <div className="space-y-1.5">
          <h1 className="text-3xl font-semibold tracking-[-0.035em] text-foreground sm:text-4xl xl:text-[2.65rem]">
            {title}
          </h1>
          {description ? <p className="max-w-3xl text-sm leading-7 text-muted-foreground sm:text-[15px]">{description}</p> : null}
        </div>
      </div>
      {actions ? <div className="flex items-center gap-3">{actions}</div> : null}
    </div>
  )
}
