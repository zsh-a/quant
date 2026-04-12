import type { ReactNode } from "react"

import { Card, CardContent } from "../ui/card"

interface EmptyStateProps {
  title: string
  description: string
  action?: ReactNode
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <Card className="border-dashed">
      <CardContent className="flex min-h-52 flex-col items-center justify-center gap-3 p-8 text-center">
        <div className="space-y-1.5">
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          <p className="max-w-sm text-[13px] text-muted-foreground">{description}</p>
        </div>
        {action}
      </CardContent>
    </Card>
  )
}
