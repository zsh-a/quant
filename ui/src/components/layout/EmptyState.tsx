import type { ReactNode } from "react"

import { Card, CardContent } from "../ui/card"

interface EmptyStateProps {
  title: string
  description: string
  action?: ReactNode
}

export function EmptyState({ title, description, action }: EmptyStateProps) {
  return (
    <Card className="border-dashed border-border/40">
      <CardContent className="flex min-h-56 flex-col items-center justify-center gap-4 p-10 text-center">
        <div className="space-y-2">
          <h2 className="text-sm font-semibold text-foreground">{title}</h2>
          <p className="max-w-sm text-sm text-muted-foreground">{description}</p>
        </div>
        {action}
      </CardContent>
    </Card>
  )
}
