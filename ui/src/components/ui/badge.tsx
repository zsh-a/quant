import * as React from "react"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "../../lib/utils"

const badgeVariants = cva(
  "inline-flex items-center rounded-full border px-2.5 py-1 text-[11px] font-semibold uppercase tracking-[0.24em] transition-colors",
  {
    variants: {
      variant: {
        default: "border-border bg-secondary/70 text-secondary-foreground",
        success: "border-emerald-400/20 bg-emerald-400/10 text-emerald-200",
        warning: "border-amber-400/20 bg-amber-400/10 text-amber-200",
        danger: "border-rose-400/20 bg-rose-400/10 text-rose-200",
        info: "border-sky-400/20 bg-sky-400/10 text-sky-200",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  },
)

function Badge({
  className,
  variant,
  ...props
}: React.HTMLAttributes<HTMLDivElement> & VariantProps<typeof badgeVariants>) {
  return <div className={cn(badgeVariants({ variant }), className)} {...props} />
}

export { Badge, badgeVariants }
