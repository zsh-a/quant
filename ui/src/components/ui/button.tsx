import * as React from "react"
import { Slot } from "@radix-ui/react-slot"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "../../lib/utils"

const buttonVariants = cva(
  "inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-xl text-sm font-medium transition-all duration-200 outline-none ring-offset-background disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg]:size-4 shrink-0 focus-visible:ring-2 focus-visible:ring-ring/70 focus-visible:ring-offset-2",
  {
    variants: {
      variant: {
        default:
          "bg-primary text-primary-foreground shadow-[0_18px_36px_-18px_hsl(var(--primary)/0.95)] hover:bg-[color-mix(in_oklab,hsl(var(--primary))_88%,white)]",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-[color-mix(in_oklab,hsl(var(--secondary))_82%,white)]",
        outline:
          "border border-primary/25 bg-card text-foreground shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] hover:border-primary/45 hover:bg-accent hover:text-accent-foreground",
        ghost:
          "text-foreground/82 hover:bg-accent hover:text-accent-foreground",
        danger:
          "bg-destructive text-destructive-foreground shadow-[0_18px_36px_-20px_rgba(244,63,94,0.9)] hover:bg-[color-mix(in_oklab,hsl(var(--destructive))_90%,white)]",
      },
      size: {
        default: "h-10 px-4 py-2",
        sm: "h-8 rounded-lg px-3 text-xs",
        lg: "h-11 px-5 text-sm",
        icon: "size-10",
      },
    },
    defaultVariants: {
      variant: "default",
      size: "default",
    },
  },
)

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {
  asChild?: boolean
}

const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant, size, asChild = false, ...props }, ref) => {
    const Comp = asChild ? Slot : "button"
    return (
      <Comp
        className={cn(buttonVariants({ variant, size, className }))}
        ref={ref}
        {...props}
      />
    )
  },
)
Button.displayName = "Button"

export { Button, buttonVariants }
