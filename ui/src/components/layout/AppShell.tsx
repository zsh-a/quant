import type { ReactNode } from "react"
import { Menu } from "lucide-react"

import { Button } from "../ui/button"
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "../ui/sheet"

interface AppShellProps {
  sidebar: ReactNode
  header: ReactNode
  children: ReactNode
}

export function AppShell({ sidebar, header, children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="mx-auto flex min-h-screen max-w-[1920px]">
        {/* Sidebar — fixed left rail */}
        <aside className="sticky top-0 hidden h-screen w-64 shrink-0 border-r border-border/50 lg:block">
          {sidebar}
        </aside>

        {/* Main area */}
        <div className="flex min-h-screen min-w-0 flex-1 flex-col">
          {/* Mobile header */}
          <div className="flex items-center gap-3 border-b border-border/50 px-5 py-3.5 lg:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="ghost" size="icon">
                  <Menu className="size-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-64 p-0">
                <SheetHeader className="border-b border-border/50 px-5 py-3.5">
                  <SheetTitle className="text-sm">Quent</SheetTitle>
                </SheetHeader>
                <div className="h-[calc(100%-48px)]">{sidebar}</div>
              </SheetContent>
            </Sheet>
            <div className="min-w-0 flex-1">{header}</div>
          </div>

          {/* Desktop header */}
          <div className="hidden border-b border-border/50 lg:block">{header}</div>

          {/* Content */}
          <main className="min-w-0 flex-1 p-6">{children}</main>
        </div>
      </div>
    </div>
  )
}
