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
      <div className="mx-auto flex min-h-screen max-w-[1800px] gap-6 px-4 py-4 sm:px-6 lg:px-8">
        <aside className="sticky top-4 hidden h-[calc(100vh-2rem)] w-80 shrink-0 lg:block">
          {sidebar}
        </aside>
        <div className="flex min-h-[calc(100vh-2rem)] min-w-0 flex-1 flex-col gap-6">
          <div className="flex items-center justify-between gap-3 lg:hidden">
            <Sheet>
              <SheetTrigger asChild>
                <Button variant="outline" size="icon">
                  <Menu className="size-4" />
                </Button>
              </SheetTrigger>
              <SheetContent side="left" className="w-[88vw] max-w-sm p-0">
                <SheetHeader className="border-b border-border px-6 py-5">
                  <SheetTitle>Quent Console</SheetTitle>
                </SheetHeader>
                <div className="h-[calc(100%-65px)] p-4">{sidebar}</div>
              </SheetContent>
            </Sheet>
            <div className="min-w-0 flex-1">{header}</div>
          </div>
          <div className="hidden lg:block">{header}</div>
          <main className="min-w-0 flex-1">{children}</main>
        </div>
      </div>
    </div>
  )
}
