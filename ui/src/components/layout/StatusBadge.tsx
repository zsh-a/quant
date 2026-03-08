import { Badge } from "../ui/badge"

type StatusKind = "live" | "backtest" | "simulation" | "running" | "completed" | "failed" | "default"

const variantMap: Record<StatusKind, "default" | "success" | "warning" | "danger" | "info"> = {
  live: "success",
  backtest: "info",
  simulation: "warning",
  running: "warning",
  completed: "success",
  failed: "danger",
  default: "default",
}

export function StatusBadge({ value }: { value?: string | null }) {
  const normalized = value?.toLowerCase() ?? "default"
  const kind = (normalized in variantMap ? normalized : "default") as StatusKind

  return <Badge variant={variantMap[kind]}>{value ?? "unknown"}</Badge>
}
