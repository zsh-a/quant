import { Badge } from "../ui/badge"
import { formatModeLabel, formatStatusLabel } from "../../utils/display"

type StatusKind =
  | "live"
  | "backtest"
  | "simulation"
  | "running"
  | "completed"
  | "failed"
  | "failed_timeout"
  | "pending"
  | "success"
  | "partial_success"
  | "idle"
  | "default"

const variantMap: Record<StatusKind, "default" | "success" | "warning" | "danger" | "info"> = {
  live: "success",
  backtest: "info",
  simulation: "warning",
  running: "warning",
  completed: "success",
  failed: "danger",
  failed_timeout: "danger",
  pending: "warning",
  success: "success",
  partial_success: "info",
  idle: "default",
  default: "default",
}

export function StatusBadge({ value }: { value?: string | null }) {
  const normalized = value?.toLowerCase() ?? "default"
  const kind = (normalized in variantMap ? normalized : "default") as StatusKind

  const label =
    kind === "live" || kind === "backtest" || kind === "simulation"
      ? formatModeLabel(value)
      : formatStatusLabel(value)

  return <Badge variant={variantMap[kind]}>{label}</Badge>
}
