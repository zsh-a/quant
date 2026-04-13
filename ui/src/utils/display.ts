const STATUS_LABELS: Record<string, string> = {
  running: "Running",
  completed: "Completed",
  failed: "Failed",
  failed_timeout: "Timed Out",
  pending: "Pending",
  success: "Success",
  failure: "Failed",
  partial_success: "Partial Success",
  idle: "Idle",
}

const MODE_LABELS: Record<string, string> = {
  live: "Live",
  backtest: "Backtest",
  simulation: "Simulation",
}

const SOURCE_LABELS: Record<string, string> = {
  manual: "Manual",
  automation: "Automation",
  schedule: "Scheduled",
}

export function formatStatusLabel(value?: string | null) {
  const normalized = value?.toLowerCase() ?? ""
  return STATUS_LABELS[normalized] ?? value ?? "Unknown"
}

export function formatModeLabel(value?: string | null) {
  const normalized = value?.toLowerCase() ?? ""
  return MODE_LABELS[normalized] ?? value ?? "Unknown"
}

export function formatSourceLabel(value?: string | null) {
  const normalized = value?.toLowerCase() ?? "manual"
  return SOURCE_LABELS[normalized] ?? value ?? "Unknown"
}
