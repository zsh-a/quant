const STATUS_LABELS: Record<string, string> = {
  running: "运行中",
  completed: "已完成",
  failed: "失败",
  pending: "等待中",
  success: "成功",
  failure: "失败",
}

const MODE_LABELS: Record<string, string> = {
  live: "实盘",
  backtest: "回测",
  simulation: "模拟",
}

const SOURCE_LABELS: Record<string, string> = {
  manual: "手动",
  automation: "自动",
}

export function formatStatusLabel(value?: string | null) {
  const normalized = value?.toLowerCase() ?? ""
  return STATUS_LABELS[normalized] ?? value ?? "未知"
}

export function formatModeLabel(value?: string | null) {
  const normalized = value?.toLowerCase() ?? ""
  return MODE_LABELS[normalized] ?? value ?? "未知"
}

export function formatSourceLabel(value?: string | null) {
  const normalized = value?.toLowerCase() ?? "manual"
  return SOURCE_LABELS[normalized] ?? value ?? "未知"
}
