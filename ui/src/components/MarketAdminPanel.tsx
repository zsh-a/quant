import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Database, Play, RefreshCw, TableProperties } from 'lucide-react';

import type {
  DataUpdateRun,
  MarketDbOverview,
  MarketTableSummary,
  MarketUpdateCapabilities,
} from '../types';
import { API_BASE } from '../utils/api';
import { formatSourceLabel, formatStatusLabel } from '../utils/display';
import { Button } from './ui/button';
import { MetricCard } from './layout/MetricCard';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';

function formatDateTime(value?: string | null) {
  if (!value) return '未记录';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('zh-CN', { hour12: false });
}

function formatCount(value?: number | null) {
  if (value === null || value === undefined) return 'N/A';
  return value.toLocaleString('zh-CN');
}

function durationLabel(run?: DataUpdateRun | null) {
  const startedAt = run?.started_at ? new Date(run.started_at).getTime() : null;
  const completedAt = run?.completed_at ? new Date(run.completed_at).getTime() : null;
  if (!startedAt || !completedAt || Number.isNaN(startedAt) || Number.isNaN(completedAt)) {
    return '未完成';
  }
  const seconds = Math.max(Math.round((completedAt - startedAt) / 1000), 0);
  return `${seconds}s`;
}

function summarizeTable(summary?: MarketTableSummary) {
  if (!summary) return '暂无数据';
  if (summary.status === 'error') return summary.error || '查询失败';
  const distinctPart =
    summary.distinct_count !== null && summary.distinct_count !== undefined
      ? `${formatCount(summary.distinct_count)} ${summary.distinct_label || '项'}`
      : '无去重统计';
  return `${formatCount(summary.row_count)} 行 · ${distinctPart}`;
}

export default function MarketAdminPanel() {
  const [overview, setOverview] = useState<MarketDbOverview | null>(null);
  const [capabilities, setCapabilities] = useState<MarketUpdateCapabilities | null>(null);
  const [history, setHistory] = useState<DataUpdateRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedSteps, setSelectedSteps] = useState<string[]>([]);
  const [shareStartDate, setShareStartDate] = useState('');
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);

  const fetchAll = async (background = false) => {
    if (!background) {
      setError(null);
    }
    const setter = background ? setRefreshing : setLoading;
    setter(true);
    try {
      const [overviewResp, capabilitiesResp, historyResp] = await Promise.all([
        fetch(`${API_BASE}/market-admin/overview`),
        fetch(`${API_BASE}/market-admin/update-capabilities`),
        fetch(`${API_BASE}/market-admin/update-runs?limit=12`),
      ]);

      if (!overviewResp.ok || !capabilitiesResp.ok || !historyResp.ok) {
        throw new Error('加载行情数据库看板失败');
      }

      const [overviewData, capabilitiesData, historyData] = await Promise.all([
        overviewResp.json(),
        capabilitiesResp.json(),
        historyResp.json(),
      ]);

      setOverview(overviewData);
      setCapabilities(capabilitiesData);
      setHistory(historyData);
      setSelectedRunId((current) => {
        if (current && historyData.some((item: DataUpdateRun) => item.update_run_id === current)) {
          return current;
        }
        if (overviewData.running_update?.update_run_id) {
          return overviewData.running_update.update_run_id;
        }
        return historyData[0]?.update_run_id ?? null;
      });
      setSelectedSteps((current) =>
        current.length > 0 ? current : capabilitiesData.default_selected_steps || [],
      );
      setShareStartDate((current) => current || capabilitiesData.share_start_date_default || '');
    } catch (err) {
      if (!background) {
        setError(err instanceof Error ? err.message : '加载失败');
      }
    } finally {
      setter(false);
    }
  };

  useEffect(() => {
    fetchAll();
  }, []);

  useEffect(() => {
    if (!overview?.running_update) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      fetchAll(true);
    }, 5000);
    return () => window.clearInterval(timer);
  }, [overview?.running_update?.update_run_id]);

  const selectedRun = useMemo(
    () => history.find((item) => item.update_run_id === selectedRunId) || overview?.running_update || null,
    [history, overview?.running_update, selectedRunId],
  );

  const handleStepToggle = (stepKey: string) => {
    setSelectedSteps((current) =>
      current.includes(stepKey)
        ? current.filter((item) => item !== stepKey)
        : [...current, stepKey],
    );
  };

  const handleRunUpdate = async () => {
    setSubmitting(true);
    setError(null);
    setMessage(null);
    try {
      const resp = await fetch(`${API_BASE}/market-admin/update-runs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          selected_steps: selectedSteps,
          share_start_date: shareStartDate || null,
        }),
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      const data = await resp.json();
      setMessage(`更新任务已提交：${data.update_run_id}`);
      setSelectedRunId(data.update_run_id);
      await fetchAll(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : '提交失败');
    } finally {
      setSubmitting(false);
    }
  };

  const lastRun = overview?.last_update_run ?? null;
  const latestTable = overview?.tables?.stock_daily;
  const financialTable = overview?.tables?.finicial_report;

  return (
    <div className="space-y-6">
      <PageHeader
        eyebrow="Market Database Control"
        title="行情数据库"
        description="集中查看 ClickHouse 行情库的覆盖情况、最近更新批次，并手动触发基础数据更新。"
        actions={
          <Button variant="outline" onClick={() => fetchAll(true)} disabled={refreshing || loading}>
            <RefreshCw className={refreshing ? 'animate-spin' : ''} />
            刷新面板
          </Button>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="最新行情日期"
          value={overview?.latest_market_date || '未发现'}
          hint={latestTable ? summarizeTable(latestTable) : '等待加载'}
        />
        <MetricCard
          label="数据滞后"
          value={
            overview?.data_lag_days === null || overview?.data_lag_days === undefined
              ? 'N/A'
              : `${overview.data_lag_days} 天`
          }
          hint={`参考标的 ${overview?.reference_symbol || 'N/A'}`}
        />
        <MetricCard
          label="股票覆盖"
          value={formatCount(overview?.stock_coverage?.tracked_stock_codes)}
          hint={`ETF 目录 ${formatCount(overview?.stock_coverage?.tracked_etf_codes)} 个`}
        />
        <MetricCard
          label="最近更新"
          value={lastRun ? <StatusBadge value={lastRun.status} /> : '无记录'}
          hint={lastRun ? `${formatDateTime(lastRun.completed_at || lastRun.started_at || lastRun.created_at)} · ${durationLabel(lastRun)}` : '尚未执行过数据更新'}
        />
      </div>

      {(error || message) && (
        <div className="grid gap-3 lg:grid-cols-2">
          {error ? (
            <div className="rounded-2xl border border-destructive/30 bg-destructive/8 px-4 py-3 text-sm text-destructive">
              {error}
            </div>
          ) : null}
          {message ? (
            <div className="rounded-2xl border border-primary/25 bg-primary/10 px-4 py-3 text-sm text-primary">
              {message}
            </div>
          ) : null}
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[minmax(0,1.1fr)_minmax(380px,0.9fr)]">
        <SectionCard
          title="数据库概览"
          description="汇总核心表的覆盖范围与最新日期。单表查询异常不会阻断整个看板。"
        >
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(overview?.tables || {}).map(([key, summary]) => (
              <div key={key} className="rounded-3xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-foreground">{summary.table}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{summary.status === 'error' ? '状态异常' : '聚合正常'}</div>
                  </div>
                  <StatusBadge value={summary.status === 'error' ? 'failed' : 'success'} />
                </div>
                <div className="mt-4 space-y-2 text-sm text-muted-foreground">
                  <div>行数：{formatCount(summary.row_count)}</div>
                  <div>
                    {summary.distinct_label || '去重项'}：
                    {summary.distinct_count === null || summary.distinct_count === undefined ? ' N/A' : ` ${formatCount(summary.distinct_count)}`}
                  </div>
                  <div>最早日期：{summary.earliest_date || 'N/A'}</div>
                  <div>最新日期：{summary.latest_date || 'N/A'}</div>
                  {summary.error ? (
                    <div className="rounded-2xl border border-destructive/25 bg-destructive/7 px-3 py-2 text-destructive">
                      {summary.error}
                    </div>
                  ) : null}
                </div>
              </div>
            ))}
          </div>
        </SectionCard>

        <SectionCard
          title="手动更新控制台"
          description="选择要执行的更新步骤，提交后自动触发后端已有的模拟任务联动逻辑。"
          action={
            overview?.running_update ? (
              <div className="flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                <Play className="size-3.5" />
                运行中
              </div>
            ) : null
          }
        >
          <div className="grid gap-3">
            {(capabilities?.steps || []).map((step) => {
              const selected = selectedSteps.includes(step.key);
              return (
                <button
                  key={step.key}
                  type="button"
                  onClick={() => handleStepToggle(step.key)}
                  className={`rounded-2xl border px-4 py-3 text-left transition ${
                    selected
                      ? 'border-primary/40 bg-primary/10'
                      : 'border-border/70 bg-secondary/25 hover:border-primary/25 hover:bg-accent/30'
                  }`}
                >
                  <div className="flex items-center justify-between gap-3">
                    <div className="font-medium text-foreground">{step.label}</div>
                    <div className={`size-2.5 rounded-full ${selected ? 'bg-primary' : 'bg-muted-foreground/40'}`} />
                  </div>
                  <div className="mt-1 text-sm leading-6 text-muted-foreground">{step.description}</div>
                </button>
              );
            })}
          </div>

          <div className="grid gap-3 rounded-3xl border border-border/70 bg-secondary/30 p-4">
            <label className="text-xs font-semibold uppercase tracking-[0.16em] text-muted-foreground">
              Share Start Date
            </label>
            <input
              className="h-11 rounded-2xl border border-border/70 bg-background/70 px-4 text-sm text-foreground outline-none transition focus:border-primary/40"
              value={shareStartDate}
              onChange={(event) => setShareStartDate(event.target.value)}
              placeholder="20250101"
            />
            <div className="text-sm text-muted-foreground">
              当前默认值为 {capabilities?.share_start_date_default || 'N/A'}。当勾选“股本信息”步骤时会使用此起始日期。
            </div>
          </div>

          <div className="grid gap-3 rounded-3xl border border-border/70 bg-card/60 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-foreground">当前运行批次</div>
                <div className="text-sm text-muted-foreground">
                  {overview?.running_update ? formatDateTime(overview.running_update.started_at) : '当前没有运行中的数据更新'}
                </div>
              </div>
              <StatusBadge value={overview?.running_update?.status || 'completed'} />
            </div>
            {overview?.running_update ? (
              <div className="space-y-2 text-sm text-muted-foreground">
                <div>批次 ID：{overview.running_update.update_run_id}</div>
                <div>触发来源：{formatSourceLabel(overview.running_update.trigger_source)}</div>
              </div>
            ) : null}
          </div>

          <Button onClick={handleRunUpdate} disabled={submitting || selectedSteps.length === 0}>
            <Database className={submitting ? 'animate-pulse' : ''} />
            {submitting ? '提交中...' : '开始更新'}
          </Button>
        </SectionCard>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <SectionCard
          title="最近更新批次"
          description="查看最近执行历史，选择右侧查看步骤结果与自动触发的模拟任务。"
        >
          <div className="space-y-3">
            {history.map((item) => {
              const selected = item.update_run_id === selectedRunId;
              const failedSteps = Array.isArray(item.details?.steps)
                ? item.details.steps.filter((step: { status?: string }) => step.status !== 'success').length
                : 0;
              const triggeredCount = Array.isArray(item.details?.triggered_jobs)
                ? item.details.triggered_jobs.length
                : 0;

              return (
                <button
                  key={item.update_run_id}
                  type="button"
                  onClick={() => setSelectedRunId(item.update_run_id)}
                  className={`w-full rounded-3xl border px-4 py-4 text-left transition ${
                    selected
                      ? 'border-primary/35 bg-primary/10'
                      : 'border-border/70 bg-secondary/25 hover:border-primary/25 hover:bg-accent/30'
                  }`}
                >
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-sm font-semibold text-foreground">{formatStatusLabel(item.status)}</div>
                      <div className="mt-1 text-xs text-muted-foreground">
                        {formatSourceLabel(item.trigger_source)} · {formatDateTime(item.completed_at || item.started_at || item.created_at)}
                      </div>
                    </div>
                    <StatusBadge value={item.status} />
                  </div>
                  <div className="mt-4 grid gap-1 text-sm text-muted-foreground">
                    <div>有新数据：{item.has_new_data ? '是' : '否'}</div>
                    <div>步骤数：{Array.isArray(item.details?.steps) ? item.details.steps.length : 0}</div>
                    <div>失败步骤：{failedSteps}</div>
                    <div>触发模拟：{triggeredCount}</div>
                  </div>
                </button>
              );
            })}
            {history.length === 0 ? (
              <div className="rounded-3xl border border-dashed border-border/70 bg-card/40 p-5 text-sm text-muted-foreground">
                暂无数据更新批次。
              </div>
            ) : null}
          </div>
        </SectionCard>

        <SectionCard
          title="批次明细"
          description="逐步查看执行结果、耗时、错误和由更新触发的模拟任务。"
        >
          {selectedRun ? (
            <div className="space-y-5">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-3xl border border-border/70 bg-secondary/25 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-foreground">批次状态</div>
                    <StatusBadge value={selectedRun.status} />
                  </div>
                  <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                    <div>批次 ID：{selectedRun.update_run_id}</div>
                    <div>触发来源：{formatSourceLabel(selectedRun.trigger_source)}</div>
                    <div>开始时间：{formatDateTime(selectedRun.started_at)}</div>
                    <div>完成时间：{formatDateTime(selectedRun.completed_at)}</div>
                    <div>发现新数据：{selectedRun.has_new_data ? '是' : '否'}</div>
                  </div>
                </div>

                <div className="rounded-3xl border border-border/70 bg-secondary/25 p-4">
                  <div className="text-sm font-semibold text-foreground">更新前后对比</div>
                  <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                    <div>更新前：{selectedRun.details?.before_latest_date || 'N/A'}</div>
                    <div>更新后：{selectedRun.details?.after_latest_date || 'N/A'}</div>
                    <div>参考标的：{selectedRun.details?.reference_symbol || overview?.reference_symbol || 'N/A'}</div>
                    <div>自动触发任务：{Array.isArray(selectedRun.details?.triggered_jobs) ? selectedRun.details.triggered_jobs.length : 0}</div>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  <TableProperties className="size-4" />
                  步骤结果
                </div>
                {Array.isArray(selectedRun.details?.steps) && selectedRun.details.steps.length > 0 ? (
                  selectedRun.details.steps.map((step: Record<string, unknown>, index: number) => (
                    <div key={`${selectedRun.update_run_id}-${index}`} className="rounded-3xl border border-border/70 bg-card/60 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-semibold text-foreground">{String(step.step || 'unknown')}</div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            {formatStatusLabel(String(step.status || 'unknown'))}
                            {' · '}
                            {typeof step.duration_seconds === 'number' ? `${step.duration_seconds}s` : '无耗时'}
                          </div>
                        </div>
                        <StatusBadge value={String(step.status || 'unknown')} />
                      </div>
                      {step.detail ? (
                        <pre className="mt-4 overflow-x-auto rounded-2xl bg-background/70 p-3 text-xs leading-6 text-muted-foreground">
                          {JSON.stringify(step.detail, null, 2)}
                        </pre>
                      ) : null}
                      {step.error ? (
                        <div className="mt-4 rounded-2xl border border-destructive/25 bg-destructive/7 px-3 py-2 text-sm text-destructive">
                          {String(step.error)}
                        </div>
                      ) : null}
                    </div>
                  ))
                ) : (
                  <div className="rounded-3xl border border-dashed border-border/70 bg-card/40 p-5 text-sm text-muted-foreground">
                    当前批次还没有步骤明细。
                  </div>
                )}
              </div>

              {Array.isArray(selectedRun.details?.triggered_jobs) && selectedRun.details.triggered_jobs.length > 0 ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Play className="size-4" />
                    自动触发的模拟任务
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {selectedRun.details.triggered_jobs.map((job: Record<string, unknown>) => (
                      <div key={String(job.task_id || job.job_id)} className="rounded-3xl border border-border/70 bg-secondary/25 p-4">
                        <div className="text-sm font-semibold text-foreground">{String(job.name || job.job_id || 'unknown')}</div>
                        <div className="mt-2 text-sm text-muted-foreground">任务 ID：{String(job.task_id || 'N/A')}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {Array.isArray(selectedRun.details?.errors) && selectedRun.details.errors.length > 0 ? (
                <div className="rounded-3xl border border-destructive/25 bg-destructive/7 p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
                    <AlertTriangle className="size-4" />
                    错误摘要
                  </div>
                  <div className="mt-3 space-y-2 text-sm text-destructive">
                    {selectedRun.details.errors.map((item: Record<string, unknown>, index: number) => (
                      <div key={`${selectedRun.update_run_id}-error-${index}`}>
                        {String(item.step || 'unknown')}：{String(item.error || 'unknown error')}
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}
            </div>
          ) : (
            <div className="rounded-3xl border border-dashed border-border/70 bg-card/40 p-6 text-sm text-muted-foreground">
              选择左侧批次后在这里查看更新步骤、错误和自动触发的模拟任务。
            </div>
          )}
        </SectionCard>
      </div>

      {loading ? (
        <div className="rounded-3xl border border-dashed border-border/70 bg-card/40 p-6 text-sm text-muted-foreground">
          正在加载行情数据库面板...
        </div>
      ) : null}

      {financialTable ? (
        <SectionCard
          title="补充指标"
          description="帮助快速判断行情与财报数据是否同步更新。"
        >
          <div className="grid gap-4 md:grid-cols-3">
            <MetricCard
              label="财报最新披露"
              value={financialTable.latest_date || 'N/A'}
              hint={summarizeTable(financialTable)}
            />
            <MetricCard
              label="最近运行批次"
              value={overview?.running_update ? '运行中' : lastRun ? formatStatusLabel(lastRun.status) : '无记录'}
              hint={overview?.running_update ? overview.running_update.update_run_id : lastRun?.update_run_id || 'N/A'}
            />
            <MetricCard
              label="面板刷新时间"
              value={formatDateTime(overview?.generated_at)}
              hint="运行中批次存在时自动每 5 秒刷新一次"
            />
          </div>
        </SectionCard>
      ) : null}
    </div>
  );
}
