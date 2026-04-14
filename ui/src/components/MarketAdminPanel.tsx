import { useEffect, useMemo, useState } from 'react';
import { AlertTriangle, Database, Play, RefreshCw, TableProperties } from 'lucide-react';

import type {
  DataUpdateRun,
  MarketDbOverview,
  MarketTableSummary,
  MarketUpdateCapabilities,
} from '../types';
import { apiFetch } from '../utils/api';
import { formatSourceLabel, formatStatusLabel } from '../utils/display';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { MetricCard } from './layout/MetricCard';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';

function formatDateTime(value?: string | null) {
  if (!value) return 'Not recorded';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString('en-US', { hour12: false });
}

function formatCount(value?: number | null) {
  if (value === null || value === undefined) return 'N/A';
  return value.toLocaleString('en-US');
}

function formatProgress(value?: number | null) {
  if (value === null || value === undefined || Number.isNaN(value)) return 'N/A';
  return `${value.toFixed(1)}%`;
}

function durationLabel(run?: DataUpdateRun | null) {
  const startedAt = run?.started_at ? new Date(run.started_at).getTime() : null;
  const completedAt = run?.completed_at ? new Date(run.completed_at).getTime() : null;
  if (!startedAt || !completedAt || Number.isNaN(startedAt) || Number.isNaN(completedAt)) {
    return 'Not completed';
  }
  const seconds = Math.max(Math.round((completedAt - startedAt) / 1000), 0);
  return `${seconds}s`;
}

function summarizeTable(summary?: MarketTableSummary) {
  if (!summary) return 'No data';
  if (summary.status === 'error') return summary.error || 'Query failed';
  const distinctPart =
    summary.distinct_count !== null && summary.distinct_count !== undefined
      ? `${formatCount(summary.distinct_count)} ${summary.distinct_label || 'items'}`
      : 'No distinct stats';
  return `${formatCount(summary.row_count)} rows · ${distinctPart}`;
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
        apiFetch(`/market-admin/overview`),
        apiFetch(`/market-admin/update-capabilities`),
        apiFetch(`/market-admin/update-runs?limit=12`),
      ]);

      if (!overviewResp.ok || !capabilitiesResp.ok || !historyResp.ok) {
        throw new Error('Failed to load market database dashboard');
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
        setError(err instanceof Error ? err.message : 'Loading failed');
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
  const runningProgress =
    typeof overview?.running_update?.details?.progress === 'number'
      ? overview?.running_update?.details?.progress
      : null;
  const runningCurrentStep = overview?.running_update?.details?.current_step || null;
  const selectedProgress =
    typeof selectedRun?.details?.progress === 'number' ? selectedRun?.details?.progress : null;
  const selectedCurrentStep = selectedRun?.details?.current_step || null;

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
      const resp = await apiFetch(`/market-admin/update-runs`, {
        method: 'POST',
        body: JSON.stringify({
          selected_steps: selectedSteps,
          share_start_date: shareStartDate || null,
        }),
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      const data = await resp.json();
      setMessage(`Update task submitted: ${data.update_run_id}`);
      setSelectedRunId(data.update_run_id);
      await fetchAll(true);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Submission failed');
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
        title="Market Database"
        description="View ClickHouse market database coverage, recent update batches, and manually trigger data updates."
        actions={
          <Button variant="outline" onClick={() => fetchAll(true)} disabled={refreshing || loading}>
            <RefreshCw className={refreshing ? 'animate-spin' : ''} />
            Refresh Panel
          </Button>
        }
      />

      <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-4">
        <MetricCard
          label="Latest Market Date"
          value={overview?.latest_market_date || 'Not found'}
          hint={latestTable ? summarizeTable(latestTable) : 'Loading'}
        />
        <MetricCard
          label="Data Lag"
          value={
            overview?.data_lag_days === null || overview?.data_lag_days === undefined
              ? 'N/A'
              : `${overview.data_lag_days} days`
          }
          hint={`Reference symbol: ${overview?.reference_symbol || 'N/A'}`}
        />
        <MetricCard
          label="Stock Coverage"
          value={formatCount(overview?.stock_coverage?.tracked_stock_codes)}
          hint={`ETF catalog: ${formatCount(overview?.stock_coverage?.tracked_etf_codes)}`}
        />
        <MetricCard
          label="Latest Update"
          value={lastRun ? <StatusBadge value={lastRun.status} /> : 'No records'}
          hint={lastRun ? `${formatDateTime(lastRun.completed_at || lastRun.started_at || lastRun.created_at)} · ${durationLabel(lastRun)}` : 'No data updates have been run yet'}
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
          title="Database Overview"
          description="Summary of core table coverage and latest dates. Single-table query errors will not block the entire dashboard."
        >
          <div className="grid gap-4 md:grid-cols-2">
            {Object.entries(overview?.tables || {}).map(([key, summary]) => (
              <div key={key} className="rounded-3xl border border-border/70 bg-secondary/35 p-4">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <div className="text-sm font-semibold text-foreground">{summary.table}</div>
                    <div className="mt-1 text-xs text-muted-foreground">{summary.status === 'error' ? 'Status error' : 'Aggregation OK'}</div>
                  </div>
                  <StatusBadge value={summary.status === 'error' ? 'failed' : 'success'} />
                </div>
                <div className="mt-4 space-y-2 text-sm text-muted-foreground">
                  <div>Rows: {formatCount(summary.row_count)}</div>
                  <div>
                    {summary.distinct_label || 'Distinct'}:
                    {summary.distinct_count === null || summary.distinct_count === undefined ? ' N/A' : ` ${formatCount(summary.distinct_count)}`}
                  </div>
                  <div>Earliest date: {summary.earliest_date || 'N/A'}</div>
                  <div>Latest date: {summary.latest_date || 'N/A'}</div>
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
          title="Manual Update Console"
          description="Select update steps to execute. Submission automatically triggers linked simulation jobs on the backend."
          action={
            overview?.running_update ? (
              <div className="flex items-center gap-2 rounded-full bg-primary/10 px-3 py-1 text-xs font-semibold text-primary">
                <Play className="size-3.5" />
                Running
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
              Current default is {capabilities?.share_start_date_default || 'N/A'}. This start date is used when the "Share Info" step is selected.
            </div>
          </div>

          <div className="grid gap-3 rounded-3xl border border-border/70 bg-card/60 p-4">
            <div className="flex items-center justify-between gap-3">
              <div>
                <div className="text-sm font-semibold text-foreground">Current Running Batch</div>
                <div className="text-sm text-muted-foreground">
                  {overview?.running_update ? formatDateTime(overview.running_update.started_at) : 'No data update currently running'}
                </div>
              </div>
              <StatusBadge value={overview?.running_update?.status || 'completed'} />
            </div>
            {overview?.running_update ? (
              <div className="space-y-2 text-sm text-muted-foreground">
                <div>Batch ID: {overview.running_update.update_run_id}</div>
                <div>Trigger source: {formatSourceLabel(overview.running_update.trigger_source)}</div>
                <div>Last heartbeat: {formatDateTime(overview.running_update.last_heartbeat_at || overview.running_update.started_at)}</div>
                {typeof runningProgress === 'number' ? (
                  <div className="space-y-2 rounded-2xl border border-border/70 bg-background/70 p-3">
                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                      <span>Overall progress</span>
                      <span>{formatProgress(runningProgress)}</span>
                    </div>
                    <Progress value={runningProgress} />
                  </div>
                ) : null}
                {runningCurrentStep ? (
                  <div className="space-y-2 rounded-2xl border border-border/70 bg-background/70 p-3">
                    <div className="text-xs font-semibold text-foreground">Current step</div>
                    <div className="text-sm text-muted-foreground">
                      {String(runningCurrentStep.label || runningCurrentStep.key || 'unknown')}
                    </div>
                    {typeof runningCurrentStep.progress === 'number' ? (
                      <>
                        <div className="flex items-center justify-between text-xs text-muted-foreground">
                          <span>Step progress</span>
                          <span>{formatProgress(runningCurrentStep.progress)}</span>
                        </div>
                        <Progress value={runningCurrentStep.progress} />
                      </>
                    ) : null}
                    {typeof runningCurrentStep.current === 'number' &&
                    typeof runningCurrentStep.total === 'number' ? (
                      <div className="text-xs text-muted-foreground">
                        Batch {runningCurrentStep.current}/{runningCurrentStep.total}
                      </div>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
          </div>

          <Button onClick={handleRunUpdate} disabled={submitting || selectedSteps.length === 0}>
            <Database className={submitting ? 'animate-pulse' : ''} />
            {submitting ? 'Submitting...' : 'Start Update'}
          </Button>
        </SectionCard>
      </div>

      <div className="grid gap-6 xl:grid-cols-[minmax(0,0.95fr)_minmax(0,1.05fr)]">
        <SectionCard
          title="Recent Update Batches"
          description="View recent execution history. Select a batch on the right to see step results and auto-triggered simulation jobs."
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
                    <div>New data: {item.has_new_data ? 'Yes' : 'No'}</div>
                    <div>Steps: {Array.isArray(item.details?.steps) ? item.details.steps.length : 0}</div>
                    <div>Failed steps: {failedSteps}</div>
                    <div>Triggered simulations: {triggeredCount}</div>
                  </div>
                </button>
              );
            })}
            {history.length === 0 ? (
              <div className="rounded-3xl border border-dashed border-border/70 bg-card/40 p-5 text-sm text-muted-foreground">
                No data update batches yet.
              </div>
            ) : null}
          </div>
        </SectionCard>

        <SectionCard
          title="Batch Details"
          description="Step-by-step execution results, duration, errors, and simulation jobs triggered by the update."
        >
          {selectedRun ? (
            <div className="space-y-5">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-3xl border border-border/70 bg-secondary/25 p-4">
                  <div className="flex items-center justify-between gap-3">
                    <div className="text-sm font-semibold text-foreground">Batch Status</div>
                    <StatusBadge value={selectedRun.status} />
                  </div>
                  <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                    <div>Batch ID: {selectedRun.update_run_id}</div>
                    <div>Trigger source: {formatSourceLabel(selectedRun.trigger_source)}</div>
                    <div>Start time: {formatDateTime(selectedRun.started_at)}</div>
                    <div>Last heartbeat: {formatDateTime(selectedRun.last_heartbeat_at || selectedRun.started_at)}</div>
                    <div>Completion time: {formatDateTime(selectedRun.completed_at)}</div>
                    <div>New data found: {selectedRun.has_new_data ? 'Yes' : 'No'}</div>
                  </div>
                  {typeof selectedProgress === 'number' ? (
                    <div className="mt-4 space-y-2">
                      <div className="flex items-center justify-between text-xs text-muted-foreground">
                        <span>Overall progress</span>
                        <span>{formatProgress(selectedProgress)}</span>
                      </div>
                      <Progress value={selectedProgress} />
                    </div>
                  ) : null}
                  {selectedCurrentStep ? (
                    <div className="mt-3 space-y-2 rounded-2xl border border-border/70 bg-background/70 p-3 text-xs text-muted-foreground">
                      <div className="font-semibold text-foreground">Current step</div>
                      <div>{String(selectedCurrentStep.label || selectedCurrentStep.key || 'unknown')}</div>
                      {typeof selectedCurrentStep.progress === 'number' ? (
                        <>
                          <div className="flex items-center justify-between">
                            <span>Step progress</span>
                            <span>{formatProgress(selectedCurrentStep.progress)}</span>
                          </div>
                          <Progress value={selectedCurrentStep.progress} />
                        </>
                      ) : null}
                      {typeof selectedCurrentStep.current === 'number' &&
                      typeof selectedCurrentStep.total === 'number' ? (
                        <div>
                          Batch {selectedCurrentStep.current}/{selectedCurrentStep.total}
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                </div>

                <div className="rounded-3xl border border-border/70 bg-secondary/25 p-4">
                  <div className="text-sm font-semibold text-foreground">Before / After Comparison</div>
                  <div className="mt-3 space-y-2 text-sm text-muted-foreground">
                    <div>Before update: {selectedRun.details?.before_latest_date || 'N/A'}</div>
                    <div>After update: {selectedRun.details?.after_latest_date || 'N/A'}</div>
                    <div>Reference symbol: {selectedRun.details?.reference_symbol || overview?.reference_symbol || 'N/A'}</div>
                    <div>Auto-triggered jobs: {Array.isArray(selectedRun.details?.triggered_jobs) ? selectedRun.details.triggered_jobs.length : 0}</div>
                  </div>
                </div>
              </div>

              <div className="space-y-3">
                <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                  <TableProperties className="size-4" />
                  Step Results
                </div>
                {Array.isArray(selectedRun.details?.steps) && selectedRun.details.steps.length > 0 ? (
                  selectedRun.details.steps.map((step: Record<string, unknown>, index: number) => (
                    <div key={`${selectedRun.update_run_id}-${index}`} className="rounded-3xl border border-border/70 bg-card/60 p-4">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-sm font-semibold text-foreground">
                            {String(step.label || step.step || 'unknown')}
                          </div>
                          <div className="mt-1 text-xs text-muted-foreground">
                            {formatStatusLabel(String(step.status || 'unknown'))}
                            {' · '}
                            {typeof step.duration_seconds === 'number' ? `${step.duration_seconds}s` : 'No duration'}
                          </div>
                        </div>
                        <StatusBadge value={String(step.status || 'unknown')} />
                      </div>
                      {typeof step.progress === 'number' ? (
                        <div className="mt-3 space-y-2">
                          <div className="flex items-center justify-between text-xs text-muted-foreground">
                            <span>Progress</span>
                            <span>{formatProgress(Number(step.progress))}</span>
                          </div>
                          <Progress value={Number(step.progress)} />
                          {typeof step.current === 'number' && typeof step.total === 'number' ? (
                            <div className="text-xs text-muted-foreground">
                              Batch {String(step.current)}/{String(step.total)}
                            </div>
                          ) : null}
                        </div>
                      ) : null}
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
                    No step details for this batch yet.
                  </div>
                )}
              </div>

              {Array.isArray(selectedRun.details?.triggered_jobs) && selectedRun.details.triggered_jobs.length > 0 ? (
                <div className="space-y-3">
                  <div className="flex items-center gap-2 text-sm font-semibold text-foreground">
                    <Play className="size-4" />
                    Auto-triggered Simulation Jobs
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    {selectedRun.details.triggered_jobs.map((job: Record<string, unknown>) => (
                      <div key={String(job.task_id || job.job_id)} className="rounded-3xl border border-border/70 bg-secondary/25 p-4">
                        <div className="text-sm font-semibold text-foreground">{String(job.name || job.job_id || 'unknown')}</div>
                        <div className="mt-2 text-sm text-muted-foreground">Task ID: {String(job.task_id || 'N/A')}</div>
                      </div>
                    ))}
                  </div>
                </div>
              ) : null}

              {Array.isArray(selectedRun.details?.errors) && selectedRun.details.errors.length > 0 ? (
                <div className="rounded-3xl border border-destructive/25 bg-destructive/7 p-4">
                  <div className="flex items-center gap-2 text-sm font-semibold text-destructive">
                    <AlertTriangle className="size-4" />
                    Error Summary
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
              Select a batch on the left to view update steps, errors, and auto-triggered simulation jobs.
            </div>
          )}
        </SectionCard>
      </div>

      {loading ? (
        <div className="rounded-3xl border border-dashed border-border/70 bg-card/40 p-6 text-sm text-muted-foreground">
          Loading market database panel...
        </div>
      ) : null}

      {financialTable ? (
        <SectionCard
          title="Supplementary Metrics"
          description="Quickly check if market and financial report data are updated in sync."
        >
          <div className="grid gap-4 md:grid-cols-3">
            <MetricCard
              label="Latest Financial Report"
              value={financialTable.latest_date || 'N/A'}
              hint={summarizeTable(financialTable)}
            />
            <MetricCard
              label="Latest Run Batch"
              value={overview?.running_update ? 'Running' : lastRun ? formatStatusLabel(lastRun.status) : 'No records'}
              hint={overview?.running_update ? overview.running_update.update_run_id : lastRun?.update_run_id || 'N/A'}
            />
            <MetricCard
              label="Panel Refresh Time"
              value={formatDateTime(overview?.generated_at)}
              hint="Auto-refreshes every 5 seconds when a batch is running"
            />
          </div>
        </SectionCard>
      ) : null}
    </div>
  );
}
