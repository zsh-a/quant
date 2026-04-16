/**
 * Automation Jobs — job 管理 + 运行触发，不含策略配置（已统一到 NewSessionForm）。
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import { toast } from 'sonner';
import type { SimulationJob, SimulationRun } from '../types';
import { apiFetch } from '../utils/api';
import { formatStatusLabel } from '../utils/display';
import { Button } from './ui/button';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';

interface SimulationPanelProps {
  onSelectSession: (sessionId: string) => void;
  onOpenMarketAdmin: () => void;
}

export const SimulationPanel: React.FC<SimulationPanelProps> = ({ onSelectSession, onOpenMarketAdmin }) => {
  const [jobs, setJobs] = useState<SimulationJob[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [runningJobKey, setRunningJobKey] = useState<string | null>(null);
  const [runTriggering, setRunTriggering] = useState(false);
  const [dataUpdating, setDataUpdating] = useState(false);
  const [feedback, setFeedback] = useState<{ type: 'ok' | 'err'; text: string } | null>(null);
  const pollRef = useRef(false);
  const jobsAbort = useRef<AbortController | null>(null);
  const runsAbort = useRef<AbortController | null>(null);

  const currentJob = useMemo(() => jobs.find((j) => j.job_id === selectedJobId) ?? null, [jobs, selectedJobId]);

  // ── Fetch ──

  const fetchJobs = async () => {
    jobsAbort.current?.abort();
    const ctrl = new AbortController();
    jobsAbort.current = ctrl;
    const data = await (await apiFetch('/simulation-jobs', { signal: ctrl.signal })).json();
    setJobs(data);
    setSelectedJobId((prev) => {
      if (data.length === 0) return null;
      if (prev && data.some((j: SimulationJob) => j.job_id === prev)) return prev;
      return data[0].job_id;
    });
  };

  const fetchRuns = async (jobId: string) => {
    runsAbort.current?.abort();
    const ctrl = new AbortController();
    runsAbort.current = ctrl;
    const data = await (await apiFetch(`/simulation-jobs/${jobId}/runs?limit=20`, { signal: ctrl.signal })).json();
    setRuns(data);
  };

  useEffect(() => {
    let active = true;
    const poll = async () => {
      if (!active || pollRef.current) return;
      pollRef.current = true;
      try { await fetchJobs(); if (selectedJobId) await fetchRuns(selectedJobId); }
      finally { pollRef.current = false; }
    };
    poll();
    const id = window.setInterval(poll, 5000);
    return () => { active = false; window.clearInterval(id); jobsAbort.current?.abort(); runsAbort.current?.abort(); };
  }, [selectedJobId]);

  useEffect(() => { if (selectedJobId) fetchRuns(selectedJobId); else setRuns([]); }, [selectedJobId]);

  // ── Actions ──

  const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

  const refreshRuns = async (jobId: string) => {
    for (const ms of [0, 600, 1500, 3000]) { if (ms) await sleep(ms); await fetchJobs(); await fetchRuns(jobId); }
  };

  const handleToggleJob = async (job: SimulationJob) => {
    await apiFetch(`/simulation-jobs/${job.job_id}/${job.enabled ? 'disable' : 'enable'}`, { method: 'POST' });
    await fetchJobs();
  };

  const handleToggleNotify = async (job: SimulationJob) => {
    setFeedback(null);
    const next = !job.notification?.telegram?.enabled;
    const resp = await apiFetch(`/simulation-jobs/${job.job_id}/notification`, {
      method: 'POST',
      body: JSON.stringify({ notification: { telegram: { enabled: next, chat_id: job.notification?.telegram?.chat_id || undefined } } }),
    });
    if (!resp.ok) { setFeedback({ type: 'err', text: await resp.text() }); return; }
    await fetchJobs();
    setFeedback({ type: 'ok', text: next ? 'Notifications enabled' : 'Notifications disabled' });
  };

  const handleRunJob = async (jobId: string, force = false) => {
    setFeedback(null);
    setSelectedJobId(jobId);
    const key = `${force ? 'force' : 'run'}:${jobId}`;
    setRunningJobKey(key);
    try {
      const resp = await apiFetch(`/simulation-jobs/${jobId}/run${force ? '?force=true' : ''}`, { method: 'POST' });
      if (!resp.ok) throw new Error(await resp.text());
      setFeedback({ type: 'ok', text: force ? 'Force re-run submitted' : 'Continue-run submitted' });
      await refreshRuns(jobId);
    } catch (err) {
      setFeedback({ type: 'err', text: err instanceof Error ? err.message : 'Failed' });
    } finally {
      setRunningJobKey((cur) => cur === key ? null : cur);
    }
  };

  const handleDeleteJob = async (job: SimulationJob) => {
    const confirmed = window.confirm(`Delete job "${job.name}" and all its run history? Associated sessions will be kept.`);
    if (!confirmed) return;
    try {
      const resp = await apiFetch(`/simulation-jobs/${job.job_id}`, { method: 'DELETE' });
      if (!resp.ok) throw new Error(await resp.text());
      toast.success('Job deleted');
      if (selectedJobId === job.job_id) setSelectedJobId(null);
      await fetchJobs();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : 'Delete failed');
    }
  };

  const handleRunEnabled = async () => {
    setFeedback(null); setRunTriggering(true);
    try {
      const resp = await apiFetch('/simulation-jobs/run-enabled', { method: 'POST' });
      if (!resp.ok) throw new Error(await resp.text());
      await fetchJobs();
      if (selectedJobId) await refreshRuns(selectedJobId);
      setFeedback({ type: 'ok', text: 'Submitted for enabled jobs' });
    } catch (err) {
      setFeedback({ type: 'err', text: err instanceof Error ? err.message : 'Failed' });
    } finally { setRunTriggering(false); }
  };

  const handleDataUpdate = async () => {
    setFeedback(null); setDataUpdating(true);
    try {
      const resp = await apiFetch('/data-update/run', { method: 'POST', body: JSON.stringify({}) });
      if (!resp.ok) throw new Error(await resp.text());
      await fetchJobs();
      setFeedback({ type: 'ok', text: 'Data update submitted' });
    } catch (err) {
      setFeedback({ type: 'err', text: err instanceof Error ? err.message : 'Failed' });
    } finally { setDataUpdating(false); }
  };

  // ── Render ──

  return (
    <SectionCard
      title="Scheduled Backtests"
      description="Recurring backtests that replay incrementally as new market data arrives."
      action={
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" size="sm" onClick={handleRunEnabled} disabled={runTriggering}>
            {runTriggering ? '...' : 'Trigger All'}
          </Button>
          <Button variant="outline" size="sm" onClick={handleDataUpdate} disabled={dataUpdating}>
            {dataUpdating ? '...' : 'Update Data'}
          </Button>
          <Button variant="ghost" size="sm" onClick={onOpenMarketAdmin}>Market DB</Button>
        </div>
      }
    >
      {feedback && (
        <div className={`text-sm ${feedback.type === 'ok' ? 'text-emerald-500' : 'text-destructive'}`}>{feedback.text}</div>
      )}

      <div className="overflow-x-auto">
        <table className="data-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Strategy · Symbol</th>
              <th>Status</th>
              <th>Notify</th>
              <th>Last Processed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr
                key={job.job_id}
                onClick={() => setSelectedJobId(job.job_id)}
                style={{
                  backgroundColor: selectedJobId === job.job_id ? 'rgba(34, 211, 238, 0.08)' : 'transparent',
                  cursor: 'pointer',
                }}
              >
                <td><div style={{ fontWeight: 600 }}>{job.name}</div></td>
                <td>
                  <div>{job.strategy_name}</div>
                  <div className="tagline" style={{ fontSize: '0.7rem' }}>{job.symbol}</div>
                </td>
                <td>
                  <div className="space-y-1">
                    <StatusBadge value={job.enabled ? 'running' : 'idle'} />
                    <div className="tagline" style={{ fontSize: '0.7rem' }}>{formatStatusLabel(job.status)}</div>
                  </div>
                </td>
                <td>
                  {job.notification?.telegram?.enabled
                    ? <StatusBadge value="success" />
                    : <span className="tagline">Muted</span>}
                </td>
                <td style={{ fontSize: '0.8rem' }}>{job.last_processed_at || '—'}</td>
                <td onClick={(e) => e.stopPropagation()}>
                  <div className="flex flex-wrap gap-2">
                    <Button variant="outline" size="sm" onClick={() => handleToggleJob(job)}>
                      {job.enabled ? 'Disable' : 'Enable'}
                    </Button>
                    <Button variant="outline" size="sm" onClick={() => handleToggleNotify(job)}>
                      {job.notification?.telegram?.enabled ? 'Mute' : 'Notify'}
                    </Button>
                    <Button
                      variant="outline"
                      size="sm"
                      disabled={!!runningJobKey}
                      onClick={() => handleRunJob(job.job_id)}
                    >
                      {runningJobKey === `run:${job.job_id}` ? '...' : 'Run'}
                    </Button>
                    <Button
                      variant="ghost"
                      size="sm"
                      disabled={!!runningJobKey}
                      onClick={() => handleRunJob(job.job_id, true)}
                    >
                      {runningJobKey === `force:${job.job_id}` ? '...' : 'Force'}
                    </Button>
                    {job.latest_session_id && (
                      <Button variant="ghost" size="sm" onClick={() => onSelectSession(job.latest_session_id!)}>
                        View
                      </Button>
                    )}
                    <Button variant="danger" size="sm" onClick={() => handleDeleteJob(job)}>
                      Delete
                    </Button>
                  </div>
                </td>
              </tr>
            ))}
            {jobs.length === 0 && (
              <tr>
                <td colSpan={6} style={{ textAlign: 'center', color: 'var(--color-text-dim)', padding: '2rem' }}>
                  No automation jobs
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {currentJob && (
        <div className="mt-2">
          <div className="text-sm font-medium mb-3">Runs — {currentJob.name}</div>
          <div className="grid gap-3">
            {runs.map((run) => (
              <div key={run.run_id} className="rounded-xl border border-border/60 p-3.5 bg-secondary/20">
                <div className="flex justify-between text-sm">
                  <strong>{formatStatusLabel(run.status)}</strong>
                  <span className="tagline">{run.progress.toFixed(0)}%</span>
                </div>
                <div className="tagline mt-1.5">
                  {run.start_date} → {run.end_date || '...'} · {run.bars_processed} bars
                </div>
                {run.session_id && (
                  <Button variant="ghost" size="sm" className="mt-2" onClick={() => onSelectSession(run.session_id!)}>
                    Open Session
                  </Button>
                )}
              </div>
            ))}
            {runs.length === 0 && <div className="tagline">No runs yet</div>}
          </div>
        </div>
      )}
    </SectionCard>
  );
};

export default SimulationPanel;
