/**
 * Automation Jobs — job 管理 + 运行触发，不含策略配置（已统一到 NewSessionForm）。
 */
import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { SimulationJob, SimulationRun } from '../types';
import { apiFetch } from '../utils/api';
import { formatStatusLabel } from '../utils/display';
import { Button } from './ui/button';
import { SectionCard } from './layout/SectionCard';

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

  useEffect(() => { if (selectedJobId) fetchRuns(selectedJobId); }, [selectedJobId]);

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

  if (jobs.length === 0) return null;

  return (
    <SectionCard title="Automation Jobs">
      <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
        <span className="text-sm text-muted-foreground">
          {jobs.length} job{jobs.length !== 1 ? 's' : ''}
        </span>
        <div className="flex gap-2 flex-wrap">
          <Button variant="outline" size="sm" onClick={handleRunEnabled} disabled={runTriggering}>
            {runTriggering ? '...' : 'Trigger All'}
          </Button>
          <Button variant="outline" size="sm" onClick={handleDataUpdate} disabled={dataUpdating}>
            {dataUpdating ? '...' : 'Update Data'}
          </Button>
          <Button variant="ghost" size="sm" onClick={onOpenMarketAdmin}>Market DB</Button>
        </div>
      </div>
      {feedback && (
        <div className={`text-sm mb-3 ${feedback.type === 'ok' ? 'text-emerald-500' : 'text-destructive'}`}>{feedback.text}</div>
      )}

      {/* Jobs */}
      <div className="grid gap-3">
        {jobs.map((job) => (
          <div
            key={job.job_id}
            onClick={() => setSelectedJobId(job.job_id)}
            className="rounded-xl border p-4 cursor-pointer transition-colors"
            style={{
              borderColor: selectedJobId === job.job_id ? 'var(--primary)' : 'rgba(255,255,255,0.08)',
              background: selectedJobId === job.job_id ? 'rgba(99,102,241,0.08)' : 'rgba(255,255,255,0.03)',
            }}
          >
            <div className="flex justify-between gap-3">
              <div>
                <div className="font-bold">{job.name}</div>
                <div className="tagline mt-1">{job.strategy_name} · {job.symbol}</div>
              </div>
              <span className={`status-badge ${job.enabled ? 'status-backtest' : ''}`}>
                {job.enabled ? 'On' : 'Off'}
              </span>
            </div>
            <div className="tagline mt-2">
              {formatStatusLabel(job.status)} · Last: {job.last_processed_at || '—'}
            </div>
            <div className="flex gap-2 mt-3 flex-wrap">
              <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleToggleJob(job); }}>
                {job.enabled ? 'Disable' : 'Enable'}
              </button>
              <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleToggleNotify(job); }}>
                {job.notification?.telegram?.enabled ? 'Mute' : 'Notify'}
              </button>
              <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleRunJob(job.job_id); }}
                disabled={!!runningJobKey}>
                {runningJobKey === `run:${job.job_id}` ? '...' : 'Run'}
              </button>
              <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleRunJob(job.job_id, true); }}
                disabled={!!runningJobKey}>
                {runningJobKey === `force:${job.job_id}` ? '...' : 'Force'}
              </button>
              {job.latest_session_id && (
                <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); onSelectSession(job.latest_session_id!); }}>
                  View
                </button>
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Runs for selected job */}
      {currentJob && (
        <div className="mt-5">
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
