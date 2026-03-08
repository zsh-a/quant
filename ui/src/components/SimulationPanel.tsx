import React, { useEffect, useMemo, useRef, useState } from 'react';
import type { DataUpdateRun, SimulationJob, SimulationRun, SimulationStep, StrategyMeta } from '../types';
import StrategyConfigForm from './StrategyConfigForm';
import { API_BASE } from '../utils/api';
import { formatPrice } from '../utils/format';
import { formatSourceLabel, formatStatusLabel } from '../utils/display';

interface SimulationPanelProps {
  strategies: StrategyMeta[];
  onSelectSession: (sessionId: string) => void;
}

const cardStyle: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  padding: '1rem',
};

export const SimulationPanel: React.FC<SimulationPanelProps> = ({ strategies, onSelectSession }) => {
  const [jobs, setJobs] = useState<SimulationJob[]>([]);
  const [dataUpdates, setDataUpdates] = useState<DataUpdateRun[]>([]);
  const [selectedJobId, setSelectedJobId] = useState<string | null>(null);
  const [runs, setRuns] = useState<SimulationRun[]>([]);
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [steps, setSteps] = useState<SimulationStep[]>([]);
  const [stepFilter, setStepFilter] = useState<'important' | 'all' | 'trade' | 'error'>('important');
  const [submitting, setSubmitting] = useState(false);
  const [dataUpdating, setDataUpdating] = useState(false);
  const [runTriggering, setRunTriggering] = useState(false);
  const [runningJobKey, setRunningJobKey] = useState<string | null>(null);
  const [name, setName] = useState('模拟任务');
  const [strategy, setStrategy] = useState('');
  const [symbol, setSymbol] = useState('sh.000300');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [endDate, setEndDate] = useState('');
  const [paramValues, setParamValues] = useState<Record<string, any>>({});
  const [notifyOnOrder, setNotifyOnOrder] = useState(false);
  const [telegramChatId, setTelegramChatId] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const pollInFlightRef = useRef(false);
  const jobsAbortRef = useRef<AbortController | null>(null);
  const updatesAbortRef = useRef<AbortController | null>(null);
  const runsAbortRef = useRef<AbortController | null>(null);
  const stepsAbortRef = useRef<AbortController | null>(null);

  const currentJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) || null,
    [jobs, selectedJobId],
  );

  const filteredSteps = useMemo(() => {
    return steps.filter((step) => {
      const hasTrades = Boolean(step.payload?.new_trades?.length);
      const hasError = Boolean(step.payload?.error);
      const isLifecycle = step.event_type !== 'strategy_step';

      switch (stepFilter) {
        case 'trade':
          return hasTrades;
        case 'error':
          return hasError;
        case 'all':
          return true;
        case 'important':
        default:
          return isLifecycle || hasTrades || hasError;
      }
    });
  }, [stepFilter, steps]);

  const sleep = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

  const refreshRunsWithRetry = async (jobId: string) => {
    for (const waitMs of [0, 600, 1500, 3000]) {
      if (waitMs > 0) {
        await sleep(waitMs);
      }
      await fetchJobs();
      await fetchRuns(jobId);
    }
  };

  useEffect(() => {
    if (!strategy && strategies.length > 0) {
      setStrategy(strategies[0].name);
    }
  }, [strategies, strategy]);

  useEffect(() => {
    const strat = strategies.find((item) => item.name === strategy);
    if (!strat?.params) return;
    const defaults: Record<string, unknown> = {};
    Object.entries(strat.params).forEach(([key, conf]) => {
      defaults[key] = conf.default;
    });
    setParamValues(defaults);
  }, [strategy, strategies]);

  const resetDefaults = () => {
    const strat = strategies.find((item) => item.name === strategy);
    if (!strat?.params) return;
    const defaults: Record<string, unknown> = {};
    Object.entries(strat.params).forEach(([key, conf]) => {
      defaults[key] = conf.default;
    });
    setParamValues(defaults);
  };

  const normalizeParams = () => {
    const strat = strategies.find((item) => item.name === strategy);
    const finalParams: Record<string, any> = {};
    if (!strat?.params) return finalParams;

    Object.entries(strat.params).forEach(([key, conf]) => {
      const userVal = paramValues[key];
      if (userVal === undefined || userVal === '') {
        finalParams[key] = conf.default;
      } else if (conf.type === 'int') {
        finalParams[key] = parseInt(userVal, 10) || 0;
      } else if (conf.type === 'float') {
        finalParams[key] = parseFloat(userVal) || 0;
      } else if (conf.type === 'bool') {
        finalParams[key] = Boolean(userVal);
      } else {
        finalParams[key] = userVal;
      }
    });
    return finalParams;
  };

  const fetchJobs = async () => {
    jobsAbortRef.current?.abort();
    const controller = new AbortController();
    jobsAbortRef.current = controller;
    const resp = await fetch(`${API_BASE}/simulation-jobs`, { signal: controller.signal });
    const data = await resp.json();
    setJobs(data);
    setSelectedJobId((prev) => {
      if (data.length === 0) return null;
      if (prev && data.some((job: SimulationJob) => job.job_id === prev)) return prev;
      return data[0].job_id;
    });
  };

  const fetchDataUpdates = async () => {
    updatesAbortRef.current?.abort();
    const controller = new AbortController();
    updatesAbortRef.current = controller;
    const resp = await fetch(`${API_BASE}/data-update/history?limit=10`, { signal: controller.signal });
    const data = await resp.json();
    setDataUpdates(data);
  };

  const fetchRuns = async (jobId: string) => {
    runsAbortRef.current?.abort();
    const controller = new AbortController();
    runsAbortRef.current = controller;
    const resp = await fetch(`${API_BASE}/simulation-jobs/${jobId}/runs?limit=20`, { signal: controller.signal });
    const data = await resp.json();
    setRuns(data);
    if (data.length > 0) {
      setSelectedRunId((prev) => prev && data.some((item: SimulationRun) => item.run_id === prev) ? prev : data[0].run_id);
    } else {
      setSelectedRunId(null);
      setSteps([]);
    }
  };

  const fetchSteps = async (runId: string) => {
    stepsAbortRef.current?.abort();
    const controller = new AbortController();
    stepsAbortRef.current = controller;
    const latestStep = steps[0]?.run_id === runId ? Math.max(...steps.map((step) => step.step_index)) : null;
    const query = latestStep !== null ? `?limit=300&since_step=${latestStep}` : '?limit=120';
    const resp = await fetch(`${API_BASE}/simulation-runs/${runId}/steps${query}`, { signal: controller.signal });
    const data = await resp.json();
    setSteps((prev) => {
      if (latestStep === null) {
        return data;
      }
      if (!Array.isArray(data) || data.length === 0) {
        return prev;
      }
      const seen = new Set(prev.map((step) => step.id));
      const incoming = data.filter((step: SimulationStep) => !seen.has(step.id));
      return incoming.length > 0 ? [...incoming.reverse(), ...prev] : prev;
    });
  };

  useEffect(() => {
    let active = true;

    const poll = async () => {
      if (!active || pollInFlightRef.current) {
        return;
      }
      pollInFlightRef.current = true;
      try {
        await Promise.all([fetchJobs(), fetchDataUpdates()]);
        if (selectedJobId) {
          await fetchRuns(selectedJobId);
        }
        if (selectedRunId) {
          await fetchSteps(selectedRunId);
        }
      } finally {
        pollInFlightRef.current = false;
      }
    };

    poll();
    const interval = window.setInterval(poll, 5000);
    return () => {
      active = false;
      window.clearInterval(interval);
      jobsAbortRef.current?.abort();
      updatesAbortRef.current?.abort();
      runsAbortRef.current?.abort();
      stepsAbortRef.current?.abort();
    };
  }, [selectedJobId, selectedRunId]);

  useEffect(() => {
    if (selectedJobId) {
      fetchRuns(selectedJobId);
    }
  }, [selectedJobId]);

  useEffect(() => {
    if (selectedRunId) {
      fetchSteps(selectedRunId);
    }
  }, [selectedRunId]);

  const handleCreateJob = async () => {
    setError(null);
    setMessage(null);
    setSubmitting(true);
    try {
      const resp = await fetch(`${API_BASE}/simulation-jobs`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name,
          strategy,
          symbol,
          start_date: startDate,
          end_date: endDate || null,
          params: normalizeParams(),
          notification: {
            telegram: {
              enabled: notifyOnOrder,
              chat_id: telegramChatId.trim() || undefined,
            },
          },
          enabled: true,
          schedule: 'daily',
        }),
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      await fetchJobs();
      setMessage('模拟任务已创建');
    } catch (err) {
      setError(err instanceof Error ? err.message : '创建模拟任务失败');
    } finally {
      setSubmitting(false);
    }
  };

  const handleToggleJob = async (job: SimulationJob) => {
    const endpoint = job.enabled ? 'disable' : 'enable';
    await fetch(`${API_BASE}/simulation-jobs/${job.job_id}/${endpoint}`, { method: 'POST' });
    await fetchJobs();
  };

  const handleToggleNotification = async (job: SimulationJob) => {
    setError(null);
    setMessage(null);
    const telegram = job.notification?.telegram;
    const nextEnabled = !telegram?.enabled;
    const resp = await fetch(`${API_BASE}/simulation-jobs/${job.job_id}/notification`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        notification: {
          telegram: {
            enabled: nextEnabled,
            chat_id: telegram?.chat_id || undefined,
          },
        },
      }),
    });
    if (!resp.ok) {
      setError(await resp.text());
      return;
    }
    await fetchJobs();
    setMessage(nextEnabled ? '已开启下单通知' : '已关闭下单通知');
  };

  const handleRunJob = async (jobId: string, force = false) => {
    setError(null);
    setMessage(null);
    setSelectedJobId(jobId);
    const actionKey = `${force ? 'force' : 'run'}:${jobId}`;
    setRunningJobKey(actionKey);
    try {
      const query = force ? '?force=true' : '';
      const resp = await fetch(`${API_BASE}/simulation-jobs/${jobId}/run${query}`, { method: 'POST' });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      const data = await resp.json();
      setMessage(force ? '已提交强制重跑任务，正在刷新批次列表…' : '已提交续跑任务，正在刷新批次列表…');
      await refreshRunsWithRetry(jobId);
      if (data.task_id) {
        setMessage(force ? `强制重跑任务已提交：${data.task_id}` : `续跑任务已提交：${data.task_id}`);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : force ? '强制重跑失败' : '触发模拟任务失败');
    } finally {
      setRunningJobKey((current) => current === actionKey ? null : current);
    }
  };

  const handleRunEnabledJobs = async () => {
    setError(null);
    setMessage(null);
    setRunTriggering(true);
    try {
      const resp = await fetch(`${API_BASE}/simulation-jobs/run-enabled`, { method: 'POST' });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      await fetchJobs();
      if (selectedJobId) {
        await refreshRunsWithRetry(selectedJobId);
      }
      setMessage('已提交启用任务的续跑请求');
    } catch (err) {
      setError(err instanceof Error ? err.message : '仅触发运行失败');
    } finally {
      setRunTriggering(false);
    }
  };

  const handleDataUpdate = async () => {
    setError(null);
    setMessage(null);
    setDataUpdating(true);
    try {
      const resp = await fetch(`${API_BASE}/data-update/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      await fetchDataUpdates();
      await fetchJobs();
    } catch (err) {
      setError(err instanceof Error ? err.message : '更新数据并触发模拟失败');
    } finally {
      setDataUpdating(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr 1fr', gap: '1rem' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <StrategyConfigForm
          title="模拟任务"
          strategies={strategies}
          selectedStrategy={strategy}
          onStrategyChange={setStrategy}
          symbol={symbol}
          onSymbolChange={setSymbol}
          startDate={startDate}
          onStartDateChange={setStartDate}
          endDate={endDate}
          onEndDateChange={setEndDate}
          paramValues={paramValues}
          onParamChange={(key, value) => setParamValues((prev) => ({ ...prev, [key]: value }))}
          onResetDefaults={resetDefaults}
          showEndDate
          headerAction={
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap', justifyContent: 'flex-end' }}>
              <button className="btn-ghost" onClick={handleRunEnabledJobs} disabled={runTriggering}>
                {runTriggering ? '触发中...' : '仅触发运行'}
              </button>
              <button className="btn-primary" onClick={handleDataUpdate} disabled={dataUpdating}>
                {dataUpdating ? '更新中...' : '更新数据并触发模拟'}
              </button>
            </div>
          }
          footer={
            <>
              <div style={{ display: 'grid', gap: '0.4rem' }}>
                <label className="tagline">模拟任务名称</label>
                <input className="glass-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="模拟任务名称" />
              </div>
              <label style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <input
                  type="checkbox"
                  checked={notifyOnOrder}
                  onChange={(e) => setNotifyOnOrder(e.target.checked)}
                  style={{ width: 'auto' }}
                />
                <span className="tagline" style={{ marginBottom: 0 }}>Telegram 下单通知</span>
              </label>
              {notifyOnOrder && (
                <div style={{ display: 'grid', gap: '0.4rem' }}>
                  <label className="tagline">Telegram Chat ID，可留空走服务端默认值</label>
                  <input
                    className="glass-input"
                    value={telegramChatId}
                    onChange={(e) => setTelegramChatId(e.target.value)}
                    placeholder="例如 123456789"
                  />
                </div>
              )}
              {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}
              {message && <div style={{ color: 'var(--success)' }}>{message}</div>}
              <div className="tagline">可选结束时间仅用于首次验证窗口，后续“立即续跑”会自动追到最新日期。</div>
              <button className="btn-primary" onClick={handleCreateJob} disabled={submitting || !strategy}>
                {submitting ? '创建中...' : '创建模拟任务'}
              </button>
            </>
          }
        />

        <section style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>模拟任务列表</h3>
          <div style={{ display: 'grid', gap: '0.75rem' }}>
            {jobs.map((job) => (
              <div
                key={job.job_id}
                onClick={() => setSelectedJobId(job.job_id)}
                style={{
                  padding: '0.9rem',
                  borderRadius: 12,
                  cursor: 'pointer',
                  border: selectedJobId === job.job_id ? '1px solid var(--primary)' : '1px solid rgba(255,255,255,0.08)',
                  background: selectedJobId === job.job_id ? 'rgba(99, 102, 241, 0.08)' : 'rgba(255,255,255,0.03)',
                }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
                  <div>
                    <div style={{ fontWeight: 700 }}>{job.name}</div>
                    <div className="tagline">{job.strategy_name} · {job.symbol}</div>
                  </div>
                  <span className={`status-badge ${job.enabled ? 'status-backtest' : ''}`}>{job.enabled ? '启用中' : '已停用'}</span>
                </div>
                <div className="tagline" style={{ marginTop: '0.5rem' }}>
                  首次验证：{job.start_date} → {job.end_date || '最新数据'}
                </div>
                <div className="tagline" style={{ marginTop: '0.35rem' }}>
                  状态：{formatStatusLabel(job.status)} · 最近处理到：{job.last_processed_at || '未开始'}
                </div>
                <div className="tagline" style={{ marginTop: '0.35rem' }}>
                  下单通知：{job.notification?.telegram?.enabled ? '已开启' : '已关闭'}
                  {job.notification?.telegram?.chat_id ? ` · Chat ID ${job.notification.telegram.chat_id}` : ''}
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
                  <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleToggleJob(job); }}>
                    {job.enabled ? '停用' : '启用'}
                  </button>
                  <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleToggleNotification(job); }}>
                    {job.notification?.telegram?.enabled ? '关闭通知' : '开启通知'}
                  </button>
                  <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleRunJob(job.job_id); }} disabled={runningJobKey === `run:${job.job_id}` || runningJobKey === `force:${job.job_id}`}>
                    {runningJobKey === `run:${job.job_id}` ? '提交中...' : '立即续跑'}
                  </button>
                  <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleRunJob(job.job_id, true); }} disabled={runningJobKey === `run:${job.job_id}` || runningJobKey === `force:${job.job_id}`}>
                    {runningJobKey === `force:${job.job_id}` ? '提交中...' : '强制重跑'}
                  </button>
                  {job.latest_session_id && (
                    <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); onSelectSession(job.latest_session_id!); }}>
                      查看会话
                    </button>
                  )}
                </div>
              </div>
            ))}
            {jobs.length === 0 && <div className="tagline">暂无模拟任务</div>}
          </div>
        </section>
      </div>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>数据更新与批次</h3>
        <div style={{ display: 'grid', gap: '0.75rem' }}>
          {dataUpdates.map((item) => (
            <div key={item.update_run_id} style={{ padding: '0.8rem', borderRadius: 12, background: 'rgba(255,255,255,0.03)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <strong>{formatStatusLabel(item.status)}</strong>
                <span className="tagline">{formatSourceLabel(item.trigger_source)}</span>
              </div>
              <div className="tagline" style={{ marginTop: '0.4rem' }}>
                新数据：{item.has_new_data ? '是' : '否'} · {item.completed_at || item.started_at || item.created_at}
              </div>
              {item.details?.triggered_jobs?.length ? (
                <div className="tagline" style={{ marginTop: '0.4rem' }}>
                  触发任务：{item.details.triggered_jobs.map((job: { name: string }) => job.name).join('，')}
                </div>
              ) : null}
            </div>
          ))}
        </div>

        <h3 style={{ marginBottom: '0.75rem', marginTop: '1.2rem' }}>运行批次</h3>
        <div style={{ display: 'grid', gap: '0.75rem' }}>
          {runs.map((run) => (
            <div
              key={run.run_id}
              onClick={() => setSelectedRunId(run.run_id)}
              style={{
                padding: '0.8rem',
                borderRadius: 12,
                cursor: 'pointer',
                border: selectedRunId === run.run_id ? '1px solid var(--primary)' : '1px solid rgba(255,255,255,0.08)',
                background: 'rgba(255,255,255,0.03)',
              }}
            >
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <strong>{formatStatusLabel(run.status)}</strong>
                <span className="tagline">{run.progress.toFixed(0)}%</span>
              </div>
              <div className="tagline" style={{ marginTop: '0.4rem' }}>
                {run.start_date} → {run.end_date || '进行中'}
              </div>
              <div className="tagline">bar 数：{run.bars_processed} · step：{run.steps_recorded}</div>
              {run.session_id && (
                <button className="btn-ghost" style={{ marginTop: '0.5rem' }} onClick={(e) => { e.stopPropagation(); onSelectSession(run.session_id!); }}>
                  打开会话
                </button>
              )}
            </div>
          ))}
          {currentJob && runs.length === 0 && <div className="tagline">该模拟任务暂无运行批次</div>}
        </div>
      </section>

      <section style={cardStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <h3 style={{ marginTop: 0, marginBottom: 0 }}>逐 Bar 执行轨迹</h3>
          <div style={{ display: 'flex', gap: '0.4rem', flexWrap: 'wrap' }}>
            <button className="btn-ghost" onClick={() => setStepFilter('important')} disabled={stepFilter === 'important'}>重要事件</button>
            <button className="btn-ghost" onClick={() => setStepFilter('trade')} disabled={stepFilter === 'trade'}>仅成交</button>
            <button className="btn-ghost" onClick={() => setStepFilter('error')} disabled={stepFilter === 'error'}>仅异常</button>
            <button className="btn-ghost" onClick={() => setStepFilter('all')} disabled={stepFilter === 'all'}>全部</button>
          </div>
        </div>
        <div className="tagline" style={{ marginTop: '0.5rem', marginBottom: '0.8rem' }}>默认仅展示生命周期、成交和异常；普通 bar 以紧凑摘要显示。</div>
        {selectedRunId ? (
          <div style={{ display: 'grid', gap: '0.55rem', maxHeight: '80vh', overflowY: 'auto' }}>
            {filteredSteps.map((step) => {
              const hasTrades = Boolean(step.payload?.new_trades?.length);
              const hasError = Boolean(step.payload?.error);
              const closePrices = step.payload?.close_prices
                ? Object.entries(step.payload.close_prices).map(([code, price]) => `${code}=${formatPrice(Number(price), 2)}`).join('，')
                : null;
              const summaryLine = [
                step.payload?.progress !== undefined ? `进度 ${Number(step.payload.progress).toFixed(2)}%` : null,
                step.payload?.total_equity !== undefined ? `权益 ${step.payload.total_equity}` : null,
                step.payload?.cash !== undefined ? `现金 ${step.payload.cash}` : null,
                closePrices ? `行情 ${closePrices}` : null,
              ].filter(Boolean).join(' · ');

              return (
                <div
                  key={step.id}
                  style={{
                    padding: '0.7rem 0.8rem',
                    borderRadius: 12,
                    background: hasError
                      ? 'rgba(239,68,68,0.08)'
                      : hasTrades
                        ? 'rgba(34,197,94,0.10)'
                        : 'rgba(255,255,255,0.03)',
                    border: step.event_type !== 'strategy_step'
                      ? '1px solid rgba(99,102,241,0.25)'
                      : hasError
                        ? '1px solid rgba(239,68,68,0.25)'
                        : hasTrades
                          ? '1px solid rgba(34,197,94,0.28)'
                          : '1px solid rgba(255,255,255,0.06)',
                    boxShadow: hasTrades ? '0 0 0 1px rgba(34,197,94,0.08) inset' : 'none',
                  }}
                >
                  <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem', alignItems: 'center' }}>
                    <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center', flexWrap: 'wrap' }}>
                      <strong>{step.event_type}</strong>
                      {hasTrades && (
                        <span
                          style={{
                            color: 'var(--success)',
                            background: 'rgba(34,197,94,0.12)',
                            border: '1px solid rgba(34,197,94,0.25)',
                            borderRadius: 999,
                            padding: '0.1rem 0.45rem',
                            fontSize: '0.75rem',
                            fontWeight: 700,
                          }}
                        >
                          成交 {step.payload.new_trades.length}
                        </span>
                      )}
                      {hasError && <span className="tagline" style={{ color: 'var(--danger)' }}>异常</span>}
                    </div>
                    <span className="tagline">#{step.step_index} · {step.timestamp || step.created_at}</span>
                  </div>
                  {summaryLine && (
                    <div className="tagline" style={{ marginTop: '0.35rem' }}>{summaryLine}</div>
                  )}
                  {hasTrades ? (
                    <div style={{ marginTop: '0.45rem', display: 'grid', gap: '0.35rem' }}>
                      {step.payload.new_trades.map((trade: any, index: number) => (
                        <div
                          key={`${step.id}-${index}-${trade.symbol}-${trade.timestamp || trade.price}`}
                          style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            gap: '0.75rem',
                            padding: '0.45rem 0.6rem',
                            borderRadius: 10,
                            background: 'rgba(34,197,94,0.08)',
                            border: '1px solid rgba(34,197,94,0.18)',
                            fontSize: '0.85rem',
                          }}
                        >
                          <span style={{ fontWeight: 700, color: 'var(--success)' }}>
                            {String(trade.type || '').toUpperCase()} {trade.symbol}
                          </span>
                          <span className="tagline">
                            @{formatPrice(trade.price, 2)} × {trade.quantity}
                          </span>
                        </div>
                      ))}
                    </div>
                  ) : null}
                  {step.event_type !== 'strategy_step' && step.payload?.positions && Object.keys(step.payload.positions).length > 0 ? (
                    <div className="tagline" style={{ marginTop: '0.35rem' }}>
                      持仓：{Object.entries(step.payload.positions).map(([code, position]: any) => `${code}(${position.qty})`).join('，')}
                    </div>
                  ) : null}
                  {hasError && (
                    <div style={{ color: 'var(--danger)', marginTop: '0.35rem' }}>{step.payload.error}</div>
                  )}
                </div>
              );
            })}
            {filteredSteps.length === 0 && <div className="tagline">当前筛选下暂无轨迹事件</div>}
          </div>
        ) : (
          <div className="tagline">请选择左侧运行批次查看执行轨迹</div>
        )}
      </section>
    </div>
  );
};

export default SimulationPanel;
