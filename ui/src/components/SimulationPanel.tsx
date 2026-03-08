import React, { useEffect, useMemo, useState } from 'react';
import type { DataUpdateRun, SimulationJob, SimulationRun, SimulationStep, StrategyMeta } from '../types';
import StrategyConfigForm from './StrategyConfigForm';
import { API_BASE } from '../utils/api';

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
  const [submitting, setSubmitting] = useState(false);
  const [dataUpdating, setDataUpdating] = useState(false);
  const [name, setName] = useState('模拟任务');
  const [strategy, setStrategy] = useState('');
  const [symbol, setSymbol] = useState('sh.000300');
  const [startDate, setStartDate] = useState('2024-01-01');
  const [paramValues, setParamValues] = useState<Record<string, any>>({});
  const [error, setError] = useState<string | null>(null);

  const currentJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) || null,
    [jobs, selectedJobId],
  );

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
    const resp = await fetch(`${API_BASE}/simulation-jobs`);
    const data = await resp.json();
    setJobs(data);
    if (!selectedJobId && data.length > 0) {
      setSelectedJobId(data[0].job_id);
    }
  };

  const fetchDataUpdates = async () => {
    const resp = await fetch(`${API_BASE}/data-update/history?limit=10`);
    const data = await resp.json();
    setDataUpdates(data);
  };

  const fetchRuns = async (jobId: string) => {
    const resp = await fetch(`${API_BASE}/simulation-jobs/${jobId}/runs?limit=20`);
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
    const resp = await fetch(`${API_BASE}/simulation-runs/${runId}/steps?limit=300`);
    const data = await resp.json();
    setSteps(data);
  };

  useEffect(() => {
    fetchJobs();
    fetchDataUpdates();
    const interval = window.setInterval(() => {
      fetchJobs();
      fetchDataUpdates();
      if (selectedJobId) fetchRuns(selectedJobId);
      if (selectedRunId) fetchSteps(selectedRunId);
    }, 3000);
    return () => clearInterval(interval);
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
          params: normalizeParams(),
          enabled: true,
          schedule: 'daily',
        }),
      });
      if (!resp.ok) {
        throw new Error(await resp.text());
      }
      await fetchJobs();
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

  const handleRunJob = async (jobId: string) => {
    await fetch(`${API_BASE}/simulation-jobs/${jobId}/run`, { method: 'POST' });
    await fetchJobs();
    await fetchRuns(jobId);
  };

  const handleDataUpdate = async () => {
    setDataUpdating(true);
    try {
      await fetch(`${API_BASE}/data-update/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      await fetchDataUpdates();
    } finally {
      setDataUpdating(false);
    }
  };

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.1fr 1fr 1fr', gap: '1rem' }}>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <StrategyConfigForm
          title="Simulation Tasks"
          strategies={strategies}
          selectedStrategy={strategy}
          onStrategyChange={setStrategy}
          symbol={symbol}
          onSymbolChange={setSymbol}
          startDate={startDate}
          onStartDateChange={setStartDate}
          paramValues={paramValues}
          onParamChange={(key, value) => setParamValues((prev) => ({ ...prev, [key]: value }))}
          onResetDefaults={resetDefaults}
          showEndDate={false}
          headerAction={
            <button className="btn-primary" onClick={handleDataUpdate} disabled={dataUpdating}>
              {dataUpdating ? '触发中...' : '更新数据并触发模拟'}
            </button>
          }
          footer={
            <>
              <div style={{ display: 'grid', gap: '0.4rem' }}>
                <label className="tagline">模拟任务名称</label>
                <input className="glass-input" value={name} onChange={(e) => setName(e.target.value)} placeholder="模拟任务名称" />
              </div>
              {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}
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
                  <span className={`status-badge ${job.enabled ? 'status-backtest' : ''}`}>{job.enabled ? 'enabled' : 'disabled'}</span>
                </div>
                <div className="tagline" style={{ marginTop: '0.5rem' }}>
                  状态：{job.status} · 最近处理到：{job.last_processed_at || '未开始'}
                </div>
                <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.75rem' }}>
                  <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleToggleJob(job); }}>
                    {job.enabled ? '停用' : '启用'}
                  </button>
                  <button className="btn-ghost" onClick={(e) => { e.stopPropagation(); handleRunJob(job.job_id); }}>
                    立即续跑
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
                <strong>{item.status}</strong>
                <span className="tagline">{item.trigger_source}</span>
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
                <strong>{run.status}</strong>
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
        <h3 style={{ marginTop: 0 }}>逐 Bar 执行轨迹</h3>
        {selectedRunId ? (
          <div style={{ display: 'grid', gap: '0.75rem', maxHeight: '80vh', overflowY: 'auto' }}>
            {steps.map((step) => (
              <div key={step.id} style={{ padding: '0.8rem', borderRadius: 12, background: 'rgba(255,255,255,0.03)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.75rem' }}>
                  <strong>{step.event_type}</strong>
                  <span className="tagline">#{step.step_index} · {step.timestamp || step.created_at}</span>
                </div>
                {step.payload?.progress !== undefined && (
                  <div className="tagline" style={{ marginTop: '0.4rem' }}>
                    进度：{Number(step.payload.progress).toFixed(2)}% · 总权益：{step.payload.total_equity ?? '--'} · 现金：{step.payload.cash ?? '--'}
                  </div>
                )}
                {step.payload?.close_prices && (
                  <div className="tagline" style={{ marginTop: '0.4rem' }}>
                    行情：{Object.entries(step.payload.close_prices).map(([code, price]) => `${code}=${price}`).join('，')}
                  </div>
                )}
                {step.payload?.new_trades?.length ? (
                  <div className="tagline" style={{ marginTop: '0.4rem' }}>
                    成交：{step.payload.new_trades.map((trade: any) => `${trade.type} ${trade.symbol} @ ${trade.price}`).join('；')}
                  </div>
                ) : null}
                {step.payload?.positions && Object.keys(step.payload.positions).length > 0 && (
                  <div className="tagline" style={{ marginTop: '0.4rem' }}>
                    持仓：{Object.entries(step.payload.positions).map(([code, position]: any) => `${code}(${position.qty})`).join('，')}
                  </div>
                )}
                {step.payload?.error && (
                  <div style={{ color: 'var(--danger)', marginTop: '0.4rem' }}>{step.payload.error}</div>
                )}
              </div>
            ))}
          </div>
        ) : (
          <div className="tagline">请选择左侧运行批次查看执行轨迹</div>
        )}
      </section>
    </div>
  );
};

export default SimulationPanel;
