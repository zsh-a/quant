import React, { useEffect, useMemo, useState } from 'react';
import type { SessionSummary, SimulationRun, SimulationStep, Trade } from '../types';
import { formatPrice } from '../utils/format';
import { formatModeLabel, formatSourceLabel, formatStatusLabel } from '../utils/display';

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
  ? 'http://localhost:8000'
  : `http://${window.location.hostname}:8000`;

interface SessionExecutionPanelProps {
  session: SessionSummary;
  trades: Trade[];
}

const cardStyle: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  padding: '1rem',
};

const SessionExecutionPanel: React.FC<SessionExecutionPanelProps> = ({ session, trades }) => {
  const [run, setRun] = useState<SimulationRun | null>(null);
  const [steps, setSteps] = useState<SimulationStep[]>([]);

  useEffect(() => {
    let interval: number | undefined;

    const fetchExecution = async () => {
      if (!session.run_id) {
        setRun(null);
        setSteps([]);
        return;
      }

      try {
        const [runResp, stepsResp] = await Promise.all([
          fetch(`${API_BASE}/simulation-runs/${session.run_id}`),
          fetch(`${API_BASE}/simulation-runs/${session.run_id}/steps?limit=150`),
        ]);

        if (runResp.ok) {
          setRun(await runResp.json());
        }
        if (stepsResp.ok) {
          setSteps(await stepsResp.json());
        }
      } catch (error) {
        console.error('Failed to fetch execution details', error);
      }
    };

    fetchExecution();
    if (session.status === 'running' && session.run_id) {
      interval = window.setInterval(fetchExecution, 3000);
    }

    return () => {
      if (interval) clearInterval(interval);
    };
  }, [session.run_id, session.status]);

  const recentTrades = useMemo(() => [...trades].slice(-10).reverse(), [trades]);

  if (!session.run_id) {
    return (
      <div style={{ display: 'grid', gap: '1rem' }}>
        <section style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>Execution Summary</h3>
          <div className="tagline">该 session 没有关联模拟批次，当前仅展示运行摘要。</div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '1rem', marginTop: '1rem' }}>
            <div>
              <div className="tagline">状态</div>
              <div style={{ fontWeight: 700 }}>{formatStatusLabel(session.status)}</div>
            </div>
            <div>
              <div className="tagline">模式</div>
              <div style={{ fontWeight: 700 }}>{formatModeLabel(session.mode)}</div>
            </div>
            <div>
              <div className="tagline">来源</div>
              <div style={{ fontWeight: 700 }}>{formatSourceLabel(session.source)}</div>
            </div>
            <div>
              <div className="tagline">进度</div>
              <div style={{ fontWeight: 700 }}>{(session.progress || 0).toFixed(0)}%</div>
            </div>
          </div>
        </section>

        <section style={cardStyle}>
          <h3 style={{ marginTop: 0 }}>最近成交</h3>
          <div style={{ display: 'grid', gap: '0.75rem' }}>
            {recentTrades.map((trade) => (
              <div key={`${trade.timestamp}-${trade.symbol}-${trade.type}`} style={{ padding: '0.8rem', borderRadius: 12, background: 'rgba(255,255,255,0.03)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                  <strong>{trade.type.toUpperCase()} {trade.symbol}</strong>
                  <span className="tagline">{trade.timestamp}</span>
                </div>
                <div className="tagline">{trade.quantity} @ {formatPrice(trade.price, 2)} · 成交额 {trade.amount}</div>
              </div>
            ))}
            {recentTrades.length === 0 && <div className="tagline">暂无成交记录。</div>}
          </div>
        </section>
      </div>
    );
  }

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <section style={cardStyle}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem' }}>
          <div>
            <h3 style={{ marginTop: 0 }}>执行时间线</h3>
            <div className="tagline">展示模拟批次、逐 Bar 执行轨迹和触发来源。</div>
          </div>
          <div className="tagline">run {session.run_id?.slice(0, 8)}...</div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, minmax(0, 1fr))', gap: '1rem', marginTop: '1rem' }}>
          <div><div className="tagline">状态</div><div style={{ fontWeight: 700 }}>{formatStatusLabel(run?.status || session.status)}</div></div>
          <div><div className="tagline">进度</div><div style={{ fontWeight: 700 }}>{(run?.progress ?? session.progress ?? 0).toFixed(0)}%</div></div>
          <div><div className="tagline">Bar 数</div><div style={{ fontWeight: 700 }}>{run?.bars_processed ?? 0}</div></div>
          <div><div className="tagline">步骤数</div><div style={{ fontWeight: 700 }}>{run?.steps_recorded ?? steps.length}</div></div>
          <div><div className="tagline">触发来源</div><div style={{ fontWeight: 700 }}>{formatSourceLabel(run?.trigger_source || 'manual')}</div></div>
        </div>

        {session.last_processed_at && <div className="tagline" style={{ marginTop: '0.75rem' }}>最近处理到：{session.last_processed_at}</div>}
      </section>

      <section style={cardStyle}>
        <h3 style={{ marginTop: 0 }}>步骤轨迹</h3>
        <div style={{ display: 'grid', gap: '0.75rem', maxHeight: '70vh', overflowY: 'auto' }}>
          {steps.map((step) => (
            <div key={step.id} style={{ padding: '0.85rem', borderRadius: 12, background: 'rgba(255,255,255,0.03)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: '0.5rem' }}>
                <strong>{step.event_type}</strong>
                <span className="tagline">#{step.step_index} · {step.timestamp || step.created_at}</span>
              </div>
              {step.payload?.progress !== undefined && (
                <div className="tagline" style={{ marginTop: '0.35rem' }}>进度：{Number(step.payload.progress).toFixed(2)}% · 总权益：{step.payload.total_equity ?? '--'}</div>
              )}
              {step.payload?.close_prices && (
                <div className="tagline" style={{ marginTop: '0.35rem' }}>
                  行情：{Object.entries(step.payload.close_prices).map(([code, price]) => `${code}=${formatPrice(Number(price), 2)}`).join('，')}
                </div>
              )}
              {step.payload?.new_trades?.length > 0 && (
                <div className="tagline" style={{ marginTop: '0.35rem' }}>
                  成交：{step.payload.new_trades.map((trade: any) => `${trade.type} ${trade.symbol} @ ${formatPrice(trade.price, 2)}`).join('；')}
                </div>
              )}
              {step.payload?.error && <div style={{ marginTop: '0.35rem', color: 'var(--danger)' }}>{step.payload.error}</div>}
            </div>
          ))}
          {steps.length === 0 && <div className="tagline">暂无执行轨迹。</div>}
        </div>
      </section>
    </div>
  );
};

export default SessionExecutionPanel;
