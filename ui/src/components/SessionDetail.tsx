import React, { useEffect, useState } from 'react';
import type { BenchmarkData, EquityPoint, Position, SessionSummary, Trade } from '../types';
import Dashboard from './Dashboard';
import SessionExecutionPanel from './SessionExecutionPanel';
import { RiskPanel } from './RiskPanel';
import { CheckpointList } from './CheckpointList';
import { AttributionPanel } from './AttributionPanel';
import { StrategyLogViewer } from './StrategyLogViewer';

interface SessionDetailProps {
  primarySession?: SessionSummary;
  allSessions: SessionSummary[];
  equityHistory: EquityPoint[];
  trades: Trade[];
  positions: Record<string, Position>;
  comparisonData: { id: string; name: string; data: EquityPoint[] }[];
  benchmarksData: Record<string, BenchmarkData[]>;
  selectedBenchmarks: string[];
  onToggleBenchmark: (code: string) => void;
  availableBenchmarks: { code: string; name: string }[];
  onSelectSession: (id: string) => void;
  onRestoreCheckpoint: () => void;
}

const tabStyle = (active: boolean): React.CSSProperties => ({
  padding: '0.5rem 1rem',
  borderRadius: 999,
  border: '1px solid rgba(255,255,255,0.1)',
  background: active ? 'var(--primary)' : 'rgba(255,255,255,0.04)',
  color: 'white',
  cursor: 'pointer',
});

const SessionDetail: React.FC<SessionDetailProps> = ({
  primarySession,
  allSessions,
  equityHistory,
  trades,
  positions,
  comparisonData,
  benchmarksData,
  selectedBenchmarks,
  onToggleBenchmark,
  availableBenchmarks,
  onSelectSession,
  onRestoreCheckpoint,
}) => {
  const [subtab, setSubtab] = useState<'overview' | 'execution' | 'risk' | 'analysis' | 'logs'>(() => {
    const stored = window.localStorage.getItem('quent.session.subtab');
    if (stored === 'execution' || stored === 'risk' || stored === 'analysis' || stored === 'logs') return stored;
    return 'overview';
  });

  useEffect(() => {
    window.localStorage.setItem('quent.session.subtab', subtab);
  }, [subtab]);

  if (!primarySession) {
    return (
      <div className="dashboard-view" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', height: '400px', color: 'var(--text-dim)' }}>
        <h2>No Session Selected</h2>
        <p>从 Overview 或 Lab 选择一个 session 查看详情。</p>
      </div>
    );
  }

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: '1rem', flexWrap: 'wrap' }}>
        <div>
          <h2 style={{ margin: 0 }}>Session Detail</h2>
          <div className="tagline" style={{ marginTop: '0.4rem' }}>
            {primarySession.strategy} · {primarySession.symbol} · {primarySession.mode} · {primarySession.source || 'manual'}
          </div>
          <div className="tagline">状态：{primarySession.status} · 进度：{(primarySession.progress || 0).toFixed(0)}%</div>
        </div>
        <div style={{ minWidth: 280 }}>
          <div className="tagline" style={{ marginBottom: '0.35rem' }}>切换 Session</div>
          <select className="glass-input" value={primarySession.id} onChange={(e) => onSelectSession(e.target.value)}>
            {allSessions.map((session) => (
              <option key={session.id} value={session.id}>
                {session.strategy} - {session.mode} ({session.id.slice(0, 6)}...)
              </option>
            ))}
          </select>
        </div>
      </div>

      <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
        <button style={tabStyle(subtab === 'overview')} onClick={() => setSubtab('overview')}>Overview</button>
        <button style={tabStyle(subtab === 'execution')} onClick={() => setSubtab('execution')}>Execution</button>
        <button style={tabStyle(subtab === 'risk')} onClick={() => setSubtab('risk')}>Risk</button>
        <button style={tabStyle(subtab === 'analysis')} onClick={() => setSubtab('analysis')}>Analysis</button>
        <button style={tabStyle(subtab === 'logs')} onClick={() => setSubtab('logs')}>Logs</button>
      </div>

      {subtab === 'overview' && (
        <Dashboard
          primarySession={primarySession}
          equityHistory={equityHistory}
          trades={trades}
          positions={positions}
          comparisonData={comparisonData}
          benchmarksData={benchmarksData}
          selectedBenchmarks={selectedBenchmarks}
          onToggleBenchmark={onToggleBenchmark}
          availableBenchmarks={availableBenchmarks}
          onSelectSession={onSelectSession}
          allSessions={allSessions}
        />
      )}

      {subtab === 'execution' && (
        <SessionExecutionPanel session={primarySession} trades={trades} />
      )}

      {subtab === 'risk' && (
        <div style={{ display: 'grid', gap: '1rem' }}>
          <RiskPanel sessionId={primarySession.id} />
          <CheckpointList sessionId={primarySession.id} onRestore={onRestoreCheckpoint} />
        </div>
      )}

      {subtab === 'analysis' && (
        <AttributionPanel sessionId={primarySession.id} />
      )}

      {subtab === 'logs' && (
        <StrategyLogViewer sessionId={primarySession.id} />
      )}
    </div>
  );
};

export default SessionDetail;
