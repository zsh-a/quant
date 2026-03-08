import React from 'react';
import type { SessionSummary } from '../types';

interface GlobalOverviewProps {
  sessions: SessionSummary[];
  activeSessions: SessionSummary[];
  primarySession?: SessionSummary;
  onOpenSession: (sessionId: string) => void;
  onOpenLab: () => void;
}

const cardStyle: React.CSSProperties = {
  background: 'var(--glass-bg)',
  border: '1px solid rgba(255,255,255,0.08)',
  borderRadius: 16,
  padding: '1rem',
};

const GlobalOverview: React.FC<GlobalOverviewProps> = ({
  sessions,
  activeSessions,
  primarySession,
  onOpenSession,
  onOpenLab,
}) => {
  const completedSessions = sessions.filter((session) => session.status === 'completed');
  const simulationSessions = sessions.filter((session) => session.source === 'automation');
  const recentSessions = sessions.slice(0, 8);

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: '1rem' }}>
        <div style={cardStyle}>
          <div className="tagline">Total Sessions</div>
          <div style={{ fontSize: '2rem', fontWeight: 800 }}>{sessions.length}</div>
        </div>
        <div style={cardStyle}>
          <div className="tagline">Running Now</div>
          <div style={{ fontSize: '2rem', fontWeight: 800, color: 'var(--primary)' }}>{activeSessions.length}</div>
        </div>
        <div style={cardStyle}>
          <div className="tagline">Completed</div>
          <div style={{ fontSize: '2rem', fontWeight: 800 }}>{completedSessions.length}</div>
        </div>
        <div style={cardStyle}>
          <div className="tagline">Simulation Sessions</div>
          <div style={{ fontSize: '2rem', fontWeight: 800 }}>{simulationSessions.length}</div>
        </div>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: primarySession ? '1.2fr 1fr' : '1fr', gap: '1rem' }}>
        <section style={cardStyle}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
            <div>
              <h2 style={{ margin: 0 }}>Platform Overview</h2>
              <p className="tagline" style={{ marginTop: '0.4rem' }}>从这里进入 Lab、打开最近 session，或查看当前运行情况。</p>
            </div>
            <button className="btn-primary" onClick={onOpenLab}>打开 Lab</button>
          </div>

          <div style={{ display: 'grid', gap: '0.75rem' }}>
            {recentSessions.map((session) => (
              <div key={session.id} style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                gap: '1rem',
                padding: '0.9rem',
                borderRadius: 12,
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}>
                <div>
                  <div style={{ fontWeight: 700 }}>{session.strategy}</div>
                  <div className="tagline">{session.symbol} · {session.mode} · {session.source || 'manual'}</div>
                  <div className="tagline">{session.start_date} → {session.end_date || 'Ongoing'}</div>
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <div style={{ textAlign: 'right' }}>
                    <div style={{ fontWeight: 700 }}>{session.status}</div>
                    <div className="tagline">{session.progress?.toFixed?.(0) ?? session.progress}%</div>
                  </div>
                  <button className="btn-ghost" onClick={() => onOpenSession(session.id)}>查看详情</button>
                </div>
              </div>
            ))}
            {recentSessions.length === 0 && (
              <div className="tagline">还没有 session，去 Lab 创建第一个运行任务。</div>
            )}
          </div>
        </section>

        {primarySession && (
          <section style={cardStyle}>
            <div className="tagline">Current Session</div>
            <h3 style={{ marginTop: '0.4rem', marginBottom: '0.75rem' }}>{primarySession.strategy}</h3>
            <div className="tagline">{primarySession.symbol} · {primarySession.mode}</div>
            <div className="tagline">状态：{primarySession.status}</div>
            <div className="tagline">来源：{primarySession.source || 'manual'}</div>
            {primarySession.last_processed_at && (
              <div className="tagline">最近处理到：{primarySession.last_processed_at}</div>
            )}
            <div style={{ marginTop: '1rem', height: 6, background: 'rgba(255,255,255,0.08)', borderRadius: 999 }}>
              <div style={{ width: `${primarySession.progress || 0}%`, height: '100%', background: 'var(--primary)', borderRadius: 999 }} />
            </div>
            <button className="btn-primary" style={{ marginTop: '1rem', width: '100%' }} onClick={() => onOpenSession(primarySession.id)}>
              打开 Session Detail
            </button>
          </section>
        )}
      </div>
    </div>
  );
};

export default GlobalOverview;
