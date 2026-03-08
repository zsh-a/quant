import React from 'react';
import { SessionSummary } from '../types';

interface SessionListProps {
    sessions: SessionSummary[];
    selectedSessionIds: string[];
    onToggleSelection: (id: string) => void;
    onViewSession: (id: string) => void;
    onStopSession: (id: string) => void;
}

const SessionList: React.FC<SessionListProps> = ({ sessions, selectedSessionIds, onToggleSelection, onViewSession, onStopSession }) => {
    return (
        <div className="glass card">
            <h3 style={{ marginBottom: '1.5rem' }}>All Sessions</h3>
            <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th style={{ width: '40px' }}></th>
                            <th>ID</th>
                            <th>Strategy</th>
                            <th>Timeframe</th>
                            <th>Mode</th>
                            <th>Status</th>
                            <th>Source</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sessions.map(s => (
                            <tr key={s.id} style={{ backgroundColor: selectedSessionIds.includes(s.id) ? 'rgba(99, 102, 241, 0.05)' : 'transparent' }}>
                                <td>
                                    <input 
                                        type="checkbox" 
                                        checked={selectedSessionIds.includes(s.id)} 
                                        onChange={() => onToggleSelection(s.id)}
                                        style={{ cursor: 'pointer', width: '16px', height: '16px', accentColor: 'var(--primary)' }}
                                    />
                                </td>
                                <td style={{ fontFamily: 'monospace', fontSize: '0.8rem', color: 'var(--text-dim)' }}>{s.id.slice(0, 8)}...</td>
                                <td>
                                    <div style={{ fontWeight: 500 }}>{s.strategy}</div>
                                    <div className="tagline" style={{ fontSize: '0.7rem' }}>{s.symbol}</div>
                                </td>
                                <td style={{ fontSize: '0.8rem' }}>
                                    {s.start_date}<br />
                                    {s.end_date || 'Ongoing'}
                                </td>
                                <td><span className={`status-badge ${s.mode === 'live' ? 'status-live' : s.mode === 'simulation' ? 'status-backtest' : ''}`}>{s.mode}</span></td>
                                <td>
                                    {s.status}
                                    {s.status === 'running' && <span style={{ marginLeft: '0.5rem', fontSize: '0.7rem' }}>({s.progress.toFixed(0)}%)</span>}
                                </td>
                                <td>
                                    <span className="tagline">{s.source || 'manual'}</span>
                                    {s.job_id && <div className="tagline" style={{ fontSize: '0.65rem' }}>job {s.job_id.slice(0, 6)}</div>}
                                </td>
                                <td>
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <button 
                                            className="tagline" 
                                            style={{ padding: '0.3rem 0.8rem', background: 'rgba(255,255,255,0.1)', color: 'white', fontSize: '0.7rem', border: '1px solid rgba(255,255,255,0.1)' }} 
                                            onClick={() => onViewSession(s.id)}
                                        >
                                            View
                                        </button>
                                        {s.status === 'running' && (
                                            <button 
                                                className="tagline" 
                                                style={{ padding: '0.3rem 0.8rem', background: 'rgba(239, 68, 68, 0.2)', color: 'var(--danger)', fontSize: '0.7rem', border: '1px solid rgba(239, 68, 68, 0.2)' }} 
                                                onClick={() => onStopSession(s.id)}
                                            >
                                                Stop
                                            </button>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                        {sessions.length === 0 && <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '2rem' }}>No sessions found</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default SessionList;