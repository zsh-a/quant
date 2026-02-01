import React, { useState } from 'react';
import { SessionSummary } from '../types';

interface SessionListProps {
    sessions: SessionSummary[];
    selectedSessionIds: string[];
    onToggleSelection: (id: string) => void;
    onStopSession: (id: string) => void;
}

const SessionList: React.FC<SessionListProps> = ({ sessions, selectedSessionIds, onToggleSelection, onStopSession }) => {
    return (
        <div className="glass card">
            <h3 style={{ marginBottom: '1.5rem' }}>All Sessions</h3>
            <div style={{ overflowX: 'auto' }}>
                <table className="data-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Strategy</th>
                            <th>Timeframe</th>
                            <th>Mode</th>
                            <th>Status</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody>
                        {sessions.map(s => (
                            <tr key={s.id} style={{ backgroundColor: selectedSessionIds.includes(s.id) ? 'rgba(99, 102, 241, 0.1)' : 'transparent' }}>
                                <td style={{ fontFamily: 'monospace', fontSize: '0.8rem' }}>{s.id.slice(0, 8)}...</td>
                                <td>
                                    {s.strategy}
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
                                    <div style={{ display: 'flex', gap: '0.5rem' }}>
                                        <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(255,255,255,0.1)', fontSize: '0.6rem' }} onClick={() => onToggleSelection(s.id)}>
                                            {selectedSessionIds.includes(s.id) ? 'Deselect' : 'Select'}
                                        </button>
                                        {s.status === 'running' && (
                                            <button className="tagline" style={{ padding: '0.2rem 0.5rem', background: 'rgba(239, 68, 68, 0.2)', color: 'var(--danger)', fontSize: '0.6rem' }} onClick={() => onStopSession(s.id)}>
                                                Stop
                                            </button>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                        {sessions.length === 0 && <tr><td colSpan={6} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '1rem' }}>No sessions found</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default SessionList;
