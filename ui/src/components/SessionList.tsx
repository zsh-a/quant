import React, { useEffect, useMemo, useState } from 'react';
import { SessionSummary } from '../types';

type SessionFilter = 'all' | 'manual' | 'simulation' | 'running';

interface SessionListProps {
    sessions: SessionSummary[];
    selectedSessionIds: string[];
    onToggleSelection: (id: string) => void;
    onViewSession: (id: string) => void;
    onStopSession: (id: string) => void;
    title?: string;
    defaultFilter?: SessionFilter;
}

const filterStyle = (active: boolean): React.CSSProperties => ({
    padding: '0.35rem 0.8rem',
    borderRadius: 999,
    border: '1px solid rgba(255,255,255,0.1)',
    background: active ? 'var(--primary)' : 'rgba(255,255,255,0.04)',
    color: 'white',
    cursor: 'pointer',
    fontSize: '0.75rem',
});

const SessionList: React.FC<SessionListProps> = ({
    sessions,
    selectedSessionIds,
    onToggleSelection,
    onViewSession,
    onStopSession,
    title = 'All Sessions',
    defaultFilter = 'all',
}) => {
    const [filter, setFilter] = useState<SessionFilter>(defaultFilter);

    useEffect(() => {
        setFilter(defaultFilter);
    }, [defaultFilter, title]);

    const counts = useMemo(() => ({
        all: sessions.length,
        manual: sessions.filter((s) => s.source !== 'automation' && s.mode !== 'simulation').length,
        simulation: sessions.filter((s) => s.source === 'automation' || s.mode === 'simulation').length,
        running: sessions.filter((s) => s.status === 'running').length,
    }), [sessions]);

    const filteredSessions = useMemo(() => {
        switch (filter) {
            case 'manual':
                return sessions.filter((s) => s.source !== 'automation' && s.mode !== 'simulation');
            case 'simulation':
                return sessions.filter((s) => s.source === 'automation' || s.mode === 'simulation');
            case 'running':
                return sessions.filter((s) => s.status === 'running');
            default:
                return sessions;
        }
    }, [sessions, filter]);

    return (
        <div className="glass card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', gap: '1rem', flexWrap: 'wrap' }}>
                <h3 style={{ margin: 0 }}>{title}</h3>
                <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
                    <button style={filterStyle(filter === 'all')} onClick={() => setFilter('all')}>All ({counts.all})</button>
                    <button style={filterStyle(filter === 'manual')} onClick={() => setFilter('manual')}>Manual ({counts.manual})</button>
                    <button style={filterStyle(filter === 'simulation')} onClick={() => setFilter('simulation')}>Simulation ({counts.simulation})</button>
                    <button style={filterStyle(filter === 'running')} onClick={() => setFilter('running')}>Running ({counts.running})</button>
                </div>
            </div>

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
                        {filteredSessions.map(s => (
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
                        {filteredSessions.length === 0 && <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '2rem' }}>No sessions found</td></tr>}
                    </tbody>
                </table>
            </div>
        </div>
    );
};

export default SessionList;
