import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LazyLog, ScrollFollow } from '@melloware/react-logviewer';

interface LogViewerProps {
    sessionId: string | null;
}

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? "http://localhost:8000"
    : `http://${window.location.hostname}:8000`;

export const StrategyLogViewer: React.FC<LogViewerProps> = ({ sessionId }) => {
    const [logs, setLogs] = useState<string>('Waiting for session...');
    const [filter, setFilter] = useState<string>('');
    const [levelFilter, setLevelFilter] = useState<string>('');
    const [sourceFilter, setSourceFilter] = useState<string>('');
    const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
    const refreshInterval = 2000;
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const fetchLogs = useCallback(async () => {
        if (!sessionId) {
            setLogs('No session selected. Please choose a session from Overview or Lab.');
            return;
        }

        try {
            const params = new URLSearchParams();
            if (levelFilter) params.append('level', levelFilter);
            if (sourceFilter) params.append('source', sourceFilter);
            params.append('limit', '1000');

            const response = await fetch(`${API_BASE}/logs/${sessionId}?${params.toString()}`);
            if (response.ok) {
                const text = await response.text();
                setLogs(text || 'No logs yet for this session.');
            } else {
                setLogs(`Error fetching logs: ${response.statusText}`);
            }
        } catch (error) {
            setLogs(`Error: ${error}`);
        }
    }, [sessionId, levelFilter, sourceFilter]);

    useEffect(() => {
        fetchLogs();

        if (autoRefresh && sessionId) {
            intervalRef.current = setInterval(fetchLogs, refreshInterval);
        }

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current);
            }
        };
    }, [fetchLogs, autoRefresh, refreshInterval, sessionId]);

    const handleClearLogs = async () => {
        if (!sessionId) return;
        try {
            await fetch(`${API_BASE}/logs/${sessionId}`, { method: 'DELETE' });
            setLogs('Logs cleared.');
        } catch (error) {
            console.error('Failed to clear logs:', error);
        }
    };

    // Filter logs client-side if there's a text filter
    const filteredLogs = filter
        ? logs.split('\n').filter(line => line.toLowerCase().includes(filter.toLowerCase())).join('\n')
        : logs;

    return (
        <div style={containerStyle}>
            <div style={headerStyle}>
                <h2 style={{ margin: 0, fontSize: '1.2rem' }}>📋 策略调试日志</h2>
                <span style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>
                    Session: {sessionId || 'None'}
                </span>
            </div>

            {/* Filters */}
            <div style={filtersStyle}>
                <input
                    type="text"
                    placeholder="🔍 搜索日志..."
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    style={inputStyle}
                />
                <select
                    value={levelFilter}
                    onChange={(e) => setLevelFilter(e.target.value)}
                    style={selectStyle}
                >
                    <option value="">All Levels</option>
                    <option value="DEBUG">DEBUG</option>
                    <option value="INFO">INFO</option>
                    <option value="WARNING">WARNING</option>
                    <option value="ERROR">ERROR</option>
                </select>
                <select
                    value={sourceFilter}
                    onChange={(e) => setSourceFilter(e.target.value)}
                    style={selectStyle}
                >
                    <option value="">All Sources</option>
                    <option value="strategy">Strategy</option>
                    <option value="broker">Broker</option>
                    <option value="engine">Engine</option>
                </select>
                <label style={checkboxLabelStyle}>
                    <input
                        type="checkbox"
                        checked={autoRefresh}
                        onChange={(e) => setAutoRefresh(e.target.checked)}
                    />
                    Auto Refresh
                </label>
                <button onClick={fetchLogs} style={buttonStyle}>
                    🔄 刷新
                </button>
                <button onClick={handleClearLogs} style={{ ...buttonStyle, background: '#dc3545' }}>
                    🗑️ 清除
                </button>
            </div>

            {/* Log Viewer */}
            <div style={logContainerStyle}>
                <ScrollFollow
                    startFollowing={true}
                    render={({ follow, onScroll }) => (
                        <LazyLog
                            text={filteredLogs}
                            follow={follow}
                            onScroll={onScroll}
                            enableSearch={true}
                            caseInsensitive={true}
                            enableHotKeys={true}
                            selectableLines={true}
                            style={{
                                backgroundColor: '#0d1117',
                                color: '#c9d1d9',
                                fontSize: '12px',
                                fontFamily: 'JetBrains Mono, Consolas, Monaco, monospace',
                            }}
                            lineClassName="log-line"
                        />
                    )}
                />
            </div>

            <style>{`
                .log-line {
                    padding: 2px 8px;
                    border-bottom: 1px solid #21262d;
                }
                .log-line:hover {
                    background: #161b22 !important;
                }
            `}</style>
        </div>
    );
};

const containerStyle: React.CSSProperties = {
    display: 'flex',
    flexDirection: 'column',
    height: 'calc(100vh - 200px)',
    background: 'var(--glass-bg)',
    borderRadius: '16px',
    padding: '1rem',
    gap: '1rem',
};

const headerStyle: React.CSSProperties = {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
};

const filtersStyle: React.CSSProperties = {
    display: 'flex',
    gap: '0.75rem',
    alignItems: 'center',
    flexWrap: 'wrap',
};

const inputStyle: React.CSSProperties = {
    flex: 1,
    minWidth: '200px',
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    background: 'var(--input-bg)',
    color: 'var(--text)',
    fontSize: '0.9rem',
};

const selectStyle: React.CSSProperties = {
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    border: '1px solid var(--border)',
    background: 'var(--input-bg)',
    color: 'var(--text)',
    fontSize: '0.9rem',
    cursor: 'pointer',
};

const buttonStyle: React.CSSProperties = {
    padding: '0.5rem 1rem',
    borderRadius: '8px',
    border: 'none',
    background: 'var(--primary)',
    color: 'white',
    fontSize: '0.9rem',
    cursor: 'pointer',
    transition: 'opacity 0.2s',
};

const checkboxLabelStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    color: 'var(--text-dim)',
    fontSize: '0.9rem',
};

const logContainerStyle: React.CSSProperties = {
    flex: 1,
    borderRadius: '8px',
    overflow: 'hidden',
    border: '1px solid var(--border)',
};

export default StrategyLogViewer;
