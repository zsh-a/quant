import React, { useState, useEffect, useRef, useCallback } from 'react';
import { LazyLog, ScrollFollow } from '@melloware/react-logviewer';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { API_BASE } from '../utils/api';

interface LogViewerProps {
    sessionId: string | null;
}

interface SessionLogEntry {
    timestamp: string;
    level: string;
    source: string;
    message: string;
    extra?: Record<string, unknown>;
}

export const StrategyLogViewer: React.FC<LogViewerProps> = ({ sessionId }) => {
    const [logs, setLogs] = useState<string>('等待会话日志...');
    const [filter, setFilter] = useState<string>('');
    const [levelFilter, setLevelFilter] = useState<string>('');
    const [sourceFilter, setSourceFilter] = useState<string>('');
    const [autoRefresh, setAutoRefresh] = useState<boolean>(true);
    const refreshInterval = 2000;
    const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

    const formatLogLines = (entries: SessionLogEntry[]) => {
        if (!entries.length) return '当前会话还没有日志。';

        return entries.map((entry) => {
            const level = `[${String(entry.level || 'INFO').padEnd(7, ' ')}]`;
            const source = `[${String(entry.source || 'system').padEnd(10, ' ')}]`;
            const extras = entry.extra && Object.keys(entry.extra).length > 0
                ? ` | ${Object.entries(entry.extra).map(([key, value]) => `${key}=${String(value)}`).join(' | ')}`
                : '';
            return `${entry.timestamp} ${level} ${source} ${entry.message}${extras}`;
        }).join('\n');
    };

    const fetchLogs = useCallback(async () => {
        if (!sessionId) {
            setLogs('尚未选择会话，请先从总览或实验室中打开一个会话。');
            return;
        }

        try {
            const params = new URLSearchParams();
            if (levelFilter) params.append('level', levelFilter);
            if (sourceFilter) params.append('source', sourceFilter);
            params.append('limit', '1000');
            params.append('format', 'json');

            const response = await fetch(`${API_BASE}/logs/${sessionId}?${params.toString()}`);
            if (response.ok) {
                const payload = await response.json();
                setLogs(formatLogLines(payload.logs || []));
            } else {
                setLogs(`获取日志失败：${response.statusText}`);
            }
        } catch (error) {
            setLogs(`异常：${error}`);
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
            setLogs('日志已清空。');
        } catch (error) {
            console.error('Failed to clear logs:', error);
        }
    };

    // Filter logs client-side if there's a text filter
    const filteredLogs = filter
        ? logs.split('\n').filter(line => line.toLowerCase().includes(filter.toLowerCase())).join('\n')
        : logs;

    return (
        <SectionCard
            title="策略日志"
            description="按级别、来源和关键词筛选会话日志。"
            action={<span style={{ color: 'var(--text-dim)', fontSize: '0.85rem' }}>会话：{sessionId || '未选择'}</span>}
        >
            <div style={filtersStyle}>
                <input
                    type="text"
                    placeholder="搜索日志..."
                    value={filter}
                    onChange={(e) => setFilter(e.target.value)}
                    style={inputStyle}
                />
                <select
                    value={levelFilter}
                    onChange={(e) => setLevelFilter(e.target.value)}
                    style={selectStyle}
                >
                    <option value="">全部级别</option>
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
                    <option value="">全部来源</option>
                    <option value="strategy">策略</option>
                    <option value="broker">Broker</option>
                    <option value="engine">Engine</option>
                </select>
                <label style={checkboxLabelStyle}>
                    <input
                        type="checkbox"
                        checked={autoRefresh}
                        onChange={(e) => setAutoRefresh(e.target.checked)}
                    />
                    自动刷新
                </label>
                <Button onClick={fetchLogs} variant="outline" size="sm">刷新</Button>
                <Button onClick={handleClearLogs} variant="danger" size="sm">清空</Button>
            </div>

            <div style={logContainerStyle}>
                <div style={{ height: '100%', minHeight: 'inherit' }}>
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
                                    height: '100%',
                                    backgroundColor: '#0d1117',
                                    color: '#c9d1d9',
                                    fontSize: '12px',
                                    fontFamily: 'IBM Plex Mono, JetBrains Mono, Consolas, Monaco, monospace',
                                }}
                                lineClassName="log-line"
                            />
                        )}
                    />
                </div>
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
        </SectionCard>
    );
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

const checkboxLabelStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '0.5rem',
    color: 'var(--text-dim)',
    fontSize: '0.9rem',
};

const logContainerStyle: React.CSSProperties = {
    minHeight: 'calc(100vh - 360px)',
    height: 'calc(100vh - 360px)',
    borderRadius: '8px',
    overflow: 'hidden',
    border: '1px solid var(--border)',
};

export default StrategyLogViewer;
