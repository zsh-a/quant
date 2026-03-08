import React, { useEffect, useMemo, useState } from 'react';
import { SessionSummary } from '../types';
import { SectionCard } from './layout/SectionCard';
import { StatusBadge } from './layout/StatusBadge';
import { Button } from './ui/button';
import { Progress } from './ui/progress';
import { formatSourceLabel } from '../utils/display';

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

const SessionList: React.FC<SessionListProps> = ({
    sessions,
    selectedSessionIds,
    onToggleSelection,
    onViewSession,
    onStopSession,
    title = '全部会话',
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
        <SectionCard
            title={title}
            description="查看会话记录、选择对比对象，并快速回到执行详情。"
            action={
                <div className="flex flex-wrap gap-2">
                    {([
                        ['all', `全部 (${counts.all})`],
                        ['manual', `手动 (${counts.manual})`],
                        ['simulation', `模拟 (${counts.simulation})`],
                        ['running', `运行中 (${counts.running})`],
                    ] as Array<[SessionFilter, string]>).map(([value, label]) => (
                        <Button
                            key={value}
                            variant={filter === value ? 'default' : 'outline'}
                            size="sm"
                            onClick={() => setFilter(value)}
                        >
                            {label}
                        </Button>
                    ))}
                </div>
            }
        >
            <div className="overflow-x-auto">
                <table className="data-table">
                    <thead>
                        <tr>
                            <th style={{ width: '40px' }}></th>
                            <th>ID</th>
                            <th>策略</th>
                            <th>时间区间</th>
                            <th>模式</th>
                            <th>状态</th>
                            <th>来源</th>
                            <th>操作</th>
                        </tr>
                    </thead>
                    <tbody>
                        {filteredSessions.map(s => (
                            <tr key={s.id} style={{ backgroundColor: selectedSessionIds.includes(s.id) ? 'rgba(34, 211, 238, 0.08)' : 'transparent' }}>
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
                                    <div style={{ fontWeight: 600 }}>{s.strategy}</div>
                                    <div className="tagline" style={{ fontSize: '0.7rem' }}>{s.symbol}</div>
                                </td>
                                <td style={{ fontSize: '0.8rem' }}>
                                    {s.start_date}<br />
                                    {s.end_date || '进行中'}
                                </td>
                                <td><StatusBadge value={s.mode} /></td>
                                <td>
                                    <div className="space-y-2">
                                        <StatusBadge value={s.status} />
                                        {s.status === 'running' && (
                                            <div className="space-y-1">
                                                <div className="text-xs text-muted-foreground">{s.progress.toFixed(0)}%</div>
                                                <Progress value={s.progress} />
                                            </div>
                                        )}
                                    </div>
                                </td>
                                <td>
                                    <span className="tagline">{formatSourceLabel(s.source)}</span>
                                    {s.job_id && <div className="tagline" style={{ fontSize: '0.65rem' }}>任务 {s.job_id.slice(0, 6)}</div>}
                                </td>
                                <td>
                                    <div className="flex gap-2">
                                        <Button variant="outline" size="sm" onClick={() => onViewSession(s.id)}>
                                            查看
                                        </Button>
                                        {s.status === 'running' && (
                                            <Button variant="danger" size="sm" onClick={() => onStopSession(s.id)}>
                                                停止
                                            </Button>
                                        )}
                                    </div>
                                </td>
                            </tr>
                        ))}
                        {filteredSessions.length === 0 && <tr><td colSpan={8} style={{ textAlign: 'center', color: 'var(--text-dim)', padding: '2rem' }}>未找到会话</td></tr>}
                    </tbody>
                </table>
            </div>
        </SectionCard>
    );
};

export default SessionList;
