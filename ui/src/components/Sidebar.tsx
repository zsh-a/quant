import React from 'react';
import {
    Activity,
    BarChart3,
    BriefcaseBusiness,
    Compass,
    Database,
    FlaskConical,
    LayoutDashboard,
    Sparkles,
    Zap,
} from 'lucide-react';
import { SessionSummary } from '../types';
import { Progress } from './ui/progress';
import { ScrollArea } from './ui/scroll-area';
import { StatusBadge } from './layout/StatusBadge';

interface SidebarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    activeSessions: SessionSummary[];
    onSessionSelect: (id: string) => void;
    hasSelectedSession: boolean;
}

const NAV_ITEMS = [
    { key: 'overview', label: 'Overview', icon: LayoutDashboard },
    { key: 'lab', label: 'Lab', icon: FlaskConical },
    { key: 'session', label: 'Session', icon: Compass },
    { key: 'comparison', label: 'Compare', icon: BarChart3 },
    { key: 'heatmap', label: 'Heatmap', icon: Activity },
    { key: 'portfolio', label: 'Portfolio', icon: BriefcaseBusiness },
    { key: 'marketAdmin', label: 'Market Data', icon: Database },
    { key: 'optimizer', label: 'Optimizer', icon: Sparkles },
] as const;

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, activeSessions, onSessionSelect, hasSelectedSession }) => {
    return (
        <nav className="flex h-full flex-col bg-sidebar-background">
            {/* Brand */}
            <div className="flex items-center gap-2.5 border-b border-sidebar-border px-4 py-3.5">
                <div className="flex size-7 items-center justify-center rounded bg-primary/15 text-primary">
                    <Zap className="size-3.5" />
                </div>
                <div>
                    <div className="text-sm font-semibold text-sidebar-foreground tracking-tight">Quent</div>
                </div>
            </div>

            {/* Nav */}
            <div className="space-y-px px-2 py-2">
                {NAV_ITEMS.map((item) => {
                    const Icon = item.icon;
                    const disabled = item.key === 'session' && !hasSelectedSession;
                    const active = activeTab === item.key;

                    return (
                        <button
                            key={item.key}
                            type="button"
                            onClick={disabled ? undefined : () => onTabChange(item.key)}
                            disabled={disabled}
                            className={[
                                'flex w-full items-center gap-2.5 rounded-md px-3 py-1.5 text-left text-[13px] font-medium transition-colors',
                                active
                                    ? 'bg-sidebar-accent text-sidebar-primary'
                                    : 'text-muted-foreground hover:bg-sidebar-accent hover:text-sidebar-foreground',
                                disabled ? 'cursor-not-allowed opacity-40' : '',
                            ].join(' ')}
                        >
                            <Icon className="size-3.5 shrink-0" />
                            <span className="flex-1">{item.label}</span>
                            {active && <div className="size-1 rounded-full bg-primary" />}
                        </button>
                    );
                })}
            </div>

            {/* Live Queue */}
            <div className="mt-auto flex min-h-0 flex-1 flex-col border-t border-sidebar-border">
                <div className="flex items-center justify-between px-4 py-2.5">
                    <span className="text-[11px] font-medium uppercase tracking-[0.06em] text-muted-foreground">Live Queue</span>
                    <span className="rounded bg-primary/15 px-1.5 py-px text-[10px] font-semibold tabular-nums text-primary">
                        {activeSessions.length}
                    </span>
                </div>
                <ScrollArea className="min-h-0 flex-1 px-2 pb-2">
                    <div className="space-y-1">
                        {activeSessions.slice(0, 8).map((s) => (
                            <button
                                key={s.id}
                                type="button"
                                onClick={() => onSessionSelect(s.id)}
                                className="w-full rounded-md border border-transparent bg-sidebar-accent/50 p-2.5 text-left transition-colors hover:border-sidebar-border hover:bg-sidebar-accent"
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div className="min-w-0">
                                        <div className="truncate text-[13px] font-medium text-sidebar-foreground">{s.strategy}</div>
                                        <div className="mt-0.5 truncate text-[11px] text-muted-foreground">{s.symbol}</div>
                                    </div>
                                    <StatusBadge value={s.mode} />
                                </div>
                                <div className="mt-2 space-y-1">
                                    <div className="flex items-center justify-between text-[11px] text-muted-foreground">
                                        <span>{s.status}</span>
                                        <span className="tabular-nums">{(s.progress || 0).toFixed(0)}%</span>
                                    </div>
                                    <Progress value={s.progress || 0} />
                                </div>
                            </button>
                        ))}
                        {activeSessions.length === 0 && (
                            <div className="rounded-md border border-dashed border-sidebar-border p-3 text-center text-[12px] text-muted-foreground">
                                No active sessions
                            </div>
                        )}
                    </div>
                </ScrollArea>
            </div>
        </nav>
    );
};

export default Sidebar;
