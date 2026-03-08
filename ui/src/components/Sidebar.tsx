import React from 'react';
import {
    Activity,
    BarChart3,
    BriefcaseBusiness,
    Compass,
    FlaskConical,
    LayoutDashboard,
    Sparkles,
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
    { key: 'session', label: 'Session Detail', icon: Compass },
    { key: 'comparison', label: 'Comparison', icon: BarChart3 },
    { key: 'heatmap', label: 'Sector Heatmap', icon: Activity },
    { key: 'portfolio', label: 'Portfolio', icon: BriefcaseBusiness },
    { key: 'optimizer', label: 'Optimizer', icon: Sparkles },
] as const;

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, activeSessions, onSessionSelect, hasSelectedSession }) => {
    return (
        <nav className="glass flex h-full flex-col overflow-hidden p-4">
            <div className="rounded-[28px] border border-border/70 bg-gradient-to-br from-primary/12 via-card/90 to-card/70 p-5">
                <div className="flex items-center gap-3">
                    <div className="flex size-11 items-center justify-center rounded-2xl bg-primary/15 text-primary shadow-inner shadow-primary/10">
                        <Sparkles className="size-5" />
                    </div>
                    <div>
                        <div className="text-xs font-semibold uppercase tracking-[0.24em] text-primary/80">Codex-style UI</div>
                        <div className="text-xl font-semibold tracking-tight text-foreground">Quent Console</div>
                    </div>
                </div>
                <p className="mt-4 text-sm leading-6 text-muted-foreground">
                    Trading workflows, simulation runs and diagnostics in one unified operator shell.
                </p>
            </div>

            <div className="mt-5 space-y-1">
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
                                'flex w-full items-center gap-3 rounded-2xl px-4 py-3 text-left text-sm font-medium transition-all',
                                active
                                    ? 'bg-primary/12 text-primary shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]'
                                    : 'text-muted-foreground hover:bg-accent/70 hover:text-foreground',
                                disabled ? 'cursor-not-allowed opacity-45' : '',
                            ].join(' ')}
                        >
                            <Icon className="size-4" />
                            <span className="flex-1">{item.label}</span>
                            {active ? <div className="size-2 rounded-full bg-primary" /> : null}
                        </button>
                    );
                })}
            </div>

            <div className="mt-6 flex min-h-0 flex-1 flex-col rounded-[28px] border border-border/70 bg-secondary/35">
                <div className="flex items-center justify-between border-b border-border/70 px-4 py-4">
                    <div>
                        <div className="text-xs font-semibold uppercase tracking-[0.22em] text-muted-foreground">Live Queue</div>
                        <div className="mt-1 text-sm font-medium text-foreground">Active Sessions</div>
                    </div>
                    <div className="rounded-full bg-primary/12 px-3 py-1 text-xs font-semibold text-primary">
                        {activeSessions.length}
                    </div>
                </div>
                <ScrollArea className="min-h-0 flex-1 px-3 py-3">
                    <div className="space-y-3">
                        {activeSessions.slice(0, 8).map((s) => (
                            <button
                                key={s.id}
                                type="button"
                                onClick={() => onSessionSelect(s.id)}
                                className="w-full rounded-2xl border border-border/70 bg-card/70 p-3 text-left transition hover:border-primary/30 hover:bg-accent/60"
                            >
                                <div className="flex items-start justify-between gap-3">
                                    <div className="min-w-0">
                                        <div className="truncate text-sm font-semibold text-foreground">{s.strategy}</div>
                                        <div className="mt-1 truncate text-xs text-muted-foreground">{s.symbol}</div>
                                    </div>
                                    <StatusBadge value={s.mode} />
                                </div>
                                <div className="mt-3 space-y-2">
                                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                                        <span>{s.status}</span>
                                        <span>{(s.progress || 0).toFixed(0)}%</span>
                                    </div>
                                    <Progress value={s.progress || 0} />
                                </div>
                            </button>
                        ))}
                        {activeSessions.length === 0 ? (
                            <div className="rounded-2xl border border-dashed border-border/70 bg-card/40 p-4 text-sm text-muted-foreground">
                                No active sessions. Start a run from the Lab panel.
                            </div>
                        ) : null}
                    </div>
                </ScrollArea>
            </div>
        </nav>
    );
};

export default Sidebar;
