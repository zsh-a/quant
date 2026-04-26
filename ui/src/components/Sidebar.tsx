import React from 'react';
import {
    Activity,
    BarChart3,
    BriefcaseBusiness,
    Compass,
    Database,
    FlaskConical,
    LayoutDashboard,
    LineChart,
    Moon,
    Sparkles,
    Sun,
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
    theme: 'light' | 'dark';
    onToggleTheme: () => void;
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
    { key: 'trade', label: 'Trade', icon: LineChart },
] as const;

const SidebarInner: React.FC<SidebarProps> = ({ activeTab, onTabChange, activeSessions, onSessionSelect, hasSelectedSession, theme, onToggleTheme }) => {
    return (
        <nav className="flex h-full flex-col bg-sidebar-background">
            {/* Brand */}
            <div className="flex items-center gap-3 border-b border-sidebar-border/50 px-5 py-4">
                <div className="flex size-8 items-center justify-center rounded-lg bg-primary/12 text-primary">
                    <Zap className="size-4" />
                </div>
                <div className="flex-1">
                    <div className="text-[15px] font-semibold text-sidebar-foreground tracking-tight">Quent</div>
                </div>
                <button
                    type="button"
                    onClick={onToggleTheme}
                    className="flex size-8 cursor-pointer items-center justify-center rounded-lg text-muted-foreground transition-all duration-200 hover:bg-sidebar-accent hover:text-sidebar-foreground"
                    title={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'}
                >
                    {theme === 'dark' ? <Sun className="size-4" /> : <Moon className="size-4" />}
                </button>
            </div>

            {/* Nav */}
            <div className="space-y-0.5 px-3 py-3">
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
                                'flex w-full items-center gap-3 rounded-lg px-3 py-2 text-left text-[13px] font-medium transition-all duration-200',
                                active
                                    ? 'bg-sidebar-accent text-sidebar-primary'
                                    : 'text-muted-foreground hover:bg-sidebar-accent/70 hover:text-sidebar-foreground',
                                disabled ? 'cursor-not-allowed opacity-40' : 'cursor-pointer',
                            ].join(' ')}
                        >
                            <Icon className="size-4 shrink-0" />
                            <span className="flex-1">{item.label}</span>
                            {active && <div className="size-1.5 rounded-full bg-primary" />}
                        </button>
                    );
                })}
            </div>

            {/* Live Queue */}
            <div className="mt-auto flex min-h-0 flex-1 flex-col border-t border-sidebar-border/50">
                <div className="flex items-center justify-between px-5 py-3">
                    <span className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Live Queue</span>
                    <span className="rounded-full bg-primary/12 px-2 py-0.5 text-[10px] font-semibold tabular-nums text-primary">
                        {activeSessions.length}
                    </span>
                </div>
                <ScrollArea className="min-h-0 flex-1 px-3 pb-3">
                    <div className="space-y-1.5">
                        {activeSessions.slice(0, 8).map((s) => (
                            <button
                                key={s.id}
                                type="button"
                                onClick={() => onSessionSelect(s.id)}
                                className="w-full cursor-pointer rounded-lg border border-transparent bg-sidebar-accent/40 p-3 text-left transition-colors duration-200 hover:border-sidebar-border/50 hover:bg-sidebar-accent"
                            >
                                <div className="flex items-start justify-between gap-2">
                                    <div className="min-w-0">
                                        <div className="truncate text-[13px] font-medium text-sidebar-foreground">{s.strategy}</div>
                                        <div className="mt-0.5 truncate text-xs text-muted-foreground">{s.symbol}</div>
                                    </div>
                                    <StatusBadge value={s.mode} />
                                </div>
                                <div className="mt-2.5 space-y-1.5">
                                    <div className="flex items-center justify-between text-xs text-muted-foreground">
                                        <span>{s.status}</span>
                                        <span className="tabular-nums">{(s.progress || 0).toFixed(0)}%</span>
                                    </div>
                                    <Progress value={s.progress || 0} />
                                </div>
                            </button>
                        ))}
                        {activeSessions.length === 0 && (
                            <div className="rounded-lg border border-dashed border-sidebar-border/40 p-4 text-center text-xs text-muted-foreground">
                                No active sessions
                            </div>
                        )}
                    </div>
                </ScrollArea>
            </div>
        </nav>
    );
};

const Sidebar = React.memo(SidebarInner);

export default Sidebar;
