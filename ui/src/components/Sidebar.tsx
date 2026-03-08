import React from 'react';
import { SessionSummary } from '../types';

interface SidebarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
    activeSessions: SessionSummary[];
    onSessionSelect: (id: string) => void;
}

const NavItem: React.FC<{ icon: string, label: string, active: boolean, onClick: () => void }> = ({ icon, label, active, onClick }) => (
    <div
        className={`nav-item ${active ? 'active' : ''}`}
        onClick={onClick}
        style={{
            padding: '0.75rem 1rem',
            borderRadius: '12px',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            gap: '0.75rem',
            backgroundColor: active ? 'rgba(99, 102, 241, 0.1)' : 'transparent',
            color: active ? 'var(--primary)' : 'var(--text-dim)',
            fontWeight: active ? 700 : 500,
            transition: 'all 0.2s'
        }}
    >
        <span>{icon}</span>
        {label}
    </div>
);

const Sidebar: React.FC<SidebarProps> = ({ activeTab, onTabChange, activeSessions, onSessionSelect }) => {
    return (
        <nav className="glass sidebar">
            <div className="logo" style={{ fontSize: '1.5rem', fontWeight: 900, marginBottom: '2rem' }}>
                QUENT<span style={{ color: 'var(--primary)' }}>AI</span>
            </div>
            <div className="nav-items">
                <NavItem icon="📊" label="Dashboard" active={activeTab === 'dashboard'} onClick={() => onTabChange('dashboard')} />
                <NavItem icon="🔥" label="Sector Heatmap" active={activeTab === 'heatmap'} onClick={() => onTabChange('heatmap')} />
                <NavItem icon="🧪" label="Lab & Sessions" active={activeTab === 'lab'} onClick={() => onTabChange('lab')} />
                <NavItem icon="🤖" label="Automation" active={activeTab === 'automation'} onClick={() => onTabChange('automation')} />
                <NavItem icon="⚖️" label="Analysis" active={activeTab === 'analysis'} onClick={() => onTabChange('analysis')} />
                <NavItem icon="🛡️" label="Risk Monitor" active={activeTab === 'risk'} onClick={() => onTabChange('risk')} />
                <NavItem icon="📁" label="Portfolio" active={activeTab === 'portfolio'} onClick={() => onTabChange('portfolio')} />
                <NavItem icon="🎯" label="Optimizer" active={activeTab === 'optimizer'} onClick={() => onTabChange('optimizer')} />
                <NavItem icon="📈" label="Attribution" active={activeTab === 'attribution'} onClick={() => onTabChange('attribution')} />
                <NavItem icon="📋" label="Logs" active={activeTab === 'logs'} onClick={() => onTabChange('logs')} />
            </div>

            <div style={{ marginTop: 'auto' }}>
                <div className="tagline">Active Sessions ({activeSessions.length})</div>
                {activeSessions.slice(0, 5).map(s => (
                    <div
                        key={s.id}
                        onClick={() => onSessionSelect(s.id)}
                        style={{
                            fontSize: '0.8rem',
                            padding: '0.5rem',
                            background: 'rgba(255,255,255,0.05)',
                            borderRadius: '4px',
                            marginBottom: '0.5rem',
                            cursor: 'pointer',
                            transition: 'background 0.2s'
                        }}
                        onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.1)'}
                        onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255,255,255,0.05)'}
                    >
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                            <span>{s.strategy}</span>
                            <span className={`status-badge ${s.mode === 'live' ? 'status-live' : 'status-backtest'}`}>{s.mode}</span>
                        </div>
                        <div style={{ height: '4px', background: 'rgba(255,255,255,0.1)', marginTop: '4px', borderRadius: '2px' }}>
                            <div style={{ width: `${s.progress}%`, height: '100%', background: 'var(--primary)' }}></div>
                        </div>
                    </div>
                ))}
            </div>
        </nav>
    );
};

export default Sidebar;
