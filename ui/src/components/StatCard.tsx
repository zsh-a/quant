import React from 'react';

interface StatCardProps {
    label: string;
    value: string;
    delta?: string;
    subtext?: string;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, delta, subtext }) => (
    <div className="glass card">
        <div className="tagline">{label}</div>
        <div style={{ fontSize: '1.5rem', fontWeight: 800, marginTop: '0.5rem' }}>{value}</div>
        {delta && (
            <div style={{ 
                color: delta.startsWith('+') ? 'var(--success)' : (delta.startsWith('-') ? 'var(--danger)' : 'var(--text-dim)'), 
                fontSize: '0.75rem', 
                marginTop: '0.25rem', 
                fontWeight: 700 
            }}>
                {delta}
            </div>
        )}
        {subtext && <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', marginTop: '0.25rem' }}>{subtext}</div>}
    </div>
);

export default StatCard;
