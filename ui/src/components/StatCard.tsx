import React from 'react';
import { colorFromSign } from '../utils/format';
import { MetricCard } from './layout/MetricCard';

interface StatCardProps {
    label: string;
    value: string;
    delta?: string;
    subtext?: string;
}

const StatCard: React.FC<StatCardProps> = ({ label, value, delta, subtext }) => {
    const valueColor = colorFromSign(value);
    const deltaColor = delta ? colorFromSign(delta) : undefined;
    return (
        <MetricCard
            label={label}
            value={<span style={{ color: valueColor === 'var(--text-dim)' ? 'inherit' : valueColor }}>{value}</span>}
            trend={
                delta ? (
                    <span style={{ color: deltaColor }}>
                        {delta}
                    </span>
                ) : undefined
            }
            hint={subtext}
        />
    );
};

export default StatCard;
