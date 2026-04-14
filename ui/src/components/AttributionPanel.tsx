import React, { useEffect, useMemo, useState } from 'react';
import { apiFetch } from '../utils/api';
import { formatMoney, formatPercent, colorFromValue } from '../utils/format';
import { EmptyState } from './layout/EmptyState';
import { MetricCard } from './layout/MetricCard';
import { SectionCard } from './layout/SectionCard';

interface AttributionData {
    session_id: string;
    total_return: number;
    by_asset: Record<string, number>;
    by_sector: Record<string, number>;
    by_period: Record<string, number>;
    win_rate: number;
    avg_win: number;
    avg_loss: number;
    profit_factor: number;
}

interface Props {
    sessionId: string;
}

export const AttributionPanel: React.FC<Props> = ({ sessionId }) => {
    const [data, setData] = useState<AttributionData | null>(null);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        if (!sessionId) {
            setData(null);
            return;
        }

        const fetchAttribution = async () => {
            setLoading(true);
            try {
                const resp = await apiFetch(`/analysis/attribution/${sessionId}`);
                if (resp.ok) {
                    setData(await resp.json());
                } else {
                    setData(null);
                }
            } catch (e) {
                console.error(e);
                setData(null);
            } finally {
                setLoading(false);
            }
        };

        fetchAttribution();
    }, [sessionId]);

    const sortedAssets = useMemo(
        () => Object.entries(data?.by_asset || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])),
        [data],
    );
    const sortedSectors = useMemo(
        () => Object.entries(data?.by_sector || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])),
        [data],
    );
    const sortedPeriods = useMemo(
        () => Object.entries(data?.by_period || {}).sort((a, b) => a[0].localeCompare(b[0])),
        [data],
    );

    if (loading) {
        return (
            <SectionCard title="Attribution Analysis" description="Loading return sources, sector contributions, and time-based performance.">
                <div className="empty-state">Loading attribution analysis...</div>
            </SectionCard>
        );
    }

    if (!data) {
        return (
            <EmptyState
                title="No Attribution Data"
                description="Select a session with return data to view asset, sector, and time-based return attribution."
            />
        );
    }

    return (
        <div className="space-y-6">
            <SectionCard
                title="Attribution Analysis"
                description="View return sources, structural distribution, and monthly rhythm directly without generating report files."
            >
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                    <MetricCard
                        label="Total Return"
                        value={<span style={{ color: colorFromValue(data.total_return * 100) }}>{formatPercent(data.total_return, 2)}</span>}
                        hint="Overall return performance"
                    />
                    <MetricCard
                        label="Win Rate"
                        value={formatPercent(data.win_rate, 1)}
                        hint="Percentage of profitable trades"
                    />
                    <MetricCard
                        label="Profit Factor"
                        value={data.profit_factor.toFixed(2)}
                        hint="Ratio of gross profit to gross loss"
                    />
                    <MetricCard
                        label="Avg Win"
                        value={<span style={{ color: 'var(--color-success)' }}>{formatPercent(data.avg_win, 2)}</span>}
                        hint="Average profit per winning trade"
                    />
                    <MetricCard
                        label="Avg Loss"
                        value={<span style={{ color: 'var(--color-danger)' }}>{formatPercent(data.avg_loss, 2)}</span>}
                        hint="Average loss per losing trade"
                    />
                </div>
            </SectionCard>

            <div className="grid gap-6 xl:grid-cols-2">
                <SectionCard title="Asset Contribution" description="Identify the top return contributors and detractors.">
                    <div className="space-y-3">
                        {sortedAssets.slice(0, 8).map(([symbol, pnl]) => (
                            <ContributionBar key={symbol} label={symbol} value={pnl} entries={sortedAssets.map((item) => item[1])} />
                        ))}
                        {sortedAssets.length === 0 ? <div className="empty-state">No asset-level data available</div> : null}
                    </div>
                </SectionCard>

                <SectionCard title="Sector Contribution" description="View how each sector drives or drags overall returns.">
                    <div className="space-y-3">
                        {sortedSectors.map(([sector, pnl]) => (
                            <ContributionBar key={sector} label={sector} value={pnl} entries={sortedSectors.map((item) => item[1])} />
                        ))}
                        {sortedSectors.length === 0 ? <div className="empty-state">No sector-level data available</div> : null}
                    </div>
                </SectionCard>
            </div>

            <SectionCard title="Monthly Rhythm" description="View the return distribution aggregated by month.">
                <div className="flex flex-wrap gap-3">
                    {sortedPeriods.map(([month, ret]) => (
                        <div
                            key={month}
                            className="rounded-2xl border px-4 py-3"
                            style={{
                                borderColor: ret >= 0 ? 'color-mix(in srgb, var(--color-success) 18%, transparent)' : 'color-mix(in srgb, var(--color-danger) 18%, transparent)',
                                background: ret >= 0 ? 'color-mix(in srgb, var(--color-success) 8%, transparent)' : 'color-mix(in srgb, var(--color-danger) 8%, transparent)',
                                color: ret >= 0 ? 'var(--color-success)' : 'var(--color-danger)',
                            }}
                        >
                            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] opacity-80">{month}</div>
                            <div className="mt-1 text-base font-semibold tracking-[-0.02em]">{formatPercent(ret, 1)}</div>
                        </div>
                    ))}
                    {sortedPeriods.length === 0 ? <div className="empty-state">No monthly return data available</div> : null}
                </div>
            </SectionCard>
        </div>
    );
};

const ContributionBar: React.FC<{ label: string; value: number; entries: number[] }> = ({ label, value, entries }) => {
    const max = Math.max(...entries.map((item) => Math.abs(item)), 1);
    const width = `${Math.min((Math.abs(value) / max) * 100, 100)}%`;
    const positive = value >= 0;

    return (
        <div className="grid gap-2 sm:grid-cols-[120px_minmax(0,1fr)_88px] sm:items-center">
            <div className="truncate text-sm font-medium text-foreground">{label}</div>
            <div className="h-2 overflow-hidden rounded-full bg-secondary/90">
                <div
                    className="h-full rounded-full"
                    style={{
                        width,
                        background: positive ? 'var(--color-success)' : 'var(--color-danger)',
                    }}
                />
            </div>
            <div className="text-right text-sm font-semibold" style={{ color: positive ? 'var(--color-success)' : 'var(--color-danger)' }}>
                {formatMoney(value, { symbol: '¥' })}
            </div>
        </div>
    );
};

export default AttributionPanel;
