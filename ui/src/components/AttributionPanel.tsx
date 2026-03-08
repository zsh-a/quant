import React, { useEffect, useMemo, useState } from 'react';
import { API_BASE } from '../utils/api';
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
                const resp = await fetch(`${API_BASE}/analysis/attribution/${sessionId}`);
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
            <SectionCard title="归因分析" description="正在加载收益来源、行业贡献和时间维度表现。">
                <div className="empty-state">归因分析加载中...</div>
            </SectionCard>
        );
    }

    if (!data) {
        return (
            <EmptyState
                title="暂无归因分析"
                description="选择一个已有收益数据的会话后，这里会展示资产、行业与时间维度的收益归因。"
            />
        );
    }

    return (
        <div className="space-y-6">
            <SectionCard
                title="归因分析"
                description="直接在前端查看收益来源、结构分布和月度节奏，无需生成额外报告文件。"
            >
                <div className="grid gap-4 md:grid-cols-2 xl:grid-cols-5">
                    <MetricCard
                        label="总收益"
                        value={<span style={{ color: colorFromValue(data.total_return * 100) }}>{formatPercent(data.total_return, 2)}</span>}
                        hint="整体收益表现"
                    />
                    <MetricCard
                        label="胜率"
                        value={formatPercent(data.win_rate, 1)}
                        hint="盈利交易占比"
                    />
                    <MetricCard
                        label="盈亏比"
                        value={data.profit_factor.toFixed(2)}
                        hint="盈利与亏损的效率比"
                    />
                    <MetricCard
                        label="平均盈利"
                        value={<span style={{ color: 'var(--success)' }}>{formatPercent(data.avg_win, 2)}</span>}
                        hint="单笔盈利均值"
                    />
                    <MetricCard
                        label="平均亏损"
                        value={<span style={{ color: 'var(--danger)' }}>{formatPercent(data.avg_loss, 2)}</span>}
                        hint="单笔亏损均值"
                    />
                </div>
            </SectionCard>

            <div className="grid gap-6 xl:grid-cols-2">
                <SectionCard title="资产贡献" description="识别最主要的收益来源与拖累资产。">
                    <div className="space-y-3">
                        {sortedAssets.slice(0, 8).map(([symbol, pnl]) => (
                            <ContributionBar key={symbol} label={symbol} value={pnl} entries={sortedAssets.map((item) => item[1])} />
                        ))}
                        {sortedAssets.length === 0 ? <div className="empty-state">暂无资产维度数据</div> : null}
                    </div>
                </SectionCard>

                <SectionCard title="行业贡献" description="查看不同行业对整体收益的推动或拖累。">
                    <div className="space-y-3">
                        {sortedSectors.map(([sector, pnl]) => (
                            <ContributionBar key={sector} label={sector} value={pnl} entries={sortedSectors.map((item) => item[1])} />
                        ))}
                        {sortedSectors.length === 0 ? <div className="empty-state">暂无行业维度数据</div> : null}
                    </div>
                </SectionCard>
            </div>

            <SectionCard title="月度节奏" description="快速查看按月份聚合后的收益分布。">
                <div className="flex flex-wrap gap-3">
                    {sortedPeriods.map(([month, ret]) => (
                        <div
                            key={month}
                            className="rounded-2xl border px-4 py-3"
                            style={{
                                borderColor: ret >= 0 ? 'rgba(52,211,153,0.18)' : 'rgba(251,113,133,0.18)',
                                background: ret >= 0 ? 'rgba(52,211,153,0.08)' : 'rgba(251,113,133,0.08)',
                                color: ret >= 0 ? '#86efac' : '#fda4af',
                            }}
                        >
                            <div className="text-[11px] font-semibold uppercase tracking-[0.16em] opacity-80">{month}</div>
                            <div className="mt-1 text-base font-semibold tracking-[-0.02em]">{formatPercent(ret, 1)}</div>
                        </div>
                    ))}
                    {sortedPeriods.length === 0 ? <div className="empty-state">暂无月度收益数据</div> : null}
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
                        background: positive ? 'linear-gradient(90deg, #34d399, #22c55e)' : 'linear-gradient(90deg, #fb7185, #ef4444)',
                    }}
                />
            </div>
            <div className="text-right text-sm font-semibold" style={{ color: positive ? 'var(--success)' : 'var(--danger)' }}>
                {formatMoney(value, { symbol: '¥' })}
            </div>
        </div>
    );
};

export default AttributionPanel;
