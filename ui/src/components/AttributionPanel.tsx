import React, { useState, useEffect } from 'react';
import { formatMoney } from '../utils/format';

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

const API = 'http://localhost:8000';

interface Props {
    sessionId: string;
}

export const AttributionPanel: React.FC<Props> = ({ sessionId }) => {
    const [data, setData] = useState<AttributionData | null>(null);
    const [loading, setLoading] = useState(false);
    const [reportUrl, setReportUrl] = useState<string | null>(null);

    useEffect(() => {
        if (sessionId) fetchAttribution();
    }, [sessionId]);

    const fetchAttribution = async () => {
        setLoading(true);
        try {
            const resp = await fetch(`${API}/analysis/attribution/${sessionId}`);
            if (resp.ok) setData(await resp.json());
        } catch (e) {
            console.error(e);
        }
        setLoading(false);
    };

    const generateReport = async () => {
        const resp = await fetch(`${API}/analysis/report/${sessionId}?format=json`);
        if (resp.ok) {
            const d = await resp.json();
            setReportUrl(d.path);
        }
    };

    if (loading) return <div style={containerStyle}>加载中...</div>;
    if (!data) return <div style={containerStyle}>选择会话查看归因分析</div>;

    const sortedAssets = Object.entries(data.by_asset).sort((a, b) => b[1] - a[1]);
    const sortedSectors = Object.entries(data.by_sector).sort((a, b) => b[1] - a[1]);
    const sortedPeriods = Object.entries(data.by_period).sort();

    return (
        <div style={containerStyle}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <h2>📊 归因分析</h2>
                <button onClick={generateReport} style={btnStyle}>生成报告</button>
            </div>

            {/* Summary */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, margin: '16px 0' }}>
                <StatCard label="总收益" value={`${(data.total_return * 100).toFixed(2)}%`} positive={data.total_return >= 0} />
                <StatCard label="胜率" value={`${(data.win_rate * 100).toFixed(1)}%`} />
                <StatCard label="盈亏比" value={data.profit_factor.toFixed(2)} />
                <StatCard label="平均盈利" value={`${(data.avg_win * 100).toFixed(2)}%`} positive />
            </div>

            {/* By Asset */}
            <Section title="按资产">
                {sortedAssets.slice(0, 8).map(([symbol, pnl]) => (
                    <BarItem key={symbol} label={symbol} value={pnl} />
                ))}
            </Section>

            {/* By Sector */}
            <Section title="按行业">
                {sortedSectors.map(([sector, pnl]) => (
                    <BarItem key={sector} label={sector} value={pnl} />
                ))}
            </Section>

            {/* By Period */}
            <Section title="月度收益">
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8 }}>
                    {sortedPeriods.map(([month, ret]) => (
                        <div key={month} style={{
                            padding: '6px 10px', borderRadius: 4,
                            background: ret >= 0 ? 'rgba(40,167,69,0.2)' : 'rgba(220,53,69,0.2)',
                            color: ret >= 0 ? '#28a745' : '#dc3545',
                            fontSize: 12
                        }}>
                            {month}: {(ret * 100).toFixed(1)}%
                        </div>
                    ))}
                </div>
            </Section>

            {reportUrl && (
                <div style={{ marginTop: 16, padding: 12, background: '#252525', borderRadius: 8 }}>
                    ✅ 报告已生成: {reportUrl}
                </div>
            )}
        </div>
    );
};

const StatCard: React.FC<{ label: string; value: string; positive?: boolean }> = ({ label, value, positive }) => (
    <div style={{ background: '#252525', borderRadius: 8, padding: 12, textAlign: 'center' }}>
        <div style={{ fontSize: 11, color: '#888' }}>{label}</div>
        <div style={{ fontSize: 18, fontWeight: 600, color: positive ? '#28a745' : positive === false ? '#dc3545' : '#fff' }}>{value}</div>
    </div>
);

const Section: React.FC<{ title: string; children: React.ReactNode }> = ({ title, children }) => (
    <div style={{ marginBottom: 16 }}>
        <h4 style={{ margin: '0 0 8px', color: '#aaa', fontSize: 13 }}>{title}</h4>
        {children}
    </div>
);

const BarItem: React.FC<{ label: string; value: number }> = ({ label, value }) => {
    const max = 50000;
    const width = Math.min(Math.abs(value) / max * 100, 100);
    const positive = value >= 0;
    return (
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
            <span style={{ width: 80, fontSize: 12, color: '#888', overflow: 'hidden', textOverflow: 'ellipsis' }}>{label}</span>
            <div style={{ flex: 1, height: 8, background: '#333', borderRadius: 4, overflow: 'hidden' }}>
                <div style={{
                    width: `${width}%`, height: '100%',
                    background: positive ? '#28a745' : '#dc3545'
                }} />
            </div>
            <span style={{ width: 60, fontSize: 11, textAlign: 'right', color: positive ? '#28a745' : '#dc3545' }}>
                {formatMoney(value, { symbol: '¥' })}
            </span>
        </div>
    );
};

const containerStyle: React.CSSProperties = { padding: 20, background: '#1a1a1a', borderRadius: 12, color: '#fff' };
const btnStyle: React.CSSProperties = { background: '#667eea', border: 'none', padding: '8px 16px', borderRadius: 6, color: 'white', cursor: 'pointer' };

export default AttributionPanel;
