import React from 'react';
import { formatMoney } from '../utils/format';
import { API_BASE } from '../utils/api';
import { formatStatusLabel } from '../utils/display';

interface RiskMetric {
    label: string;
    value: number;
    limit: number;
    unit: string;
    status: 'safe' | 'warning' | 'danger';
}

interface RiskAlert {
    id: string;
    timestamp: string;
    severity: 'info' | 'warning' | 'error';
    message: string;
}

interface RiskPanelProps {
    sessionId: string;
}

export const RiskPanel: React.FC<RiskPanelProps> = ({ sessionId }) => {
    const [riskData, setRiskData] = React.useState<any>(null);
    const [alerts, setAlerts] = React.useState<RiskAlert[]>([]);
    const [loading, setLoading] = React.useState(true);

    React.useEffect(() => {
        const fetchRiskData = async () => {
            try {
                const response = await fetch(`${API_BASE}/session/${sessionId}/risk`);
                if (response.ok) {
                    const data = await response.json();
                    setRiskData(data);

                    // Convert risk alerts to UI format
                    if (data.alerts && data.alerts.length > 0) {
                        const formattedAlerts = data.alerts.map((alert: any, idx: number) => ({
                            id: `alert-${idx}`,
                            timestamp: alert.timestamp || new Date().toISOString(),
                            severity: alert.type === 'STOP_LOSS' || alert.type === 'DAILY_LOSS' ? 'error' : 'warning',
                            message: alert.message
                        }));
                        setAlerts(formattedAlerts);
                    }
                }
            } catch (error) {
                console.error('Failed to fetch risk data:', error);
            } finally {
                setLoading(false);
            }
        };

        fetchRiskData();
        const interval = setInterval(fetchRiskData, 5000); // Update every 5s
        return () => clearInterval(interval);
    }, [sessionId]);

    if (loading) {
        return (
            <div className="risk-panel loading">
                <div className="spinner" />
            </div>
        );
    }

    if (!riskData || !riskData.enabled) {
        return (
            <div className="risk-panel disabled">
                <p>当前会话未启用风险控制</p>
            </div>
        );
    }

    const metrics = riskData.metrics || {};
    const limits = riskData.limits || {};

    // Calculate risk metrics
    const riskMetrics: RiskMetric[] = [
        {
            label: '持仓数量',
            value: metrics.position_count || 0,
            limit: 10,
            unit: '',
            status: (metrics.position_count || 0) > 8 ? 'warning' : 'safe'
        },
        {
            label: '总暴露',
            value: (metrics.total_exposure_pct || 0) * 100,
            limit: (limits.max_total_position || 0.95) * 100,
            unit: '%',
            status: (metrics.total_exposure_pct || 0) > 0.85 ? 'warning' : 'safe'
        },
        {
            label: '最大单仓',
            value: (metrics.largest_position_pct || 0) * 100,
            limit: (limits.max_position_pct || 0.1) * 100,
            unit: '%',
            status: (metrics.largest_position_pct || 0) > 0.09 ? 'warning' : 'safe'
        },
        {
            label: '当日盈亏',
            value: (metrics.daily_pnl_pct || 0) * 100,
            limit: (limits.max_daily_loss_pct || 0.1) * 100,
            unit: '%',
            status: Math.abs(metrics.daily_pnl_pct || 0) > 0.08 ? 'warning' : 'safe'
        },
        {
            label: '最大回撤',
            value: (metrics.max_drawdown_pct || 0) * 100,
            limit: (limits.max_drawdown_pct || 0.2) * 100,
            unit: '%',
            status: (metrics.max_drawdown_pct || 0) > 0.15 ? 'danger' : 'safe'
        }
    ];

    return (
        <div className="risk-panel">
            <div className="risk-header">
                <h3>风险控制</h3>
                <span className={`status-badge ${riskData.status}`}>
                    {formatStatusLabel(riskData.status || 'running')}
                </span>
            </div>

            {/* Risk Metrics Grid */}
            <div className="risk-metrics">
                {riskMetrics.map((metric) => (
                    <div key={metric.label} className={`metric-card ${metric.status}`}>
                        <div className="metric-label">{metric.label}</div>
                        <div className="metric-value">
                            {metric.value.toFixed(1)}{metric.unit}
                            <span className="metric-limit">/ {metric.limit.toFixed(0)}{metric.unit}</span>
                        </div>
                        <div className="metric-bar">
                            <div
                                className="metric-fill"
                                style={{ width: `${Math.min((metric.value / metric.limit) * 100, 100)}%` }}
                            />
                        </div>
                    </div>
                ))}
            </div>

            {/* Capital Info */}
            <div className="capital-info">
                <div className="capital-item">
                    <span className="label">当前资金:</span>
                    <span className="value">{formatMoney(metrics.current_capital)}</span>
                </div>
                <div className="capital-item">
                    <span className="label">峰值资金:</span>
                    <span className="value">{formatMoney(metrics.peak_capital)}</span>
                </div>
            </div>

            {/* Alerts */}
            {alerts.length > 0 && (
                <div className="risk-alerts">
                    <h4>近期告警</h4>
                    <div className="alerts-list">
                        {alerts.slice(0, 5).map((alert) => (
                            <div key={alert.id} className={`alert-item ${alert.severity}`}>
                                <span className="alert-time">
                                    {new Date(alert.timestamp).toLocaleTimeString()}
                                </span>
                                <span className="alert-message">{alert.message}</span>
                            </div>
                        ))}
                    </div>
                </div>
            )}

            <style>{`
        .risk-panel {
          background: #1a1a1a;
          border-radius: 12px;
          padding: 20px;
          color: #fff;
        }

        .risk-header {
          display: flex;
          justify-content: space-between;
          align-items: center;
          margin-bottom: 20px;
        }

        .risk-header h3 {
          margin: 0;
          font-size: 18px;
          font-weight: 600;
        }

        .status-badge {
          padding: 4px 12px;
          border-radius: 12px;
          font-size: 12px;
          font-weight: 500;
          background: #2ecc71;
          color: #fff;
        }

        .risk-metrics {
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
          gap: 16px;
          margin-bottom: 20px;
        }

        .metric-card {
          background: #2a2a2a;
          border-radius: 8px;
          padding: 16px;
          border-left: 3px solid #2ecc71;
        }

        .metric-card.warning {
          border-left-color: #f39c12;
        }

        .metric-card.danger {
          border-left-color: #e74c3c;
        }

        .metric-label {
          font-size: 12px;
          color: #888;
          margin-bottom: 8px;
        }

        .metric-value {
          font-size: 24px;
          font-weight: 700;
          margin-bottom: 8px;
        }

        .metric-limit {
          font-size: 14px;
          color: #666;
          font-weight: 400;
          margin-left: 4px;
        }

        .metric-bar {
          height: 4px;
          background: #333;
          border-radius: 2px;
          overflow: hidden;
        }

        .metric-fill {
          height: 100%;
          background: linear-gradient(90deg, #2ecc71, #27ae60);
          transition: width 0.3s ease;
        }

        .metric-card.warning .metric-fill {
          background: linear-gradient(90deg, #f39c12, #e67e22);
        }

        .metric-card.danger .metric-fill {
          background: linear-gradient(90deg, #e74c3c, #c0392b);
        }

        .capital-info {
          display: flex;
          gap: 20px;
          padding: 16px;
          background: #2a2a2a;
          border-radius: 8px;
          margin-bottom: 20px;
        }

        .capital-item {
          flex: 1;
        }

        .capital-item .label {
          display: block;
          font-size: 12px;
          color: #888;
          margin-bottom: 4px;
        }

        .capital-item .value {
          display: block;
          font-size: 18px;
          font-weight: 600;
          color: #2ecc71;
        }

        .risk-alerts {
          margin-top: 20px;
        }

        .risk-alerts h4 {
          margin: 0 0 12px 0;
          font-size: 14px;
          font-weight: 600;
        }

        .alerts-list {
          display: flex;
          flex-direction: column;
          gap: 8px;
        }

        .alert-item {
          padding: 12px;
          background: #2a2a2a;
          border-radius: 6px;
          border-left: 3px solid #3498db;
          display: flex;
          gap: 12px;
          align-items: center;
        }

        .alert-item.warning {
          border-left-color: #f39c12;
        }

        .alert-item.error {
          border-left-color: #e74c3c;
        }

        .alert-time {
          font-size: 11px;
          color: #666;
          min-width: 70px;
        }

        .alert-message {
          font-size: 13px;
          flex: 1;
        }

        .loading, .disabled {
          text-align: center;
          padding: 40px;
          color: #666;
        }

        .spinner {
          animation: pulse 1.5s ease-in-out infinite;
        }

        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.5; }
        }
      `}</style>
        </div>
    );
};
