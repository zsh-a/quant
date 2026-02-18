import React, { useState, useEffect } from 'react';
import { formatMoney } from '../utils/format';

interface Strategy {
    name: string;
    strategy: string;
    params: Record<string, unknown>;
    weight: number;
}

interface Portfolio {
    portfolio_id: string;
    name?: string;
    n_strategies: number;
    weights: Record<string, number>;
}

interface BacktestResult {
    portfolio_id: string;
    total_return: number;
    sharpe_ratio: number;
    max_drawdown: number;
    final_equity: number;
    strategy_results: Record<string, { final_equity: number; weight: number; trades: number }>;
    n_trades: number;
}

const API_BASE = window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : `${window.location.protocol}//${window.location.hostname}:8000`;

const AVAILABLE_STRATEGIES = [
    { id: 'jsg', name: 'JSG策略', description: '金叉死叉策略' },
    { id: 'rotation', name: '轮动策略', description: '行业轮动策略' },
];

const WEIGHT_METHODS = [
    { id: 'equal', name: '等权重', description: '每个策略相同权重' },
    { id: 'vol_inverse', name: '波动率倒数', description: '低波动策略权重更高' },
    { id: 'sharpe', name: '夏普比率', description: '按夏普比率分配权重' },
];

export const PortfolioManager: React.FC = () => {
    const [portfolios, setPortfolios] = useState<Portfolio[]>([]);
    const [selectedPortfolio, setSelectedPortfolio] = useState<string | null>(null);
    const [showCreateModal, setShowCreateModal] = useState(false);
    const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null);
    const [loading, setLoading] = useState(false);

    // Create portfolio form state
    const [newPortfolioName, setNewPortfolioName] = useState('');
    const [selectedStrategies, setSelectedStrategies] = useState<Strategy[]>([]);
    const [weightMethod, setWeightMethod] = useState('equal');

    // Backtest form state
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState('2024-01-01');
    const [symbols, setSymbols] = useState('sz.300750,sz.002475');
    const [initialCapital, setInitialCapital] = useState(1000000);

    useEffect(() => {
        fetchPortfolios();
    }, []);

    const fetchPortfolios = async () => {
        try {
            const resp = await fetch(`${API_BASE}/portfolio`);
            const data = await resp.json();
            setPortfolios(data.portfolios || []);
        } catch (err) {
            console.error('Failed to fetch portfolios:', err);
        }
    };

    const addStrategy = () => {
        setSelectedStrategies([
            ...selectedStrategies,
            {
                name: `strategy_${selectedStrategies.length + 1}`,
                strategy: 'jsg',
                params: {},
                weight: 0,
            },
        ]);
    };

    const removeStrategy = (index: number) => {
        setSelectedStrategies(selectedStrategies.filter((_, i) => i !== index));
    };

    const updateStrategy = (index: number, field: keyof Strategy, value: string | number) => {
        const updated = [...selectedStrategies];
        updated[index] = { ...updated[index], [field]: value };
        setSelectedStrategies(updated);
    };

    const createPortfolio = async () => {
        if (!newPortfolioName || selectedStrategies.length === 0) return;

        setLoading(true);
        try {
            const resp = await fetch(`${API_BASE}/portfolio`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: newPortfolioName,
                    strategies: selectedStrategies,
                    weight_method: weightMethod,
                    rebalance_frequency: 'weekly',
                }),
            });

            if (resp.ok) {
                await fetchPortfolios();
                setShowCreateModal(false);
                setNewPortfolioName('');
                setSelectedStrategies([]);
            }
        } catch (err) {
            console.error('Failed to create portfolio:', err);
        } finally {
            setLoading(false);
        }
    };

    const runBacktest = async () => {
        if (!selectedPortfolio) return;

        setLoading(true);
        setBacktestResult(null);

        try {
            const resp = await fetch(`${API_BASE}/portfolio/${selectedPortfolio}/backtest`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    start_date: startDate,
                    end_date: endDate,
                    symbols: symbols.split(',').map((s) => s.trim()),
                    initial_capital: initialCapital,
                }),
            });

            if (resp.ok) {
                const data = await resp.json();
                setBacktestResult(data);
            }
        } catch (err) {
            console.error('Backtest failed:', err);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="portfolio-manager">
            <div className="portfolio-header">
                <h2>📊 多策略组合管理</h2>
                <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
                    + 创建组合
                </button>
            </div>

            {/* Portfolio List */}
            <div className="portfolio-grid">
                {portfolios.map((p) => (
                    <div
                        key={p.portfolio_id}
                        className={`portfolio-card ${selectedPortfolio === p.portfolio_id ? 'selected' : ''}`}
                        onClick={() => setSelectedPortfolio(p.portfolio_id)}
                    >
                        <h3>{p.portfolio_id}</h3>
                        <div className="portfolio-info">
                            <span>策略数: {p.n_strategies}</span>
                        </div>
                        <div className="weight-bars">
                            {Object.entries(p.weights).map(([name, weight]) => (
                                <div key={name} className="weight-bar">
                                    <span className="weight-name">{name}</span>
                                    <div className="weight-fill" style={{ width: `${weight * 100}%` }} />
                                    <span className="weight-value">{(weight * 100).toFixed(0)}%</span>
                                </div>
                            ))}
                        </div>
                    </div>
                ))}

                {portfolios.length === 0 && (
                    <div className="empty-state">
                        <p>暂无组合，点击"创建组合"开始</p>
                    </div>
                )}
            </div>

            {/* Backtest Panel */}
            {selectedPortfolio && (
                <div className="backtest-panel">
                    <h3>回测配置</h3>
                    <div className="form-grid">
                        <div className="form-group">
                            <label>开始日期</label>
                            <input
                                type="date"
                                value={startDate}
                                onChange={(e) => setStartDate(e.target.value)}
                            />
                        </div>
                        <div className="form-group">
                            <label>结束日期</label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                            />
                        </div>
                        <div className="form-group">
                            <label>标的代码</label>
                            <input
                                type="text"
                                value={symbols}
                                onChange={(e) => setSymbols(e.target.value)}
                                placeholder="sz.300750,sz.002475"
                            />
                        </div>
                        <div className="form-group">
                            <label>初始资金</label>
                            <input
                                type="number"
                                value={initialCapital}
                                onChange={(e) => setInitialCapital(Number(e.target.value))}
                            />
                        </div>
                    </div>
                    <button
                        className="btn-primary"
                        onClick={runBacktest}
                        disabled={loading}
                    >
                        {loading ? '运行中...' : '运行回测'}
                    </button>

                    {/* Backtest Results */}
                    {backtestResult && (
                        <div className="backtest-results">
                            <h4>回测结果</h4>
                            <div className="result-grid">
                                <div className="result-item">
                                    <span className="label">总收益</span>
                                    <span className={`value ${backtestResult.total_return >= 0 ? 'positive' : 'negative'}`}>
                                        {(backtestResult.total_return * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="result-item">
                                    <span className="label">夏普比率</span>
                                    <span className="value">{backtestResult.sharpe_ratio.toFixed(2)}</span>
                                </div>
                                <div className="result-item">
                                    <span className="label">最大回撤</span>
                                    <span className="value negative">
                                        {(backtestResult.max_drawdown * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="result-item">
                                    <span className="label">最终权益</span>
                                    <span className="value">{formatMoney(backtestResult.final_equity, { symbol: '¥' })}</span>
                                </div>
                            </div>

                            <h4>策略表现</h4>
                            <table className="strategy-table">
                                <thead>
                                    <tr>
                                        <th>策略</th>
                                        <th>权重</th>
                                        <th>最终权益</th>
                                        <th>交易次数</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {Object.entries(backtestResult.strategy_results).map(([name, data]) => (
                                        <tr key={name}>
                                            <td>{name}</td>
                                            <td>{(data.weight * 100).toFixed(0)}%</td>
                                            <td>{formatMoney(data.final_equity, { symbol: '¥' })}</td>
                                            <td>{data.trades}</td>
                                        </tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    )}
                </div>
            )}

            {/* Create Portfolio Modal */}
            {showCreateModal && (
                <div className="modal-overlay" onClick={() => setShowCreateModal(false)}>
                    <div className="modal" onClick={(e) => e.stopPropagation()}>
                        <h3>创建组合</h3>

                        <div className="form-group">
                            <label>组合名称</label>
                            <input
                                type="text"
                                value={newPortfolioName}
                                onChange={(e) => setNewPortfolioName(e.target.value)}
                                placeholder="my_portfolio"
                            />
                        </div>

                        <div className="form-group">
                            <label>权重方法</label>
                            <select
                                value={weightMethod}
                                onChange={(e) => setWeightMethod(e.target.value)}
                            >
                                {WEIGHT_METHODS.map((m) => (
                                    <option key={m.id} value={m.id}>
                                        {m.name} - {m.description}
                                    </option>
                                ))}
                            </select>
                        </div>

                        <div className="strategies-section">
                            <div className="strategies-header">
                                <label>策略配置</label>
                                <button className="btn-small" onClick={addStrategy}>
                                    + 添加策略
                                </button>
                            </div>

                            {selectedStrategies.map((s, i) => (
                                <div key={i} className="strategy-row">
                                    <input
                                        type="text"
                                        value={s.name}
                                        onChange={(e) => updateStrategy(i, 'name', e.target.value)}
                                        placeholder="策略名称"
                                    />
                                    <select
                                        value={s.strategy}
                                        onChange={(e) => updateStrategy(i, 'strategy', e.target.value)}
                                    >
                                        {AVAILABLE_STRATEGIES.map((as) => (
                                            <option key={as.id} value={as.id}>
                                                {as.name}
                                            </option>
                                        ))}
                                    </select>
                                    <button className="btn-danger" onClick={() => removeStrategy(i)}>
                                        ×
                                    </button>
                                </div>
                            ))}
                        </div>

                        <div className="modal-actions">
                            <button className="btn-secondary" onClick={() => setShowCreateModal(false)}>
                                取消
                            </button>
                            <button
                                className="btn-primary"
                                onClick={createPortfolio}
                                disabled={loading || !newPortfolioName || selectedStrategies.length === 0}
                            >
                                {loading ? '创建中...' : '创建'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
                .portfolio-manager {
                    padding: 20px;
                    background: #1a1a1a;
                    border-radius: 12px;
                    color: #fff;
                }
                .portfolio-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 20px;
                }
                .portfolio-header h2 {
                    margin: 0;
                }
                .btn-primary {
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    border: none;
                    padding: 10px 20px;
                    border-radius: 8px;
                    color: white;
                    cursor: pointer;
                    font-weight: 500;
                }
                .btn-primary:hover {
                    opacity: 0.9;
                }
                .btn-primary:disabled {
                    opacity: 0.5;
                    cursor: not-allowed;
                }
                .btn-secondary {
                    background: #333;
                    border: 1px solid #555;
                    padding: 10px 20px;
                    border-radius: 8px;
                    color: white;
                    cursor: pointer;
                }
                .btn-small {
                    background: #333;
                    border: 1px solid #555;
                    padding: 5px 10px;
                    border-radius: 4px;
                    color: white;
                    cursor: pointer;
                    font-size: 12px;
                }
                .btn-danger {
                    background: #dc3545;
                    border: none;
                    padding: 5px 10px;
                    border-radius: 4px;
                    color: white;
                    cursor: pointer;
                }
                .portfolio-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
                    gap: 16px;
                    margin-bottom: 24px;
                }
                .portfolio-card {
                    background: #252525;
                    border: 1px solid #333;
                    border-radius: 10px;
                    padding: 16px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .portfolio-card:hover {
                    border-color: #667eea;
                }
                .portfolio-card.selected {
                    border-color: #667eea;
                    background: #2a2a3a;
                }
                .portfolio-card h3 {
                    margin: 0 0 8px 0;
                    font-size: 14px;
                    color: #aaa;
                }
                .portfolio-info {
                    color: #888;
                    font-size: 12px;
                    margin-bottom: 12px;
                }
                .weight-bars {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                .weight-bar {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                }
                .weight-name {
                    font-size: 11px;
                    color: #888;
                    width: 60px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .weight-fill {
                    height: 6px;
                    background: linear-gradient(90deg, #667eea, #764ba2);
                    border-radius: 3px;
                    flex: 1;
                }
                .weight-value {
                    font-size: 11px;
                    color: #667eea;
                    width: 30px;
                    text-align: right;
                }
                .empty-state {
                    grid-column: 1 / -1;
                    text-align: center;
                    padding: 40px;
                    color: #666;
                }
                .backtest-panel {
                    background: #252525;
                    border-radius: 10px;
                    padding: 20px;
                }
                .backtest-panel h3 {
                    margin: 0 0 16px 0;
                }
                .form-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 16px;
                    margin-bottom: 16px;
                }
                .form-group {
                    display: flex;
                    flex-direction: column;
                    gap: 6px;
                }
                .form-group label {
                    font-size: 12px;
                    color: #888;
                }
                .form-group input,
                .form-group select {
                    background: #333;
                    border: 1px solid #444;
                    border-radius: 6px;
                    padding: 8px 12px;
                    color: #fff;
                    font-size: 14px;
                }
                .backtest-results {
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid #333;
                }
                .backtest-results h4 {
                    margin: 0 0 12px 0;
                    color: #aaa;
                }
                .result-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                    margin-bottom: 20px;
                }
                .result-item {
                    background: #1a1a1a;
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                }
                .result-item .label {
                    display: block;
                    font-size: 11px;
                    color: #888;
                    margin-bottom: 4px;
                }
                .result-item .value {
                    font-size: 18px;
                    font-weight: 600;
                }
                .result-item .value.positive {
                    color: #00d26a;
                }
                .result-item .value.negative {
                    color: #ff6b6b;
                }
                .strategy-table {
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                }
                .strategy-table th,
                .strategy-table td {
                    padding: 10px;
                    text-align: left;
                    border-bottom: 1px solid #333;
                }
                .strategy-table th {
                    color: #888;
                    font-weight: 500;
                }
                .modal-overlay {
                    position: fixed;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(0, 0, 0, 0.7);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 1000;
                }
                .modal {
                    background: #1a1a1a;
                    border-radius: 12px;
                    padding: 24px;
                    width: 500px;
                    max-width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                }
                .modal h3 {
                    margin: 0 0 20px 0;
                }
                .strategies-section {
                    margin-top: 16px;
                }
                .strategies-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 12px;
                }
                .strategy-row {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 8px;
                }
                .strategy-row input,
                .strategy-row select {
                    flex: 1;
                    background: #333;
                    border: 1px solid #444;
                    border-radius: 6px;
                    padding: 8px;
                    color: #fff;
                }
                .modal-actions {
                    display: flex;
                    justify-content: flex-end;
                    gap: 12px;
                    margin-top: 20px;
                }
            `}</style>
        </div>
    );
};

export default PortfolioManager;
