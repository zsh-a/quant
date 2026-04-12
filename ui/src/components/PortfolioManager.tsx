import React, { useState, useEffect } from 'react';
import { formatMoney } from '../utils/format';
import { PageHeader } from './layout/PageHeader';
import { API_BASE } from '../utils/api';

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

const AVAILABLE_STRATEGIES = [
    { id: 'jsg', name: 'JSG Strategy', description: 'Golden/Death Cross strategy' },
    { id: 'rotation', name: 'Rotation Strategy', description: 'Sector rotation strategy' },
];

const WEIGHT_METHODS = [
    { id: 'equal', name: 'Equal Weight', description: 'Same weight for each strategy' },
    { id: 'vol_inverse', name: 'Inverse Volatility', description: 'Higher weight for lower volatility strategies' },
    { id: 'sharpe', name: 'Sharpe Ratio', description: 'Weight by Sharpe ratio' },
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
            <PageHeader
                eyebrow="Portfolio Lab"
                title="Portfolio Management"
                description="Create multi-strategy portfolios, adjust configurations, and run portfolio backtests."
                actions={
                    <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
                        Create Portfolio
                    </button>
                }
            />

            <div className="portfolio-header" style={{ marginTop: '1.25rem' }}>
                <h2>Portfolio List</h2>
                <button className="btn-primary" onClick={() => setShowCreateModal(true)}>
                    + Create Portfolio
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
                        <h3>{p.name || p.portfolio_id}</h3>
                        <div className="portfolio-info">
                            <span>Strategies: {p.n_strategies}</span>
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
                        <p>No portfolios yet. Click "Create Portfolio" to get started.</p>
                    </div>
                )}
            </div>

            {/* Backtest Panel */}
            {selectedPortfolio && (
                <div className="backtest-panel">
                    <h3>Backtest Configuration</h3>
                    <div className="form-grid">
                        <div className="form-group">
                            <label>Start Date</label>
                            <input
                                type="date"
                                value={startDate}
                                onChange={(e) => setStartDate(e.target.value)}
                            />
                        </div>
                        <div className="form-group">
                            <label>End Date</label>
                            <input
                                type="date"
                                value={endDate}
                                onChange={(e) => setEndDate(e.target.value)}
                            />
                        </div>
                        <div className="form-group">
                            <label>Symbols</label>
                            <input
                                type="text"
                                value={symbols}
                                onChange={(e) => setSymbols(e.target.value)}
                                placeholder="sz.300750,sz.002475"
                            />
                        </div>
                        <div className="form-group">
                            <label>Initial Capital</label>
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
                        {loading ? 'Running...' : 'Run Backtest'}
                    </button>

                    {/* Backtest Results */}
                    {backtestResult && (
                        <div className="backtest-results">
                            <h4>Backtest Results</h4>
                            <div className="result-grid">
                                <div className="result-item">
                                    <span className="label">Total Return</span>
                                    <span className={`value ${backtestResult.total_return >= 0 ? 'positive' : 'negative'}`}>
                                        {(backtestResult.total_return * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="result-item">
                                    <span className="label">Sharpe Ratio</span>
                                    <span className="value">{backtestResult.sharpe_ratio.toFixed(2)}</span>
                                </div>
                                <div className="result-item">
                                    <span className="label">Max Drawdown</span>
                                    <span className="value negative">
                                        {(backtestResult.max_drawdown * 100).toFixed(2)}%
                                    </span>
                                </div>
                                <div className="result-item">
                                    <span className="label">Final Equity</span>
                                    <span className="value">{formatMoney(backtestResult.final_equity, { symbol: '¥' })}</span>
                                </div>
                            </div>

                            <h4>Strategy Performance</h4>
                            <table className="strategy-table">
                                <thead>
                                    <tr>
                                        <th>Strategy</th>
                                        <th>Weight</th>
                                        <th>Final Equity</th>
                                        <th>Trades</th>
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
                        <h3>Create Portfolio</h3>

                        <div className="form-group">
                            <label>Portfolio Name</label>
                            <input
                                type="text"
                                value={newPortfolioName}
                                onChange={(e) => setNewPortfolioName(e.target.value)}
                                placeholder="my_portfolio"
                            />
                        </div>

                        <div className="form-group">
                            <label>Weight Method</label>
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
                                <label>Strategy Configuration</label>
                                <button className="btn-small" onClick={addStrategy}>
                                    + Add Strategy
                                </button>
                            </div>

                            {selectedStrategies.map((s, i) => (
                                <div key={i} className="strategy-row">
                                    <input
                                        type="text"
                                        value={s.name}
                                        onChange={(e) => updateStrategy(i, 'name', e.target.value)}
                                        placeholder="Strategy name"
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
                                Cancel
                            </button>
                            <button
                                className="btn-primary"
                                onClick={createPortfolio}
                                disabled={loading || !newPortfolioName || selectedStrategies.length === 0}
                            >
                                {loading ? 'Creating...' : 'Create'}
                            </button>
                        </div>
                    </div>
                </div>
            )}

            <style>{`
                .portfolio-manager {
                    padding: 20px;
                    background: var(--color-card);
                    border-radius: 12px;
                    color: var(--color-foreground);
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
                    background: var(--color-primary);
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
                    background: var(--color-accent);
                    border: 1px solid var(--color-border);
                    padding: 10px 20px;
                    border-radius: 8px;
                    color: white;
                    cursor: pointer;
                }
                .btn-small {
                    background: var(--color-accent);
                    border: 1px solid var(--color-border);
                    padding: 5px 10px;
                    border-radius: 4px;
                    color: white;
                    cursor: pointer;
                    font-size: 12px;
                }
                .btn-danger {
                    background: var(--color-destructive);
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
                    background: var(--color-secondary);
                    border: 1px solid var(--color-border);
                    border-radius: 10px;
                    padding: 16px;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .portfolio-card:hover {
                    border-color: var(--color-primary);
                }
                .portfolio-card.selected {
                    border-color: var(--color-primary);
                    background: var(--color-accent);
                }
                .portfolio-card h3 {
                    margin: 0 0 8px 0;
                    font-size: 14px;
                    color: var(--color-secondary-foreground);
                }
                .portfolio-info {
                    color: var(--color-muted-foreground);
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
                    color: var(--color-muted-foreground);
                    width: 60px;
                    overflow: hidden;
                    text-overflow: ellipsis;
                }
                .weight-fill {
                    height: 6px;
                    background: var(--color-primary);
                    border-radius: 3px;
                    flex: 1;
                }
                .weight-value {
                    font-size: 11px;
                    color: var(--color-primary);
                    width: 30px;
                    text-align: right;
                }
                .empty-state {
                    grid-column: 1 / -1;
                    text-align: center;
                    padding: 40px;
                    color: var(--color-muted-foreground);
                }
                .backtest-panel {
                    background: var(--color-secondary);
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
                    color: var(--color-muted-foreground);
                }
                .form-group input,
                .form-group select {
                    background: var(--color-accent);
                    border: 1px solid var(--color-border);
                    border-radius: 6px;
                    padding: 8px 12px;
                    color: var(--color-foreground);
                    font-size: 14px;
                }
                .backtest-results {
                    margin-top: 20px;
                    padding-top: 20px;
                    border-top: 1px solid var(--color-border);
                }
                .backtest-results h4 {
                    margin: 0 0 12px 0;
                    color: var(--color-secondary-foreground);
                }
                .result-grid {
                    display: grid;
                    grid-template-columns: repeat(4, 1fr);
                    gap: 16px;
                    margin-bottom: 20px;
                }
                .result-item {
                    background: var(--color-card);
                    padding: 12px;
                    border-radius: 8px;
                    text-align: center;
                }
                .result-item .label {
                    display: block;
                    font-size: 11px;
                    color: var(--color-muted-foreground);
                    margin-bottom: 4px;
                }
                .result-item .value {
                    font-size: 18px;
                    font-weight: 600;
                }
                .result-item .value.positive {
                    color: #46A488;
                }
                .result-item .value.negative {
                    color: #CF5A55;
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
                    border-bottom: 1px solid var(--color-border);
                }
                .strategy-table th {
                    color: var(--color-muted-foreground);
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
                    background: var(--color-card);
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
                    background: var(--color-accent);
                    border: 1px solid var(--color-border);
                    border-radius: 6px;
                    padding: 8px;
                    color: var(--color-foreground);
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
