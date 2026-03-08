import React, { useState, useEffect } from 'react';
import { API_BASE } from '../utils/api';

interface ParamSpec {
    name: string;
    param_type: 'int' | 'float';
    low: number;
    high: number;
    step: number;
}

interface OptTask {
    task_id: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    progress: number;
    result?: {
        best_params: Record<string, number>;
        best_score: number;
        elapsed_time: number;
        n_iterations: number;
    };
}


export const OptimizerPanel: React.FC = () => {
    const [strategy, setStrategy] = useState('jsg');
    const [method, setMethod] = useState('bayesian');
    const [objective, setObjective] = useState('max_sharpe');
    const [iterations, setIterations] = useState(50);
    const [params, _setParams] = useState<ParamSpec[]>([
        { name: 'ma_short', param_type: 'int', low: 5, high: 20, step: 1 },
        { name: 'ma_long', param_type: 'int', low: 20, high: 60, step: 5 },
    ]);
    const [startDate, setStartDate] = useState('2023-01-01');
    const [endDate, setEndDate] = useState('2024-01-01');
    const [symbols, setSymbols] = useState('sz.300750');
    const [tasks, setTasks] = useState<OptTask[]>([]);
    const [loading, setLoading] = useState(false);

    useEffect(() => {
        const running = tasks.filter(t => t.status === 'running' || t.status === 'pending');
        if (!running.length) return;
        const id = setInterval(async () => {
            for (const t of running) {
                const resp = await fetch(`${API_BASE}/optimize/${t.task_id}`);
                const data = await resp.json();
                setTasks(prev => prev.map(x => x.task_id === t.task_id ? { ...x, ...data } : x));
            }
        }, 2000);
        return () => clearInterval(id);
    }, [tasks]);

    const submit = async () => {
        setLoading(true);
        const resp = await fetch(`${API_BASE}/optimize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                strategy, param_space: params, method, objective,
                n_iterations: iterations,
                backtest_config: { start_date: startDate, end_date: endDate, symbols: symbols.split(',') }
            })
        });
        if (resp.ok) {
            const data = await resp.json();
            setTasks([{ task_id: data.task_id, status: 'pending', progress: 0 }, ...tasks]);
        }
        setLoading(false);
    };

    return (
        <div style={{ padding: 20, background: '#1a1a1a', borderRadius: 12, color: '#fff' }}>
            <h2>🎯 参数优化器</h2>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12, marginBottom: 16 }}>
                <select value={strategy} onChange={e => setStrategy(e.target.value)} style={inputStyle}>
                    <option value="jsg">JSG策略</option>
                    <option value="rotation">轮动策略</option>
                </select>
                <select value={method} onChange={e => setMethod(e.target.value)} style={inputStyle}>
                    <option value="grid">网格搜索</option>
                    <option value="bayesian">贝叶斯优化</option>
                </select>
                <select value={objective} onChange={e => setObjective(e.target.value)} style={inputStyle}>
                    <option value="max_sharpe">最大夏普</option>
                    <option value="max_return">最大收益</option>
                </select>
                <input type="number" value={iterations} onChange={e => setIterations(+e.target.value)} style={inputStyle} />
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 12, marginBottom: 16 }}>
                <input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} style={inputStyle} />
                <input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} style={inputStyle} />
                <input value={symbols} onChange={e => setSymbols(e.target.value)} style={inputStyle} />
            </div>
            <button onClick={submit} disabled={loading} style={btnStyle}>
                {loading ? '提交中...' : '开始优化'}
            </button>
            {tasks.map(t => (
                <div key={t.task_id} style={{ background: '#252525', borderRadius: 8, padding: 12, marginTop: 12 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>{t.task_id}</span>
                        <span style={{ color: t.status === 'completed' ? '#28a745' : '#ffc107' }}>{t.status}</span>
                    </div>
                    {t.result && (
                        <div style={{ marginTop: 8, fontSize: 13 }}>
                            <div>最优参数: {JSON.stringify(t.result.best_params)}</div>
                            <div>最优分数: {t.result.best_score.toFixed(4)}</div>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
};

const inputStyle: React.CSSProperties = { background: '#333', border: '1px solid #444', borderRadius: 6, padding: 8, color: '#fff' };
const btnStyle: React.CSSProperties = { background: 'linear-gradient(135deg, #667eea, #764ba2)', border: 'none', padding: '12px 24px', borderRadius: 8, color: 'white', cursor: 'pointer', width: '100%' };

export default OptimizerPanel;
