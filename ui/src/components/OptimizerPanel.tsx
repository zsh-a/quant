import React, { useState, useEffect } from 'react';
import { API_BASE } from '../utils/api';
import { PageHeader } from './layout/PageHeader';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { StatusBadge } from './layout/StatusBadge';
import { Progress } from './ui/progress';

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
        <div className="space-y-6">
            <PageHeader
                eyebrow="Model Search"
                title="参数优化器"
                description="Search parameter ranges, track optimization tasks and inspect best-performing configurations."
            />
            <SectionCard title="Optimization Setup" description="Choose strategy, objective, search budget and backtest universe.">
                <div className="grid gap-3 xl:grid-cols-4">
                    <select value={strategy} onChange={e => setStrategy(e.target.value)} className="glass-input">
                        <option value="jsg">JSG策略</option>
                        <option value="rotation">轮动策略</option>
                    </select>
                    <select value={method} onChange={e => setMethod(e.target.value)} className="glass-input">
                        <option value="grid">网格搜索</option>
                        <option value="bayesian">贝叶斯优化</option>
                    </select>
                    <select value={objective} onChange={e => setObjective(e.target.value)} className="glass-input">
                        <option value="max_sharpe">最大夏普</option>
                        <option value="max_return">最大收益</option>
                    </select>
                    <Input type="number" value={iterations} onChange={e => setIterations(+e.target.value)} />
                </div>
                <div className="grid gap-3 xl:grid-cols-3">
                    <Input type="date" value={startDate} onChange={e => setStartDate(e.target.value)} />
                    <Input type="date" value={endDate} onChange={e => setEndDate(e.target.value)} />
                    <Input value={symbols} onChange={e => setSymbols(e.target.value)} />
                </div>
                <Button onClick={submit} disabled={loading} className="w-full sm:w-auto">
                    {loading ? '提交中...' : '开始优化'}
                </Button>
            </SectionCard>

            <SectionCard title="Task Queue" description="Follow task state and inspect best parameters once jobs complete.">
                <div className="space-y-3">
                    {tasks.map(t => (
                        <div key={t.task_id} className="rounded-2xl border border-border/70 bg-secondary/45 p-4">
                            <div className="flex flex-wrap items-center justify-between gap-3">
                                <div className="space-y-1">
                                    <div className="font-mono text-sm text-foreground">{t.task_id}</div>
                                    <div className="text-xs text-muted-foreground">Progress {t.progress ?? 0}%</div>
                                </div>
                                <StatusBadge value={t.status} />
                            </div>
                            <div className="mt-3">
                                <Progress value={t.progress ?? 0} />
                            </div>
                            {t.result && (
                                <div className="mt-3 space-y-1 text-sm text-muted-foreground">
                                    <div>最优参数: {JSON.stringify(t.result.best_params)}</div>
                                    <div>最优分数: {t.result.best_score.toFixed(4)}</div>
                                </div>
                            )}
                        </div>
                    ))}
                    {tasks.length === 0 ? <div className="empty-state">暂无优化任务</div> : null}
                </div>
            </SectionCard>
        </div>
    );
};

export default OptimizerPanel;
