import React, { useEffect, useState } from 'react';
import { ShieldCheck, Layers } from 'lucide-react';
import { StrategyMeta } from '../types';
import StrategyConfigForm from './StrategyConfigForm';
import { Button } from './ui/button';
import { Switch } from './ui/switch';

interface NewSessionFormProps {
    strategies: StrategyMeta[];
    onStart: (config: any) => Promise<void>;
    error: string | null;
}

const NewSessionForm: React.FC<NewSessionFormProps> = ({ strategies, onStart, error }) => {
    const [selectedStrategy, setSelectedStrategy] = useState<string>('');
    const [paramValues, setParamValues] = useState<Record<string, any>>({});
    const [symbol, setSymbol] = useState('sh.000300');
    const [startDate, setStartDate] = useState('2024-01-01');
    const [endDate, setEndDate] = useState<string>('');
    const [mode, setMode] = useState('backtest');
    const [useAsync, setUseAsync] = useState(true);
    const [enableRisk, setEnableRisk] = useState(true);

    useEffect(() => {
        if (strategies.length > 0 && !selectedStrategy) {
            setSelectedStrategy(strategies[0].name);
        }
    }, [strategies, selectedStrategy]);

    useEffect(() => {
        const strat = strategies.find((item) => item.name === selectedStrategy);
        if (!strat?.params) return;
        const defaults: Record<string, any> = {};
        Object.entries(strat.params).forEach(([key, conf]) => {
            defaults[key] = conf.default;
        });
        setParamValues(defaults);
    }, [selectedStrategy, strategies]);

    const resetDefaults = () => {
        const strat = strategies.find((item) => item.name === selectedStrategy);
        if (!strat?.params) return;
        const defaults: Record<string, any> = {};
        Object.entries(strat.params).forEach(([key, conf]) => {
            defaults[key] = conf.default;
        });
        setParamValues(defaults);
    };

    const handleStart = () => {
        const strat = strategies.find((item) => item.name === selectedStrategy);
        const finalParams: Record<string, any> = {};

        if (strat?.params) {
            Object.entries(strat.params).forEach(([key, conf]) => {
                const userVal = paramValues[key];
                if (userVal === undefined || userVal === '') {
                    finalParams[key] = conf.default;
                } else if (conf.type === 'int') {
                    finalParams[key] = parseInt(userVal, 10) || 0;
                } else if (conf.type === 'float') {
                    finalParams[key] = parseFloat(userVal) || 0;
                } else if (conf.type === 'bool') {
                    finalParams[key] = Boolean(userVal);
                } else {
                    finalParams[key] = userVal;
                }
            });
        }

        const payload: any = {
            strategy: selectedStrategy,
            symbol,
            start_date: startDate,
            mode,
            params: finalParams,
            async: useAsync && mode === 'backtest',
            enable_risk_management: enableRisk,
        };

        if (endDate) payload.end_date = endDate;
        onStart(payload);
    };

    return (
        <StrategyConfigForm
            title="Manual Task"
            strategies={strategies}
            selectedStrategy={selectedStrategy}
            onStrategyChange={setSelectedStrategy}
            symbol={symbol}
            onSymbolChange={setSymbol}
            startDate={startDate}
            onStartDateChange={setStartDate}
            endDate={endDate}
            onEndDateChange={setEndDate}
            paramValues={paramValues}
            onParamChange={(key, value) => setParamValues((prev) => ({ ...prev, [key]: value }))}
            onResetDefaults={resetDefaults}
            headerAction={
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <select className="glass-input" style={{ width: 'auto' }} value={mode} onChange={(e) => setMode(e.target.value)}>
                        <option value="backtest">Backtest</option>
                        <option value="paper">Paper Trading</option>
                        <option value="live">Live</option>
                    </select>
                </div>
            }
            footer={
                <div className="space-y-3">
                    <div className="rounded-2xl border border-border/60 bg-secondary/30 divide-y divide-border/40">
                        <label
                            htmlFor="sw-risk"
                            className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer select-none"
                        >
                            <div className="flex items-center gap-2.5">
                                <ShieldCheck size={15} className={enableRisk ? 'text-emerald-500' : 'text-muted-foreground/50'} />
                                <div>
                                    <span className="text-sm font-medium leading-none">Risk Control</span>
                                    <p className="text-xs text-muted-foreground mt-0.5">止损 / 止盈 / 仓位限制 / 回撤控制</p>
                                </div>
                            </div>
                            <Switch id="sw-risk" checked={enableRisk} onCheckedChange={setEnableRisk} />
                        </label>
                        {mode === 'backtest' && (
                            <label
                                htmlFor="sw-async"
                                className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer select-none"
                            >
                                <div className="flex items-center gap-2.5">
                                    <Layers size={15} className={useAsync ? 'text-blue-500' : 'text-muted-foreground/50'} />
                                    <div>
                                        <span className="text-sm font-medium leading-none">Background</span>
                                        <p className="text-xs text-muted-foreground mt-0.5">Celery 后台队列异步执行</p>
                                    </div>
                                </div>
                                <Switch id="sw-async" checked={useAsync} onCheckedChange={setUseAsync} />
                            </label>
                        )}
                    </div>
                    {error && <div className="text-sm text-destructive">{error}</div>}
                    <Button className="w-full" onClick={handleStart} disabled={!selectedStrategy}>
                        Start Session
                    </Button>
                </div>
            }
        />
    );
};

export default NewSessionForm;
