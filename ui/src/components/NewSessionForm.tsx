import React, { useEffect, useState } from 'react';
import { StrategyMeta } from '../types';
import StrategyConfigForm from './StrategyConfigForm';
import { Button } from './ui/button';

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
        };

        if (endDate) payload.end_date = endDate;
        onStart(payload);
    };

    return (
        <StrategyConfigForm
            title="手动任务"
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
                        <option value="backtest">回测</option>
                        <option value="live">实盘</option>
                    </select>
                </div>
            }
            footer={
                <div className="space-y-4">
                    {mode === 'backtest' && (
                        <label className="flex items-center gap-3 rounded-2xl border border-border/70 bg-secondary/35 px-4 py-3">
                            <input
                                type="checkbox"
                                checked={useAsync}
                                onChange={(e) => setUseAsync(e.target.checked)}
                                style={{ width: 'auto' }}
                            />
                            <span className="tagline !mb-0">后台运行（Celery 队列）</span>
                        </label>
                    )}
                    {error && <div style={{ color: 'var(--danger)' }}>{error}</div>}
                    <Button className="w-full" onClick={handleStart} disabled={!selectedStrategy}>
                        启动会话
                    </Button>
                </div>
            }
        />
    );
};

export default NewSessionForm;
