import React, { useEffect, useState } from 'react';
import { ShieldCheck, Layers, FlaskConical, CalendarClock } from 'lucide-react';
import { StrategyMeta } from '../types';
import StrategyConfigForm from './StrategyConfigForm';
import { apiFetch } from '../utils/api';
import { Button } from './ui/button';
import { Switch } from './ui/switch';
import { Separator } from './ui/separator';

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
    const [paperMode, setPaperMode] = useState(true);
    const [useAsync, setUseAsync] = useState(true);
    const [enableRisk, setEnableRisk] = useState(true);

    // Job scheduling state
    const [showJobConfig, setShowJobConfig] = useState(false);
    const [jobName, setJobName] = useState('');
    const [notifyOnOrder, setNotifyOnOrder] = useState(false);
    const [telegramChatId, setTelegramChatId] = useState('');
    const [creatingJob, setCreatingJob] = useState(false);
    const [jobMessage, setJobMessage] = useState<string | null>(null);

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

    const buildParams = () => {
        const strat = strategies.find((item) => item.name === selectedStrategy);
        const finalParams: Record<string, any> = {};
        if (!strat?.params) return finalParams;
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
        return finalParams;
    };

    const handleStart = () => {
        const actualMode = mode === 'live' && paperMode ? 'paper' : mode;
        const payload: any = {
            strategy: selectedStrategy,
            symbol,
            start_date: startDate,
            mode: actualMode,
            params: buildParams(),
            async: useAsync && (mode === 'backtest' || mode === 'simulation'),
            enable_risk_management: enableRisk,
        };
        if (endDate) payload.end_date = endDate;
        onStart(payload);
    };

    const handleCreateJob = async () => {
        setCreatingJob(true);
        setJobMessage(null);
        try {
            const resp = await apiFetch('/simulation-jobs', {
                method: 'POST',
                body: JSON.stringify({
                    name: jobName || `${selectedStrategy} Job`,
                    strategy: selectedStrategy,
                    symbol,
                    start_date: startDate,
                    end_date: endDate || null,
                    params: buildParams(),
                    notification: { telegram: { enabled: notifyOnOrder, chat_id: telegramChatId.trim() || undefined } },
                    enabled: true,
                    schedule: 'daily',
                }),
            });
            if (!resp.ok) throw new Error(await resp.text());
            setJobMessage('Job created');
            setShowJobConfig(false);
            setJobName('');
        } catch (err) {
            setJobMessage(err instanceof Error ? err.message : 'Failed to create job');
        } finally {
            setCreatingJob(false);
        }
    };

    return (
        <StrategyConfigForm
            title="Task Config"
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
                <select className="glass-input" style={{ width: 'auto' }} value={mode} onChange={(e) => setMode(e.target.value)}>
                    <option value="backtest">Backtest</option>
                    <option value="simulation">Simulation</option>
                    <option value="live">Live Trading</option>
                </select>
            }
            footer={
                <div className="space-y-3">
                    {/* Toggles */}
                    <div className="rounded-2xl border border-border/60 bg-secondary/30 divide-y divide-border/40">
                        <label htmlFor="sw-risk" className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer select-none">
                            <div className="flex items-center gap-2.5">
                                <ShieldCheck size={15} className={enableRisk ? 'text-emerald-500' : 'text-muted-foreground/50'} />
                                <div>
                                    <span className="text-sm font-medium leading-none">Risk Control</span>
                                    <p className="text-xs text-muted-foreground mt-0.5">止损 / 止盈 / 仓位限制 / 回撤控制</p>
                                </div>
                            </div>
                            <Switch id="sw-risk" checked={enableRisk} onCheckedChange={setEnableRisk} />
                        </label>
                        {mode === 'live' && (
                            <label htmlFor="sw-paper" className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer select-none">
                                <div className="flex items-center gap-2.5">
                                    <FlaskConical size={15} className={paperMode ? 'text-amber-500' : 'text-muted-foreground/50'} />
                                    <div>
                                        <span className="text-sm font-medium leading-none">Paper Mode</span>
                                        <p className="text-xs text-muted-foreground mt-0.5">模拟撮合，不连接真实交易服务器</p>
                                    </div>
                                </div>
                                <Switch id="sw-paper" checked={paperMode} onCheckedChange={setPaperMode} />
                            </label>
                        )}
                        {(mode === 'backtest' || mode === 'simulation') && (
                            <label htmlFor="sw-async" className="flex items-center justify-between gap-3 px-4 py-3 cursor-pointer select-none">
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

                    {/* Actions */}
                    {error && <div className="text-sm text-destructive">{error}</div>}
                    <Button className="w-full" onClick={handleStart} disabled={!selectedStrategy}>
                        Start Session
                    </Button>

                    <Separator />

                    {/* Schedule as Job */}
                    <button
                        className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors w-full"
                        onClick={() => setShowJobConfig(!showJobConfig)}
                    >
                        <CalendarClock size={14} />
                        {showJobConfig ? 'Cancel Scheduling' : 'Schedule as Automation Job'}
                    </button>

                    {showJobConfig && (
                        <div className="rounded-2xl border border-border/60 bg-secondary/30 p-4 space-y-3">
                            <div className="grid gap-1.5">
                                <label className="tagline">Job Name</label>
                                <input className="glass-input" value={jobName} onChange={(e) => setJobName(e.target.value)} placeholder={`${selectedStrategy || 'Strategy'} Job`} />
                            </div>
                            <label className="flex items-center gap-2.5 cursor-pointer">
                                <input type="checkbox" checked={notifyOnOrder} onChange={(e) => setNotifyOnOrder(e.target.checked)} style={{ width: 'auto' }} />
                                <span className="tagline !mb-0">Telegram Notifications</span>
                            </label>
                            {notifyOnOrder && (
                                <div className="grid gap-1.5">
                                    <label className="tagline">Telegram Chat ID</label>
                                    <input className="glass-input" value={telegramChatId} onChange={(e) => setTelegramChatId(e.target.value)} placeholder="Leave empty for default" />
                                </div>
                            )}
                            {jobMessage && <div className={`text-sm ${jobMessage === 'Job created' ? 'text-emerald-500' : 'text-destructive'}`}>{jobMessage}</div>}
                            <Button variant="outline" className="w-full" onClick={handleCreateJob} disabled={creatingJob || !selectedStrategy}>
                                {creatingJob ? 'Creating...' : 'Create Automation Job'}
                            </Button>
                        </div>
                    )}
                </div>
            }
        />
    );
};

export default NewSessionForm;
