import React, { useState, useEffect } from 'react';
import { StrategyMeta } from '../types';

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
    const [useAsync, setUseAsync] = useState(true); // Default to async mode

    useEffect(() => {
        if (strategies && strategies.length > 0 && !selectedStrategy) {
            setSelectedStrategy(strategies[0].name);
        }
    }, [strategies]);

    useEffect(() => {
        const strat = strategies.find(s => s.name === selectedStrategy);
        if (strat && strat.params) {
            const defaults: Record<string, any> = {};
            Object.entries(strat.params).forEach(([key, conf]) => {
                defaults[key] = conf.default;
            });
            setParamValues(defaults);
        }
    }, [selectedStrategy, strategies]);

    const handleStart = () => {
        // Prepare final params by merging defaults with current values
        const finalParams: Record<string, any> = {};
        const strat = strategies.find(s => s.name === selectedStrategy);
        
        if (strat && strat.params) {
            Object.entries(strat.params).forEach(([key, conf]) => {
                const userVal = paramValues[key];
                if (userVal === undefined || userVal === '') {
                    finalParams[key] = conf.default;
                } else {
                    // Final safety cast
                    if (conf.type === 'int') finalParams[key] = parseInt(userVal) || 0;
                    else if (conf.type === 'float') finalParams[key] = parseFloat(userVal) || 0;
                    else finalParams[key] = userVal;
                }
            });
        }

        const payload: any = {
            strategy: selectedStrategy,
            symbol,
            start_date: startDate,
            mode,
            params: finalParams,
            async: useAsync && mode === 'backtest' // Only async for backtest mode
        };
        if (endDate) payload.end_date = endDate;
        onStart(payload);
    };

    const currentStrategy = strategies.find(s => s.name === selectedStrategy);

    return (
        <div className="glass card">
            <h3 style={{ marginBottom: '1.5rem' }}>Start New Session</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
                <div className="input-group">
                    <label className="tagline">Mode</label>
                    <select className="glass-input" value={mode} onChange={(e) => setMode(e.target.value)}>
                        <option value="backtest">Backtest (Historical)</option>
                        <option value="simulation">Simulation (Paper Trade)</option>
                        <option value="live">Live Trading</option>
                    </select>
                </div>

                {mode === 'backtest' && (
                    <div className="input-group" style={{ flexDirection: 'row', alignItems: 'center', gap: '0.5rem' }}>
                        <input
                            type="checkbox"
                            id="asyncMode"
                            checked={useAsync}
                            onChange={(e) => setUseAsync(e.target.checked)}
                            style={{ width: 'auto' }}
                        />
                        <label htmlFor="asyncMode" className="tagline" style={{ cursor: 'pointer' }}>
                            Run in background (Celery queue)
                        </label>
                    </div>
                )}

                <div className="input-group">
                    <label className="tagline">Strategy</label>
                    <select className="glass-input" value={selectedStrategy} onChange={(e) => setSelectedStrategy(e.target.value)}>
                        <option value="" disabled>Select a strategy...</option>
                        {strategies.map(s => (
                            <option key={s.name} value={s.name}>{s.label}</option>
                        ))}
                    </select>
                </div>

                {/* Dynamic Parameters */}
                {selectedStrategy && (
                    <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                            <div className="tagline" style={{ color: 'var(--primary)' }}>
                                Configuration {currentStrategy?.params ? `(${Object.keys(currentStrategy.params).length} fields)` : ''}
                            </div>
                            {currentStrategy?.params && (
                                <button 
                                    className="btn-ghost" 
                                    style={{ padding: '2px 8px', fontSize: '0.7rem' }}
                                    onClick={() => {
                                        const defaults: Record<string, any> = {};
                                        Object.entries(currentStrategy.params).forEach(([k, v]) => defaults[k] = v.default);
                                        setParamValues(defaults);
                                    }}
                                >
                                    Reset Defaults
                                </button>
                            )}
                        </div>
                        
                        {!currentStrategy ? (
                            <div style={{ fontSize: '0.8rem', opacity: 0.5 }}>Loading strategy metadata...</div>
                        ) : !currentStrategy.params || Object.keys(currentStrategy.params).length === 0 ? (
                            <div style={{ fontSize: '0.8rem', opacity: 0.5 }}>No configurable parameters for this strategy.</div>
                        ) : (
                            Object.entries(currentStrategy.params).map(([key, conf]) => (
                                <div key={key} className="input-group" style={{ marginBottom: '1rem' }}>
                                    <label className="tagline" title={conf.description}>
                                        {key} <span style={{ opacity: 0.5 }}>- {conf.description}</span>
                                    </label>
                                    {conf.options ? (
                                        <select
                                            className="glass-input"
                                            value={paramValues[key] !== undefined ? paramValues[key] : conf.default}
                                            onChange={(e) => setParamValues(prev => ({ ...prev, [key]: e.target.value }))}
                                        >
                                            {conf.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                        </select>
                                    ) : (
                                        <input
                                            type="text" 
                                            placeholder={`Default: ${conf.default}`}
                                            value={paramValues[key] !== undefined ? paramValues[key] : ''}
                                            onChange={(e) => {
                                                // Allow any string during typing
                                                const val = e.target.value;
                                                setParamValues(prev => ({ ...prev, [key]: val }));
                                            }}
                                            onBlur={(e) => {
                                                // Apply basic parsing on blur if empty or invalid
                                                const raw = e.target.value;
                                                if (raw === '') return;
                                                
                                                let parsed: any = raw;
                                                if (conf.type === 'int') parsed = parseInt(raw) || 0;
                                                else if (conf.type === 'float') parsed = parseFloat(raw) || 0;
                                                setParamValues(prev => ({ ...prev, [key]: parsed }));
                                            }}
                                            className="glass-input"
                                        />
                                    )}
                                </div>
                            ))
                        )}
                    </div>
                )}

                <div className="input-group">
                    <label className="tagline">Symbol</label>
                    <input type="text" value={symbol} onChange={(e) => setSymbol(e.target.value)} className="glass-input" />
                </div>
                <div className="input-group">
                    <label className="tagline">Start Date</label>
                    <input type="date" value={startDate} onChange={(e) => setStartDate(e.target.value)} className="glass-input" />
                </div>
                <div className="input-group">
                    <label className="tagline">End Date (Optional)</label>
                    <input type="date" value={endDate} onChange={(e) => setEndDate(e.target.value)} className="glass-input" />
                </div>
                <button onClick={handleStart}>
                    Launch {mode.charAt(0).toUpperCase() + mode.slice(1)} Session
                </button>
                {error && <div style={{ color: 'var(--danger)', fontSize: '0.8rem' }}>{error}</div>}
            </div>
        </div>
    );
};

export default NewSessionForm;
