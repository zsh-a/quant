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

    useEffect(() => {
        if (strategies.length > 0 && !selectedStrategy) {
            setSelectedStrategy(strategies[0].name);
        }
    }, [strategies]);

    useEffect(() => {
        const strat = strategies.find(s => s.name === selectedStrategy);
        if (strat) {
            const defaults: Record<string, any> = {};
            Object.entries(strat.params).forEach(([key, conf]) => {
                defaults[key] = conf.default;
            });
            setParamValues(defaults);
        }
    }, [selectedStrategy, strategies]);

    const handleStart = () => {
        const payload: any = { strategy: selectedStrategy, symbol, start_date: startDate, mode, params: paramValues };
        if (endDate) payload.end_date = endDate;
        onStart(payload);
    };

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
                <div className="input-group">
                    <label className="tagline">Strategy</label>
                    <select className="glass-input" value={selectedStrategy} onChange={(e) => setSelectedStrategy(e.target.value)}>
                        {strategies.map(s => (
                            <option key={s.name} value={s.name}>{s.label}</option>
                        ))}
                    </select>
                </div>

                {/* Dynamic Parameters */}
                {selectedStrategy && strategies.find(s => s.name === selectedStrategy)?.params && (
                    <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
                        <div className="tagline" style={{ marginBottom: '1rem', color: 'var(--primary)' }}>Configuration</div>
                        {Object.entries(strategies.find(s => s.name === selectedStrategy)!.params).map(([key, conf]) => (
                            <div key={key} className="input-group" style={{ marginBottom: '1rem' }}>
                                <label className="tagline" title={conf.description}>{key} <span style={{ opacity: 0.5 }}>- {conf.description}</span></label>
                                {conf.options ? (
                                    <select
                                        className="glass-input"
                                        value={paramValues[key] || conf.default}
                                        onChange={(e) => setParamValues(prev => ({ ...prev, [key]: e.target.value }))}
                                    >
                                        {conf.options.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                                    </select>
                                ) : (
                                    <input
                                        type={conf.type === 'int' || conf.type === 'float' ? 'number' : 'text'}
                                        value={paramValues[key] !== undefined ? paramValues[key] : ''}
                                        onChange={(e) => {
                                            let val: any = e.target.value;
                                            if (conf.type === 'int') val = parseInt(val);
                                            else if (conf.type === 'float') val = parseFloat(val);
                                            setParamValues(prev => ({ ...prev, [key]: val }));
                                        }}
                                        className="glass-input"
                                    />
                                )}
                            </div>
                        ))}
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
