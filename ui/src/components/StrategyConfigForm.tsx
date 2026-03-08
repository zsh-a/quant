import React from 'react';
import type { StrategyMeta } from '../types';

interface StrategyConfigFormProps {
  title: string;
  strategies: StrategyMeta[];
  selectedStrategy: string;
  onStrategyChange: (value: string) => void;
  symbol: string;
  onSymbolChange: (value: string) => void;
  startDate: string;
  onStartDateChange: (value: string) => void;
  endDate?: string;
  onEndDateChange?: (value: string) => void;
  paramValues: Record<string, any>;
  onParamChange: (key: string, value: any) => void;
  onResetDefaults: () => void;
  headerAction?: React.ReactNode;
  footer?: React.ReactNode;
  showEndDate?: boolean;
}

const fieldStyle: React.CSSProperties = {
  display: 'flex',
  flexDirection: 'column',
  gap: '0.4rem',
};

const StrategyConfigForm: React.FC<StrategyConfigFormProps> = ({
  title,
  strategies,
  selectedStrategy,
  onStrategyChange,
  symbol,
  onSymbolChange,
  startDate,
  onStartDateChange,
  endDate = '',
  onEndDateChange,
  paramValues,
  onParamChange,
  onResetDefaults,
  headerAction,
  footer,
  showEndDate = true,
}) => {
  const currentStrategy = strategies.find((strategy) => strategy.name === selectedStrategy);

  const renderParamInput = (key: string, conf: any) => {
    const value = paramValues[key] ?? conf.default;

    if (conf.type === 'bool') {
      return (
        <label style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onParamChange(key, e.target.checked)}
            style={{ width: 'auto' }}
          />
          <span className="tagline">{conf.description || key}</span>
        </label>
      );
    }

    if (conf.options && Array.isArray(conf.options)) {
      return (
        <select className="glass-input" value={String(value)} onChange={(e) => onParamChange(key, e.target.value)}>
          {conf.options.map((option: string) => (
            <option key={option} value={option}>{option}</option>
          ))}
        </select>
      );
    }

    const inputType = conf.type === 'int' || conf.type === 'float' ? 'number' : 'text';
    return (
      <input
        className="glass-input"
        type={inputType}
        value={value ?? ''}
        min={conf.min}
        max={conf.max}
        step={conf.type === 'int' ? 1 : conf.type === 'float' ? 'any' : undefined}
        onChange={(e) => onParamChange(key, e.target.value)}
        placeholder={conf.description || key}
      />
    );
  };

  return (
    <div className="glass card">
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.25rem', gap: '1rem' }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        {headerAction}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <div style={fieldStyle}>
          <label className="tagline">Strategy</label>
          <select className="glass-input" value={selectedStrategy} onChange={(e) => onStrategyChange(e.target.value)}>
            <option value="" disabled>Select a strategy...</option>
            {strategies.map((strategy) => (
              <option key={strategy.name} value={strategy.name}>{strategy.label}</option>
            ))}
          </select>
        </div>

        <div style={fieldStyle}>
          <label className="tagline">Symbol</label>
          <input className="glass-input" value={symbol} onChange={(e) => onSymbolChange(e.target.value)} placeholder="例如 sh.000300" />
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: showEndDate ? '1fr 1fr' : '1fr', gap: '1rem' }}>
          <div style={fieldStyle}>
            <label className="tagline">Start Date</label>
            <input className="glass-input" type="date" value={startDate} onChange={(e) => onStartDateChange(e.target.value)} />
          </div>
          {showEndDate && onEndDateChange && (
            <div style={fieldStyle}>
              <label className="tagline">End Date</label>
              <input className="glass-input" type="date" value={endDate} onChange={(e) => onEndDateChange(e.target.value)} />
            </div>
          )}
        </div>

        {selectedStrategy && (
          <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.05)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div className="tagline" style={{ color: 'var(--primary)' }}>
                Configuration {currentStrategy?.params ? `(${Object.keys(currentStrategy.params).length} fields)` : ''}
              </div>
              <button className="btn-ghost" style={{ padding: '2px 8px', fontSize: '0.7rem' }} onClick={onResetDefaults}>
                Reset Defaults
              </button>
            </div>

            <div style={{ display: 'grid', gap: '0.85rem' }}>
              {currentStrategy?.params && Object.entries(currentStrategy.params).map(([key, conf]) => (
                <div key={key} style={fieldStyle}>
                  {conf.type !== 'bool' && (
                    <label className="tagline">
                      {key}
                      {conf.description ? ` · ${conf.description}` : ''}
                    </label>
                  )}
                  {renderParamInput(key, conf)}
                </div>
              ))}
            </div>
          </div>
        )}

        {footer}
      </div>
    </div>
  );
};

export default StrategyConfigForm;
