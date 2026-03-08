import React from 'react';
import type { StrategyMeta } from '../types';
import { SectionCard } from './layout/SectionCard';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Separator } from './ui/separator';

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

  const fieldClassName = 'flex flex-col gap-2';

  const renderParamInput = (key: string, conf: any) => {
    const value = paramValues[key] ?? conf.default;

    if (conf.type === 'bool') {
      return (
        <label className="flex items-center gap-3 rounded-2xl border border-border/70 bg-secondary/40 px-4 py-3">
          <input
            type="checkbox"
            checked={Boolean(value)}
            onChange={(e) => onParamChange(key, e.target.checked)}
            style={{ width: 'auto' }}
          />
          <span className="tagline !mb-0">{conf.description || key}</span>
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
      <Input
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
    <SectionCard
      title={title}
      description="Configure strategy parameters, data range and execution mode."
      action={headerAction}
      contentClassName="space-y-5"
    >
        <div className={fieldClassName}>
          <label className="tagline">Strategy</label>
          <select className="glass-input" value={selectedStrategy} onChange={(e) => onStrategyChange(e.target.value)}>
            <option value="" disabled>Select a strategy...</option>
            {strategies.map((strategy) => (
              <option key={strategy.name} value={strategy.name}>{strategy.label}</option>
            ))}
          </select>
        </div>

        <div className={fieldClassName}>
          <label className="tagline">Symbol</label>
          <Input value={symbol} onChange={(e) => onSymbolChange(e.target.value)} placeholder="例如 sh.000300" />
        </div>

        <div className={`grid gap-4 ${showEndDate ? 'md:grid-cols-2' : ''}`}>
          <div className={fieldClassName}>
            <label className="tagline">Start Date</label>
            <Input type="date" value={startDate} onChange={(e) => onStartDateChange(e.target.value)} />
          </div>
          {showEndDate && onEndDateChange && (
            <div className={fieldClassName}>
              <label className="tagline">End Date</label>
              <Input type="date" value={endDate} onChange={(e) => onEndDateChange(e.target.value)} />
            </div>
          )}
        </div>

        {selectedStrategy && (
          <div className="rounded-[1.4rem] border border-border/70 bg-secondary/35 p-4">
            <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
              <div className="tagline" style={{ color: 'var(--primary)' }}>
                Configuration {currentStrategy?.params ? `(${Object.keys(currentStrategy.params).length} fields)` : ''}
              </div>
              <Button variant="ghost" size="sm" onClick={onResetDefaults}>
                Reset Defaults
              </Button>
            </div>
            <Separator className="mb-4" />
            <div className="grid gap-4">
              {currentStrategy?.params && Object.entries(currentStrategy.params).map(([key, conf]) => (
                <div key={key} className={fieldClassName}>
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
    </SectionCard>
  );
};

export default StrategyConfigForm;
