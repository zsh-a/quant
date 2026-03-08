import React, { useEffect, useMemo, useState } from 'react';
import type { SessionSummary, StrategyMeta } from '../types';
import NewSessionForm from './NewSessionForm';
import SessionList from './SessionList';
import SimulationPanel from './SimulationPanel';

interface LabPanelProps {
  strategies: StrategyMeta[];
  sessions: SessionSummary[];
  selectedSessionIds: string[];
  onStart: (config: any) => Promise<void>;
  onToggleSelection: (id: string) => void;
  onViewSession: (id: string) => void;
  onStopSession: (id: string) => void;
  error: string | null;
}

const tabButtonStyle = (active: boolean): React.CSSProperties => ({
  padding: '0.5rem 1rem',
  borderRadius: 999,
  border: '1px solid rgba(255,255,255,0.1)',
  background: active ? 'var(--primary)' : 'rgba(255,255,255,0.04)',
  color: 'white',
  cursor: 'pointer',
});

const LabPanel: React.FC<LabPanelProps> = ({
  strategies,
  sessions,
  selectedSessionIds,
  onStart,
  onToggleSelection,
  onViewSession,
  onStopSession,
  error,
}) => {
  const [labSubtab, setLabSubtab] = useState<'manual' | 'simulation'>(() => {
    const stored = window.localStorage.getItem('quent.lab.subtab');
    return stored === 'simulation' ? 'simulation' : 'manual';
  });

  useEffect(() => {
    window.localStorage.setItem('quent.lab.subtab', labSubtab);
  }, [labSubtab]);

  const manualSessions = useMemo(
    () => sessions.filter((session) => session.mode !== 'simulation' && session.source !== 'automation'),
    [sessions],
  );

  const simulationSessions = useMemo(
    () => sessions.filter((session) => session.source === 'automation' || session.mode === 'simulation'),
    [sessions],
  );

  return (
    <div style={{ display: 'grid', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <h2 style={{ margin: 0 }}>Strategy Lab</h2>
          <p className="tagline" style={{ marginTop: '0.4rem' }}>Manual Runs 只保留 backtest / live，所有 simulation 统一在 Simulation 视图里。</p>
        </div>
        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button style={tabButtonStyle(labSubtab === 'manual')} onClick={() => setLabSubtab('manual')}>Manual Runs</button>
          <button style={tabButtonStyle(labSubtab === 'simulation')} onClick={() => setLabSubtab('simulation')}>Simulation</button>
        </div>
      </div>

      {labSubtab === 'manual' ? (
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 2fr', gap: '2rem' }}>
          <NewSessionForm strategies={strategies} onStart={onStart} error={error} />
          <SessionList
            title="Manual Session List"
            defaultFilter="manual"
            sessions={manualSessions}
            selectedSessionIds={selectedSessionIds}
            onToggleSelection={onToggleSelection}
            onViewSession={onViewSession}
            onStopSession={onStopSession}
          />
        </div>
      ) : (
        <div style={{ display: 'grid', gap: '1rem' }}>
          <SimulationPanel strategies={strategies} onSelectSession={onViewSession} />
          <SessionList
            title="Simulation Session List"
            defaultFilter="simulation"
            sessions={simulationSessions}
            selectedSessionIds={selectedSessionIds}
            onToggleSelection={onToggleSelection}
            onViewSession={onViewSession}
            onStopSession={onStopSession}
          />
        </div>
      )}
    </div>
  );
};

export default LabPanel;
