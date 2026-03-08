import React, { useEffect, useMemo, useState } from 'react';
import type { SessionSummary, StrategyMeta } from '../types';
import NewSessionForm from './NewSessionForm';
import SessionList from './SessionList';
import SimulationPanel from './SimulationPanel';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { PageHeader } from './layout/PageHeader';

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
    <div className="space-y-6">
      <PageHeader
        eyebrow="Execution Workspace"
        title="Strategy Lab"
        description="Manual Runs 只保留 backtest / live，所有 simulation 统一在 Simulation 视图里。"
      />

      <Tabs value={labSubtab} onValueChange={(value) => setLabSubtab(value as 'manual' | 'simulation')}>
        <TabsList>
          <TabsTrigger value="manual">Manual Runs</TabsTrigger>
          <TabsTrigger value="simulation">Simulation</TabsTrigger>
        </TabsList>

        <TabsContent value="manual">
          <div className="grid gap-6 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.4fr)]">
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
        </TabsContent>

        <TabsContent value="simulation">
          <div className="space-y-6">
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
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default LabPanel;
