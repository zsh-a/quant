import React, { useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import type { SessionSummary, StrategyMeta } from '../types';
import NewSessionForm from './NewSessionForm';
import SessionList from './SessionList';
import SimulationPanel from './SimulationPanel';
import AlphaLabWorkspace from './AlphaLabWorkspace';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';

interface LabPanelProps {
  strategies: StrategyMeta[];
  sessions: SessionSummary[];
  selectedSessionIds: string[];
  onStart: (config: any) => Promise<void>;
  onToggleSelection: (id: string) => void;
  onViewSession: (id: string) => void;
  onStopSession: (id: string) => void;
  onDeleteSession: (id: string) => void;
  onOpenMarketAdmin: () => void;
  error?: string | null;
}

function getSubtab(pathname: string): 'manual' | 'simulation' | 'alpha' {
  if (pathname.endsWith('/simulation')) return 'simulation';
  if (pathname.endsWith('/alpha')) return 'alpha';
  return 'manual';
}

const LabPanel: React.FC<LabPanelProps> = ({
  strategies,
  sessions,
  selectedSessionIds,
  onStart,
  onToggleSelection,
  onViewSession,
  onStopSession,
  onDeleteSession,
  onOpenMarketAdmin,
  error,
}) => {
  const location = useLocation();
  const navigate = useNavigate();
  const subtab = getSubtab(location.pathname);

  const manualSessions = useMemo(
    () => sessions.filter((s) => s.mode !== 'simulation' && s.source !== 'automation'),
    [sessions],
  );

  const simulationSessions = useMemo(
    () => sessions.filter((s) => s.source === 'automation' || s.mode === 'simulation'),
    [sessions],
  );

  const handleTabChange = (value: string) => {
    const path = value === 'manual' ? '/lab' : `/lab/${value}`;
    navigate(path, { replace: true });
  };

  return (
    <div className="space-y-6">
      <Tabs value={subtab} onValueChange={handleTabChange}>
        <TabsList>
          <TabsTrigger value="manual">Manual</TabsTrigger>
          <TabsTrigger value="simulation">Simulation</TabsTrigger>
          <TabsTrigger value="alpha">Alpha Lab</TabsTrigger>
        </TabsList>

        <TabsContent value="manual">
          <div className="grid gap-6 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.4fr)]">
            <NewSessionForm strategies={strategies} onStart={onStart} error={error || null} />
            <SessionList
              title="Manual Tasks"
              defaultFilter="manual"
              sessions={manualSessions}
              selectedSessionIds={selectedSessionIds}
              onToggleSelection={onToggleSelection}
              onViewSession={onViewSession}
              onStopSession={onStopSession}
              onDeleteSession={onDeleteSession}
            />
          </div>
        </TabsContent>

        <TabsContent value="simulation">
          <div className="space-y-6">
            <SimulationPanel
              strategies={strategies}
              onSelectSession={onViewSession}
              onOpenMarketAdmin={onOpenMarketAdmin}
            />
            <SessionList
              title="Simulation Tasks"
              defaultFilter="simulation"
              sessions={simulationSessions}
              selectedSessionIds={selectedSessionIds}
              onToggleSelection={onToggleSelection}
              onViewSession={onViewSession}
              onStopSession={onStopSession}
              onDeleteSession={onDeleteSession}
            />
          </div>
        </TabsContent>

        <TabsContent value="alpha">
          <AlphaLabWorkspace onViewSession={onViewSession} />
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default LabPanel;
