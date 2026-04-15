import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import type { SessionSummary, StrategyMeta } from '../types';
import NewSessionForm from './NewSessionForm';
import SessionList from './SessionList';
import { SimulationPanel } from './SimulationPanel';
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

function getSubtab(pathname: string): 'tasks' | 'alpha' {
  if (pathname.endsWith('/alpha')) return 'alpha';
  return 'tasks';
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

  const handleTabChange = (value: string) => {
    const path = value === 'tasks' ? '/lab' : `/lab/${value}`;
    navigate(path, { replace: true });
  };

  return (
    <div className="space-y-6">
      <Tabs value={subtab} onValueChange={handleTabChange}>
        <TabsList>
          <TabsTrigger value="tasks">Tasks</TabsTrigger>
          <TabsTrigger value="alpha">Alpha Lab</TabsTrigger>
        </TabsList>

        <TabsContent value="tasks">
          <div className="grid gap-6 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.4fr)]">
            <NewSessionForm strategies={strategies} onStart={onStart} error={error || null} />
            <div className="space-y-6">
              <SessionList
                title="Sessions"
                sessions={sessions}
                selectedSessionIds={selectedSessionIds}
                onToggleSelection={onToggleSelection}
                onViewSession={onViewSession}
                onStopSession={onStopSession}
                onDeleteSession={onDeleteSession}
              />
              <SimulationPanel
                onSelectSession={onViewSession}
                onOpenMarketAdmin={onOpenMarketAdmin}
              />
            </div>
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
