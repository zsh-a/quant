import React, { useEffect, useMemo, useState } from 'react';
import type { SessionSummary, StrategyMeta } from '../types';
import NewSessionForm from './NewSessionForm';
import SessionList from './SessionList';
import SimulationPanel from './SimulationPanel';
import AlphaLabWorkspace from './AlphaLabWorkspace';
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
  onDeleteSession: (id: string) => void;
  onOpenMarketAdmin: () => void;
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
  onDeleteSession,
  onOpenMarketAdmin,
  error,
}) => {
  const [labSubtab, setLabSubtab] = useState<'manual' | 'simulation' | 'alpha'>(() => {
    const stored = window.localStorage.getItem('quent.lab.subtab');
    if (stored === 'alpha') {
      return 'alpha';
    }
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
        title="策略实验室"
        description="策略执行、模拟编排和 Alpha 因子研究统一收纳在同一个实验室工作区。"
      />

      <Tabs value={labSubtab} onValueChange={(value) => setLabSubtab(value as 'manual' | 'simulation' | 'alpha')}>
        <TabsList>
          <TabsTrigger value="manual">手动任务</TabsTrigger>
          <TabsTrigger value="simulation">模拟任务</TabsTrigger>
          <TabsTrigger value="alpha">Alpha Lab</TabsTrigger>
        </TabsList>

        <TabsContent value="manual">
          <div className="grid gap-6 xl:grid-cols-[minmax(360px,0.9fr)_minmax(0,1.4fr)]">
            <NewSessionForm strategies={strategies} onStart={onStart} error={error} />
            <SessionList
              title="手动任务列表"
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
              title="模拟任务列表"
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
