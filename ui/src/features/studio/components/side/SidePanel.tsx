/**
 * SidePanel — tab container for the inspector sidebars (Signals / Decision /
 * Regime). All three reuse the studio store, so the tab is purely a visual
 * grouping; switching tabs does not reset state.
 */

import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../../../components/ui/tabs';
import { SignalSidebar } from './SignalSidebar';
import { DecisionInspector } from './DecisionInspector';
import { RegimeTimeline } from './RegimeTimeline';

export type SidePanelTab = 'signals' | 'decision' | 'regime';

const DEFAULT_TAB: SidePanelTab = 'decision';

export function SidePanel() {
  return (
    <div
      className="flex h-full min-h-0 flex-col bg-card/40"
      data-testid="side-panel"
    >
      <Tabs defaultValue={DEFAULT_TAB} className="flex h-full min-h-0 flex-col">
        <div className="px-2 pt-2">
          <TabsList className="w-full">
            <TabsTrigger value="signals" className="flex-1" data-testid="tab-signals">
              Signals
            </TabsTrigger>
            <TabsTrigger value="decision" className="flex-1" data-testid="tab-decision">
              Decision
            </TabsTrigger>
            <TabsTrigger value="regime" className="flex-1" data-testid="tab-regime">
              Regime
            </TabsTrigger>
          </TabsList>
        </div>
        <TabsContent value="signals" className="mt-2 min-h-0 flex-1 overflow-hidden">
          <SignalSidebar />
        </TabsContent>
        <TabsContent value="decision" className="mt-2 min-h-0 flex-1 overflow-y-auto">
          <DecisionInspector />
        </TabsContent>
        <TabsContent value="regime" className="mt-2 min-h-0 flex-1 overflow-y-auto">
          <RegimeTimeline />
        </TabsContent>
      </Tabs>
    </div>
  );
}
