# Brooks Studio — Developer Extension Guide

> Audience: anyone adding a new chart layer, side-panel tab, or
> `BarEvent` field. Pair with [studio.md](studio.md) (user guide) and
> [README.md](README.md) (platform overview).

The Studio frontend lives at `ui/src/features/studio/`. The contract
across layers and panels is small enough that you almost never have to
touch the chart container or the playback loop — the bits below are
the only sanctioned extension points.

## Architecture in 30 seconds

```
fetch SessionTimeline ──► Zustand StudioStore
                            │
                            ├── currentBarIdx / playState / speed     ◄── useReplay (keyboard + ticker)
                            ├── visibleLayers (persisted)             ◄── LayerToggle
                            └── timeline (live tail = WS coalesced)   ◄── useTimeline (50 ms batch)
                            │
                            ▼
            ChartCanvas (one IChartApi)        SidePanel (4 tabs)
              │                                  Signals / Decision / Regime / Compare
              └── ChartLayer registry
                    regime / swings / ema / channels / signals /
                    decisions / fills / stop_adj / htf_overlay /
                    pnl / reasoning
```

The store is the only thing every consumer reads, which is why playback
state survives tab switches and why the URL ↔ store sync (`useUrlState`)
needs no per-component plumbing.

---

## Add a chart layer (5 steps)

A layer is a stateless plugin that owns its own series / primitives on
the shared `IChartApi` and re-renders on `(timeline, currentBarIdx)`
changes.

1. **Create the file.** `ui/src/features/studio/layers/<name>.ts`
2. **Implement the protocol** (`layers/types.ts`):

   ```ts
   import type { ChartLayer, LayerCtx, LayerHandle } from './types';

   export const myLayer: ChartLayer = {
     id: 'my_layer',
     name: 'My Layer',
     defaultVisible: false,
     mount(ctx: LayerCtx): LayerHandle {
       const series = ctx.chart.addSeries(/* ... */);
       return {
         update(timeline, currentBarIdx) {
           // Diff against your previous render. The framework calls
           // this on every relevant store change — no event filtering
           // needed here.
         },
         unmount() {
           ctx.chart.removeSeries(series);
         },
       };
     },
   };
   ```

3. **Register it.** Add an entry to `ui/src/features/studio/layers/registry.ts`:

   ```ts
   import { myLayer } from './my_layer';

   export const STUDIO_LAYERS: ChartLayer[] = [
     // ...
     myLayer,
   ];
   ```

   Order in `STUDIO_LAYERS` controls the order rows appear in
   **LayerToggle**.

4. **Add a test.** Drop a fixture-driven test under
   `ui/src/features/studio/__tests__/<name>Layer.test.ts` (mirror
   `regimeLayer.test.ts`). The shared `mockChart.ts` gives you a
   minimal `IChartApi` substitute.

5. **Toggle the default.** If you want the layer on by default, leave
   `defaultVisible: true`; otherwise users opt in via LayerToggle and
   the choice persists to `localStorage`.

### Example — minimal "high of the day" line

```ts
// ui/src/features/studio/layers/hod.ts
import { LineSeries } from 'lightweight-charts';
import type { ChartLayer, LayerCtx } from './types';

export const hodLayer: ChartLayer = {
  id: 'hod',
  name: 'High of day',
  defaultVisible: false,
  mount(ctx: LayerCtx) {
    const series = ctx.chart.addSeries(LineSeries, { color: '#fbbf24', lineWidth: 1 });
    return {
      update(timeline, currentBarIdx) {
        if (!timeline) return series.setData([]);
        const upTo = timeline.bars.slice(0, Math.max(currentBarIdx, 0) + 1);
        let runningHigh = -Infinity;
        const points = upTo.map((b) => {
          runningHigh = Math.max(runningHigh, b.high);
          return { time: Math.floor(b.timestamp_ns / 1e9), value: runningHigh };
        });
        series.setData(points);
      },
      unmount() {
        ctx.chart.removeSeries(series);
      },
    };
  },
};
```

---

## Add a side-panel tab (3 steps)

Side panels are React components that subscribe to the store. Tabs in
`SidePanel` route to them.

1. **Create the component.**
   `ui/src/features/studio/components/side/MyTab.tsx` — read store with
   any of the existing selectors (`useTimelineState`,
   `useInspectedBarIdx`, etc.).

2. **Wire the tab.** Edit
   `ui/src/features/studio/components/side/SidePanel.tsx`:

   ```diff
   - export type SidePanelTab = 'signals' | 'decision' | 'regime' | 'compare';
   + export type SidePanelTab = 'signals' | 'decision' | 'regime' | 'compare' | 'my_tab';
   ```

   ```diff
     <TabsList>
       <TabsTrigger value="signals">Signals</TabsTrigger>
       <TabsTrigger value="decision">Decision</TabsTrigger>
       <TabsTrigger value="regime">Regime</TabsTrigger>
       <TabsTrigger value="compare">Compare</TabsTrigger>
   +   <TabsTrigger value="my_tab">My Tab</TabsTrigger>
     </TabsList>
   ```

   ```diff
   + <TabsContent value="my_tab" className="mt-2 min-h-0 flex-1 overflow-y-auto">
   +   <MyTab />
   + </TabsContent>
   ```

3. **Drop a smoke test.** Mirror `SignalSidebar.test.tsx` — render the
   panel with a fixture timeline and assert the relevant rows appear.

### Example — "current bar JSON" tab

```tsx
// ui/src/features/studio/components/side/RawBarTab.tsx
import { useInspectedBarIdx, useTimelineState } from '../../store';

export function RawBarTab() {
  const timeline = useTimelineState();
  const { barIdx } = useInspectedBarIdx();
  if (!timeline) return null;
  const bar = timeline.bars[barIdx];
  const event = timeline.events.find((e) => e.bar_idx === barIdx);
  return (
    <pre className="overflow-auto px-3 py-2 text-[11px]">
      {JSON.stringify({ bar, event }, null, 2)}
    </pre>
  );
}
```

---

## Add a `BarEvent` field (backend + frontend)

`BarEvent` is the shared schema between `src/api/schemas/brooks_studio.py`
and `ui/src/features/studio/types.ts`. The two must stay in sync.

### Backend

1. Add the field to the Pydantic model in
   `src/api/schemas/brooks_studio.py`:

   ```python
   class BarEvent(BaseModel):
       # ...
       my_field: MyFieldView | None = None
   ```

2. Populate it in `BrooksTimelineLoader` (`src/services/brooks_timeline_loader.py`)
   by reading the relevant column out of `session_db.extra` and
   mapping to the new model.

3. If the field is computed on every bar of a live session, also wire
   the writer into `src/tasks/brooks_live_task.py` so the WS push
   carries it.

4. Add a fixture-driven test in
   `tests/api/test_brooks_studio_router.py` asserting the field
   round-trips for a known session.

### Frontend

1. Mirror the type in `ui/src/features/studio/types.ts`:

   ```ts
   export interface BarEvent {
     // ...
     my_field?: MyFieldView | null;
   }
   ```

2. Consume it from a layer (`update`) or a side-panel component. The
   store already merges new events into `timeline.events` via
   `applyLiveEventBatch`, so reads "just work" — no plumbing needed.

3. Update the docs table in
   [studio.md](studio.md) if the field surfaces a new layer or panel.

### Naming guarantees

* Field keys are `snake_case` on both ends so JSON crosses the boundary
  unchanged.
* Optional fields use `T | None` (Pydantic) and `T?: T | null`
  (TypeScript) so absence stays distinct from "explicit null".

---

## Performance contract

Studio's perf budget is enforced by these rules — do not break them:

* **Mount the chart once.** `ChartCanvas` uses an empty `useEffect`
  dependency list; rebuilding the `IChartApi` discards the WebGL
  canvas + every layer. If you need state inside a closure that
  outlives the mount, use a ref (`timelineRef` is the existing
  pattern).
* **Coalesce WS bursts.** Live events flow through a 50 ms buffer in
  `useTimeline`. New live event sources should call
  `actions.applyLiveEventBatch(events)` rather than the per-event
  `applyLiveEvent` to keep that contract.
* **Lazy-render large histories.** When `bars.length > 5000`,
  `ChartCanvas` paints a 1500-bar trailing window first and hydrates
  the rest in `requestIdleCallback`. Layers should accept that the
  first `update` call may see a partial chart and recompute when the
  full history arrives.
* **Virtualize big lists.** Anything that maps over events / signals
  past ~50 rows belongs inside `react-window` — see the
  `SignalSidebar` virtualization threshold for the precedent.

---

## Tests

```bash
cd ui
bun test                                  # full suite
bun test src/features/studio/__tests__    # studio only
```

Layers go in `<name>Layer.test.ts`; side panels in
`<Name>.test.tsx`. The repo has examples for every existing layer +
panel — copy whichever is closest.

## File map

| Concern | File |
|---------|------|
| Top route | `ui/src/features/studio/components/BrooksStudioPage.tsx` |
| Landing / picker | `ui/src/features/studio/components/StudioLanding.tsx` |
| Store | `ui/src/features/studio/store.ts` |
| WS + initial load | `ui/src/features/studio/hooks/useTimeline.ts` |
| Replay loop + keyboard | `ui/src/features/studio/hooks/useReplay.ts` |
| URL ↔ store sync | `ui/src/features/studio/hooks/useUrlState.ts` |
| Mobile breakpoint | `ui/src/features/studio/hooks/useMobileViewport.ts` |
| Layer registry | `ui/src/features/studio/layers/registry.ts` |
| Side panel | `ui/src/features/studio/components/side/SidePanel.tsx` |
| Backend router | `src/api/brooks_studio_router.py` |
| Backend schemas | `src/api/schemas/brooks_studio.py` |
| Timeline loader | `src/services/brooks_timeline_loader.py` |
