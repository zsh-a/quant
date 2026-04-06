# Frontend Architecture

## Tech Stack

| Library | Version | Purpose |
|---------|---------|---------|
| React | 19.1 | UI framework |
| TypeScript | 5.7 | Type safety |
| Vite | 6.2 | Build tool + HMR |
| Chakra UI | 3.15 | Component library |
| TailwindCSS | 4.2 | Utility styles |
| Zustand | 5.0 | State management |
| ECharts | - | Charts (equity curves, heatmaps) |
| Recharts | - | Simple charts |
| KlineCharts | - | Candlestick charts |

## Project Structure

```
ui/src/
  App.tsx              # Main app with routing
  main.tsx             # React DOM render
  types.ts             # Shared TypeScript types
  index.css            # Global styles + Tailwind

  components/
    layout/            # AppShell, PageHeader, MetricCard, StatusBadge, SectionCard, EmptyState
    alpha-lab/         # Alpha Lab workspace components
    charts/            # HeatmapChart
    ui/                # Chakra primitives (button, card, dialog, tabs, etc.)

    # Page-level components
    Dashboard.tsx          # Main dashboard
    GlobalOverview.tsx     # Trading overview
    SessionList.tsx        # Session list
    SessionDetail.tsx      # Session detail view
    NewSessionForm.tsx     # Create session form
    StrategyConfigForm.tsx # Strategy parameters
    Comparison.tsx         # Multi-session comparison
    Sidebar.tsx            # Navigation

    # Feature panels
    PortfolioManager.tsx   # Portfolio management
    AlphaLabWorkspace.tsx  # Alpha lab main
    OptimizerPanel.tsx     # Parameter optimization
    RiskPanel.tsx          # Risk analysis
    SimulationPanel.tsx    # Simulation controls
    MarketAdminPanel.tsx   # Market data admin
    IndustryHeatmap.tsx    # Industry heatmap
    AttributionPanel.tsx   # Performance attribution

  hooks/
    useWebSocket.ts    # WebSocket connection + auto-reconnect
    useChartData.ts    # Chart data processing
    useSearchSSE.ts    # Server-sent events for alpha search

  store/
    sessionStore.ts    # Zustand store (sessions, active session, UI state)

  utils/
    api.ts             # Fetch wrapper for REST API
    alphaApi.ts        # Alpha Lab API client
    display.ts         # Display formatting
    format.ts          # Data formatting
    metrics.ts         # Client-side metric calculations

  workers/
    chartWorker.ts     # Web Worker for chart data downsampling

  lttb.ts             # Largest-Triangle-Three-Buckets downsampling
```

## Key Components

### App Shell

`components/layout/AppShell.tsx` - Main layout with sidebar navigation.

### Alpha Lab Workspace

`components/AlphaLabWorkspace.tsx` - Factor research interface.

Sub-components in `components/alpha-lab/`:

| Component | Purpose |
|-----------|---------|
| `SearchTab.tsx` | Formula search with parameter controls |
| `SearchProgress.tsx` | Real-time search progress (SSE) |
| `FactorsTab.tsx` | Zoo factor management |
| `HistoryTab.tsx` | Search history |
| `MonitorTab.tsx` | Real-time monitoring |
| `ResearchTab.tsx` | Research tools |
| `PipelineView.tsx` | Pipeline stage visualization |
| `StrategyManager.tsx` | Search strategy management |
| `LLMAnalysis.tsx` | LLM analysis panel |
| `MiniChart.tsx` | Inline chart component |

### Session Management

- `SessionList.tsx` - Table of all sessions with status, metrics
- `SessionDetail.tsx` - Equity curve, trade list, positions, logs
- `NewSessionForm.tsx` - Strategy selection, parameters, date range
- `Comparison.tsx` - Side-by-side session comparison

## State Management

### Zustand Store (`store/sessionStore.ts`)

```typescript
interface SessionStore {
  sessions: Session[]
  activeSessionId: string | null
  // Actions
  fetchSessions: () => Promise<void>
  setActiveSession: (id: string) => void
  // ...
}
```

## Real-time Communication

### WebSocket Hook (`hooks/useWebSocket.ts`)

```typescript
const { connected, lastMessage } = useWebSocket(sessionId)
```

- Auto-reconnect on disconnect
- Handles ping/pong heartbeat
- Dispatches events to store

### SSE Hook (`hooks/useSearchSSE.ts`)

```typescript
const { progress, events } = useSearchSSE(jobId)
```

- Streams alpha search progress events
- Used by `SearchProgress` component

## API Clients

### REST (`utils/api.ts`)

Fetch wrapper with base URL configuration:

```typescript
api.get('/sessions')
api.post('/session/run', body)
api.delete(`/session/${id}`)
```

### Alpha Lab (`utils/alphaApi.ts`)

Specialized client for alpha lab endpoints:

```typescript
alphaApi.validate(formula)
alphaApi.searchDb(params)
alphaApi.getZoo(limit)
```

## Build & Development

```bash
cd ui
bun install          # Install dependencies
bun run dev          # Dev server on :5173
bun run build        # Production build
bun run preview      # Preview production build
```

### Docker Production Build

`Dockerfile.frontend` - Multi-stage build:
1. Bun install + build
2. Copy dist to Nginx
3. Serve on port 80

`nginx.conf` proxies `/api` and `/ws` to FastAPI on `:8000`.

## Performance Optimizations

- **Web Worker**: Chart data downsampling offloaded to `chartWorker.ts`
- **LTTB algorithm**: Largest-Triangle-Three-Buckets for efficient chart rendering
- **Virtualized lists**: `VirtualizedTradeList.tsx` for large trade histories
- **Batched WebSocket**: Equity and trade updates batched to reduce renders
