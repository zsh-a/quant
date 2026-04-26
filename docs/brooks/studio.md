# Brooks Studio — User Guide

> Modern K-line replay + live workspace for Brooks-style price action.
> Replaces the legacy `BrooksLive` panel. Path: `/studio`.

Brooks Studio is the unified live + replay frontend for any
`BrooksStrategy` paper-trading session. The same component stack drives
both modes — live mode is just "replay with `currentBarIdx === LIVE_TAIL`"
under the hood, so the layer registry, side panels, keyboard shortcuts,
and URL state work identically in either case.

## Quick Start

1. Start the backend stack:

   ```bash
   ./dev_local.sh up
   ```

2. Open the UI on `http://localhost:5173/` and navigate to **Trade** in
   the sidebar (route: `/studio`). The legacy `/brooks-live` URL keeps
   working — it client-side redirects to `/studio`.

3. On the landing page, fill in *Symbol*, *Interval*, *Analyst*
   (default `BTC/USDT` / `5m` / `rule`) and click **Start session**.
   Studio opens at `/studio/<session-id>` once the backend confirms.

4. The chart starts in **live tail** mode — every closed bar from the
   Celery worker WebSocket appends to the right edge.

To open a previously running session without restarting it, click any
entry under *Active sessions* on the landing page.

> **Mobile (< 768 px wide)**: Studio renders a read-only chart +
> scrubber only. Side panel, layer toggle, HTF inset, and playback
> controls are desktop-only.

## Anatomy

```
┌──────────────────────────────────────────┬──────────────────┐
│ ChartCanvas (lightweight-charts)         │ SidePanel        │
│  • all visible ChartLayers               │  • Signals       │
│  • LayerToggle popover (top-right)       │  • Decision      │
│  • HTF inset (top-right, foldable)       │  • Regime        │
│                                          │  • Compare       │
├──────────────────────────────────────────┴──────────────────┤
│ PnLStrip                                                    │
│ TimelineScrubber (regime-coloured background)               │
│ PlaybackControls + LayerToggle (bottom)                     │
└─────────────────────────────────────────────────────────────┘
```

* **ChartCanvas** owns one `IChartApi` instance. Every active layer
  hooks into the same chart through the `ChartLayer` registry.
* **TimelineScrubber** is the bar-level seek bar. The background renders
  a low-resolution regime band for visual orientation.
* **SidePanel** is a tab container (`Signals` / `Decision` / `Regime` /
  `Compare`) — switching tabs never resets selection / hover state.
* **PlaybackControls** drives `playState`, `speed`, and offers
  *jump-to-live* / *jump-to-start* shortcuts.

## Standard Layers

Each layer is independently toggleable from **LayerToggle**. Selection is
persisted to `localStorage` under the key `brooks-studio-layers`.

| Layer | Default | What it draws |
|-------|---------|---------------|
| `regime` | on | Regime bands along the bottom of the chart, coloured by `BrooksRegime` |
| `swings` | on | ▲ / ▼ markers on confirmed swing highs / lows |
| `ema` | on | EMA20 + EMA200 lines |
| `channels` | off | Micro-channel top / bottom fits as semi-transparent dashed lines |
| `signals` | on | ✦ markers per detector signal + entry / stop preview lines |
| `decisions` | on | Solid arrow + entry / stop / target lines for EV-gated `Decision`s |
| `fills` | off | ◆ markers for actual fills |
| `stop_adj` | off | Step-line history of stop upgrades |
| `htf_overlay` | off | Higher-TF swing projections (dashed) |
| `pnl` | on | Cumulative R sub-strip below the chart |
| `reasoning` | off | Per-bar LLM/VLM reasoning surfaced in tooltip + inspector |

## Keyboard Shortcuts

Disabled while focus is inside an `INPUT`, `TEXTAREA`, `SELECT`, or
`contentEditable` element.

| Key | Action |
|-----|--------|
| `←` / `→` | Step ±1 bar |
| `Shift + ←` / `Shift + →` | Step ±10 bars |
| `Space` | Toggle play / pause |
| `[` | Slow down (cycles `1× → 2× → 5× → 10×`) |
| `]` | Speed up |
| `End` | Jump to live tail (and pause) |
| `Home` | Jump to bar 0 |

## URL Parameters

Studio mirrors playback state into the URL via `replaceState`, so any
visible chart state is shareable. URL → store sync runs on mount and
whenever the search string changes externally.

| Param | Range | Notes |
|-------|-------|-------|
| `bar` | `0..N-1` | Omitted when at live tail |
| `mode` | `live` \| `replay` | Omitted when `live` |
| `speed` | `1` \| `2` \| `5` \| `10` | Omitted when `1` |

Layer visibility is **not** in the URL — it stays in `localStorage`
because users typically want it stable across sessions.

Example:

```
/studio/3f59bfe7?bar=412&mode=replay&speed=5
```

## Reviewing a Past Session

Studio is the same component for live and replay. To review a finished
session:

1. Open `/studio/<session-id>`.
2. Click the bar of interest in **TimelineScrubber**, or use
   `←` / `→` for fine stepping.
3. The right-hand **SidePanel** auto-updates:
   * `Decision` — full JSON + reasoning for the EV-gated decision at
     that bar (or `null`).
   * `Signals` — every detector signal up to and including that bar,
     filtered by source / side / probability.
   * `Regime` — horizontal band + transition list. Click a transition
     to jump there.
   * `Compare` — reruns selected analysts on the chosen bar (uses
     `POST /brooks-studio/sessions/{id}/replay-bar`).
4. Hovering a bar shows a "ghost" preview in the side panel without
   moving the chart cursor — the badge `preview` appears next to the
   bar count when this happens.

## Screenshots

Placeholders — capture from a live `BTC/USDT 5m` session:

```
docs/brooks/img/studio-landing.png        ← landing page
docs/brooks/img/studio-live.png           ← chart + signals tab
docs/brooks/img/studio-replay.png         ← replay with channels + decisions
docs/brooks/img/studio-compare.png        ← Compare tab side-by-side
```

## Troubleshooting

* **Blank chart, "Loading timeline…" never resolves.** Check the
  backend has the studio router mounted (`/brooks-studio/sessions/{id}/timeline`).
  In dev, watch the API server log.
* **WS reconnect loop.** Studio reconnects automatically; the chart
  shows the cached snapshot while the socket bounces.
* **Layer doesn't appear after toggle.** Confirm the `ChartLayer.id`
  matches the entry in the registry — IDs are case-sensitive.
* **Performance feels off on huge timelines.** Studio renders a 1500-bar
  trailing window first when the timeline exceeds 5000 bars, then
  hydrates the full history in an idle callback. If that hydration
  never fires, check that `requestIdleCallback` (or `setTimeout`
  fallback) isn't being shimmed away.

## Related

* [studio-extension.md](studio-extension.md) — How to add a layer, side
  panel tab, or `BarEvent` field.
* [README.md](README.md) — Brooks platform overview (analysts, decision
  pipeline, eval).
