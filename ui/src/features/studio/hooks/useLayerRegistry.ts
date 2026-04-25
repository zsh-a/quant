/**
 * Bridges the registered chart layers (`layers/registry.ts`) to the
 * lightweight-charts instance owned by `ChartCanvas`.
 *
 * Mount/unmount cycle:
 *   - When a layer's `id` enters `visibleLayers`, we call `layer.mount()`
 *     against the chart context and keep the returned handle.
 *   - When it leaves, we call `handle.unmount()` and drop the handle.
 *   - On every timeline / currentBarIdx change, we call `update()` on every
 *     mounted handle.
 *
 * The chart context (`LayerCtx`) is allowed to be null while the canvas is
 * still booting; in that state the hook does nothing.
 */

import { useEffect, useRef } from 'react';
import type { LayerCtx, LayerHandle } from '../layers/types';
import { LAYERS, findLayer } from '../layers/registry';
import type { SessionTimeline } from '../types';

export function useLayerRegistry(
  ctx: LayerCtx | null,
  timeline: SessionTimeline | null,
  currentBarIdx: number,
  visibleLayers: ReadonlySet<string>,
) {
  // Map of layer id → mounted handle. Persists across renders so we can
  // diff mount/unmount across visibility changes.
  const handlesRef = useRef<Map<string, LayerHandle>>(new Map());

  // Mount / unmount on visibility change.
  useEffect(() => {
    if (!ctx) return;
    const handles = handlesRef.current;

    // Add newly visible layers.
    for (const layer of LAYERS) {
      if (visibleLayers.has(layer.id) && !handles.has(layer.id)) {
        handles.set(layer.id, layer.mount(ctx));
      }
    }
    // Remove no-longer-visible layers.
    for (const id of [...handles.keys()]) {
      if (!visibleLayers.has(id)) {
        const handle = handles.get(id);
        try {
          handle?.unmount();
        } catch {
          // already torn down — ignore
        }
        handles.delete(id);
      }
    }

    // After mount/unmount, push the current data into newly mounted layers
    // so they don't wait until the next data tick to render.
    if (timeline) {
      for (const [id, handle] of handles.entries()) {
        if (!findLayer(id)) continue;
        try {
          handle.update(timeline, currentBarIdx);
        } catch (e) {
          console.error(`[layers] update failed for "${id}"`, e);
        }
      }
    }
  }, [ctx, visibleLayers, timeline, currentBarIdx]);

  // Tear everything down when the canvas remounts or unmounts.
  useEffect(() => {
    // Capture the live Map; ref identity may change before cleanup runs.
    const handles = handlesRef.current;
    return () => {
      for (const handle of handles.values()) {
        try {
          handle.unmount();
        } catch {
          // chart already disposed
        }
      }
      handles.clear();
    };
  }, [ctx]);
}
