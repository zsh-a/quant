/**
 * Verify the registry hook mounts a layer when its id enters
 * `visibleLayers`, calls `update` on data changes, and unmounts when the
 * id leaves the set. We feed the hook a tiny synthetic layer so we can
 * assert directly on the lifecycle calls — no chart instance needed.
 */
import { describe, expect, it, vi } from 'vitest';
import { renderHook } from '@testing-library/react';
import type { ChartLayer, LayerCtx } from '../layers/types';
import type { SessionTimeline } from '../types';
import { makeTimeline } from './fixtures';

vi.mock('../layers/registry', () => {
  const lifecycle = {
    mount: vi.fn(),
    update: vi.fn(),
    unmount: vi.fn(),
  };
  const fakeLayer: ChartLayer = {
    id: 'fake',
    name: 'Fake',
    swatch: '#fff',
    defaultVisible: false,
    mount(_ctx: LayerCtx) {
      void _ctx;
      lifecycle.mount();
      return {
        update: (...args: unknown[]) => lifecycle.update(...args),
        unmount: () => lifecycle.unmount(),
      };
    },
  };
  return {
    LAYERS: [fakeLayer],
    DEFAULT_VISIBLE: new Set<string>(),
    findLayer: (id: string) => (id === 'fake' ? fakeLayer : undefined),
    __lifecycle: lifecycle,
  };
});

import { useLayerRegistry } from '../hooks/useLayerRegistry';
import * as registry from '../layers/registry';

const lifecycle = (registry as unknown as { __lifecycle: { mount: ReturnType<typeof vi.fn>; update: ReturnType<typeof vi.fn>; unmount: ReturnType<typeof vi.fn> } }).__lifecycle;

const fakeCtx = { chart: {}, primarySeries: {}, theme: 'dark' } as unknown as LayerCtx;

describe('useLayerRegistry', () => {
  it('mounts when ctx + visibleLayers contains the id; unmounts when removed', () => {
    lifecycle.mount.mockClear();
    lifecycle.update.mockClear();
    lifecycle.unmount.mockClear();

    const tl: SessionTimeline = makeTimeline(2);
    const visible = new Set<string>(['fake']);

    const { rerender, unmount } = renderHook(
      ({ ctx, vis, idx }: { ctx: LayerCtx | null; vis: Set<string>; idx: number }) =>
        useLayerRegistry(ctx, tl, idx, vis),
      { initialProps: { ctx: fakeCtx, vis: visible, idx: 0 } },
    );
    expect(lifecycle.mount).toHaveBeenCalledTimes(1);
    expect(lifecycle.update).toHaveBeenCalled();

    // Move the cursor — should re-call update.
    lifecycle.update.mockClear();
    rerender({ ctx: fakeCtx, vis: visible, idx: 1 });
    expect(lifecycle.update).toHaveBeenCalled();

    // Hide the layer — should unmount.
    rerender({ ctx: fakeCtx, vis: new Set<string>(), idx: 1 });
    expect(lifecycle.unmount).toHaveBeenCalledTimes(1);

    unmount();
  });
});
