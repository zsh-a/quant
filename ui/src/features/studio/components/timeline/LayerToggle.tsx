/**
 * LayerToggle — checkbox panel for the registered chart layers.
 *
 * On first mount, if the user has no localStorage entry yet, we seed the
 * store from `DEFAULT_VISIBLE` (the union of layers whose `defaultVisible`
 * is `true`). After that the store is the source of truth and every toggle
 * is mirrored to localStorage by the store action.
 *
 * Adding a new layer to `LAYERS` makes it appear here automatically — no
 * edits required.
 */

import { useEffect, useState } from 'react';
import { ChevronLeft, Layers } from 'lucide-react';
import { Button } from '../../../../components/ui/button';
import {
  hasStoredVisibleLayers,
  useStudioActions,
  useVisibleLayers,
} from '../../store';
import { DEFAULT_VISIBLE, LAYERS } from '../../layers/registry';

export function LayerToggle() {
  const visible = useVisibleLayers();
  const { setVisibleLayers, toggleLayer } = useStudioActions();
  const [collapsed, setCollapsed] = useState(false);

  // First-time hydrate: if the user has no stored preference, seed defaults.
  useEffect(() => {
    if (hasStoredVisibleLayers()) return;
    setVisibleLayers(DEFAULT_VISIBLE);
    // intentionally one-shot at mount
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (collapsed) {
    return (
      <Button
        variant="secondary"
        size="icon"
        onClick={() => setCollapsed(false)}
        title="Show layers"
        aria-label="Show layers"
        data-testid="layer-toggle-expand"
      >
        <Layers className="size-4" />
      </Button>
    );
  }

  return (
    <div
      className="flex flex-col gap-1.5 rounded-md border border-border/70 bg-card/70 p-2 text-xs shadow-sm"
      data-testid="layer-toggle"
    >
      <div className="flex items-center justify-between gap-2 pb-1">
        <span className="flex items-center gap-1.5 font-medium text-foreground">
          <Layers className="size-3.5" /> Layers
        </span>
        <button
          type="button"
          onClick={() => setCollapsed(true)}
          className="rounded p-0.5 text-muted-foreground hover:bg-accent hover:text-foreground"
          aria-label="Collapse layers"
          title="Collapse"
        >
          <ChevronLeft className="size-3.5" />
        </button>
      </div>
      <div className="flex flex-col gap-1">
        {LAYERS.map((layer) => {
          const isOn = visible.has(layer.id);
          return (
            <label
              key={layer.id}
              className="flex cursor-pointer items-center gap-2 rounded px-1 py-1 hover:bg-accent/50"
            >
              <input
                type="checkbox"
                checked={isOn}
                onChange={() => toggleLayer(layer.id)}
                className="size-3.5 cursor-pointer accent-primary"
                aria-label={`Toggle layer ${layer.name}`}
                data-testid={`layer-toggle-${layer.id}`}
              />
              <span
                aria-hidden
                className="size-2.5 rounded-sm"
                style={{ backgroundColor: layer.swatch }}
              />
              <span className="text-foreground">{layer.name}</span>
            </label>
          );
        })}
      </div>
    </div>
  );
}
