/**
 * AnalystSwitch — segmented control for hot-swapping the running analyst.
 *
 * The available analysts are served by ``GET /brooks-live/analysts`` so
 * newly-registered ensembles appear in the UI without a redeploy.
 */

import { useEffect, useState } from 'react';
import { Button } from '../ui/button';
import { SectionCard } from '../layout/SectionCard';
import { apiFetch } from '../../utils/api';

interface AnalystSwitchProps {
  current: string;
  onSwitch: (analyst: string) => void | Promise<void>;
}

export function AnalystSwitch({ current, onSwitch }: AnalystSwitchProps) {
  const [options, setOptions] = useState<string[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const resp = await apiFetch('/brooks-live/analysts');
        if (!resp.ok) return;
        const data = await resp.json();
        if (alive) setOptions(data.analysts || []);
      } catch {
        /* ignore — offline mode shows current analyst only */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const handle = async (name: string) => {
    if (name === current) return;
    setBusy(name);
    setError(null);
    try {
      await onSwitch(name);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const items = options.length ? options : [current];
  return (
    <SectionCard
      title="Analyst"
      description="切换 analyst 路线，新信号立即来自对应实现"
    >
      <div className="flex flex-wrap gap-2">
        {items.map((name) => (
          <Button
            key={name}
            size="sm"
            variant={name === current ? 'default' : 'outline'}
            disabled={busy === name}
            onClick={() => handle(name)}
          >
            {busy === name ? '…' : name}
          </Button>
        ))}
      </div>
      {error && <div className="mt-2 text-xs text-red-500">{error}</div>}
    </SectionCard>
  );
}
