/**
 * DecisionLog — recent signals + collapsible analyst decision JSON.
 */

import { Fragment, useState } from 'react';
import { SectionCard } from '../layout/SectionCard';
import type { DecisionPayload } from '../../hooks/useBrooksLive';

interface DecisionLogProps {
  signals: DecisionPayload[];
}

export function DecisionLog({ signals }: DecisionLogProps) {
  const [expanded, setExpanded] = useState<number | null>(null);

  return (
    <SectionCard
      title="Decision log"
      description={`最近 ${signals.length} 个决策`}
    >
      {signals.length === 0 ? (
        <div className="py-6 text-center text-sm text-muted-foreground">暂无信号</div>
      ) : (
        <div className="max-h-[360px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border/50 text-xs text-muted-foreground">
                <th className="pb-2 text-left font-medium">Pattern</th>
                <th className="pb-2 text-left font-medium">Side</th>
                <th className="pb-2 text-right font-medium">P</th>
                <th className="pb-2 text-right font-medium">E[R]</th>
                <th className="pb-2 text-right font-medium">Entry</th>
                <th className="pb-2 text-right font-medium">Stop</th>
                <th className="pb-2 pl-2 text-left font-medium">Analyst</th>
              </tr>
            </thead>
            <tbody>
              {signals.map((s, i) => (
                <Fragment key={i}>
                  <tr
                    className="cursor-pointer border-b border-border/30 hover:bg-accent/30"
                    onClick={() => setExpanded(expanded === i ? null : i)}
                  >
                    <td className="py-1.5 font-mono text-xs">{s.pattern}</td>
                    <td className={s.side === 'long' ? 'py-1.5 text-emerald-500' : 'py-1.5 text-red-500'}>
                      {s.side}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs tabular-nums">
                      {s.probability.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs tabular-nums">
                      {s.expected_r.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs tabular-nums">
                      {s.entry_px.toFixed(2)}
                    </td>
                    <td className="py-1.5 text-right font-mono text-xs tabular-nums">
                      {s.stop_px.toFixed(2)}
                    </td>
                    <td className="py-1.5 pl-2 text-xs text-muted-foreground">{s.source}</td>
                  </tr>
                  {expanded === i && (
                    <tr>
                      <td colSpan={7} className="bg-accent/20 px-3 py-2">
                        <pre className="overflow-x-auto text-xs leading-snug text-muted-foreground">
                          {JSON.stringify(s, null, 2)}
                        </pre>
                      </td>
                    </tr>
                  )}
                </Fragment>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </SectionCard>
  );
}
