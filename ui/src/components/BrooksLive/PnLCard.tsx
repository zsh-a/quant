/**
 * PnLCard — virtual equity + PnL metrics + regime badge.
 */

import { MetricCard } from '../layout/MetricCard';
import type { EquityPoint, PositionPayload, RegimePayload } from '../../hooks/useBrooksLive';

interface PnLCardProps {
  equityCurve: EquityPoint[];
  positions: Record<string, PositionPayload>;
  regime: RegimePayload | null;
  analyst: string;
}

const fmtMoney = (v: number): string =>
  v.toLocaleString(undefined, { maximumFractionDigits: 2 });

export function PnLCard({ equityCurve, positions, regime, analyst }: PnLCardProps) {
  const last = equityCurve.length ? equityCurve[equityCurve.length - 1] : undefined;
  const first = equityCurve[0];
  const pnl = last && first ? last.total_equity - first.total_equity : 0;
  const pnlPct = last && first && first.total_equity ? pnl / first.total_equity : 0;
  const posCount = Object.keys(positions).length;

  return (
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
      <MetricCard
        label="Equity"
        value={last ? fmtMoney(last.total_equity) : '--'}
        hint={last ? `cash ${fmtMoney(last.cash)}` : '等待首次 equity 推送'}
      />
      <MetricCard
        label="PnL"
        value={
          <span className={pnl >= 0 ? 'text-emerald-500' : 'text-red-500'}>
            {pnl >= 0 ? '+' : ''}
            {fmtMoney(pnl)}
          </span>
        }
        hint={first ? `${(pnlPct * 100).toFixed(2)}%` : 'n/a'}
      />
      <MetricCard
        label="Positions"
        value={posCount}
        hint={analyst ? `analyst: ${analyst}` : '—'}
      />
      <MetricCard
        label="HTF regime"
        value={regime ? regime.regime : '—'}
        hint={regime ? `conf ${regime.confidence.toFixed(2)}` : '等待首个分类结果'}
      />
    </div>
  );
}
