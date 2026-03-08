/**
 * Unified formatting utilities using Dinero.js for currency and consistent number display.
 */
import Dinero from 'dinero.js';

const USD = 'USD';
const PRECISION = 2;

/** Convert float to Dinero amount (minor units). */
function toAmount(value: number): number {
  return Math.round(value * Math.pow(10, PRECISION));
}

/** Format currency using Dinero.js - plain amount without sign. */
export function formatMoney(value: number | undefined | null, opts?: { symbol?: string; decimals?: number }): string {
  const n = Number(value ?? 0);
  const dec = opts?.decimals ?? PRECISION;
  if (!Number.isFinite(n)) return (opts?.symbol ?? '$') + '0.' + '0'.repeat(dec);
  const prec = dec;
  const amt = Math.round(n * Math.pow(10, prec));
  const d = Dinero({ amount: Math.abs(amt), currency: USD, precision: prec });
  const fmt = dec === 0 ? '$0,0' : '$0,0.' + '0'.repeat(dec);
  const formatted = d.toFormat(fmt);
  const sym = opts?.symbol ?? '$';
  return sym === '$' ? formatted : sym + formatted.replace(/^\$/, '');
}

/** Format signed currency (P&L style): +$6,400.27 or -$11,603.34. Zero displays as $0.00 (no sign). */
export function formatSignedMoney(value: number | string | undefined | null): string {
  const raw = value ?? 0;
  const n = typeof raw === 'string'
    ? parseFloat(String(raw).replace(/[,\s\u2212\u2013]/g, (m) => (m === '\u2212' || m === '\u2013' ? '-' : '')))
    : Number(raw);
  if (!Number.isFinite(n)) return '$0.00';
  const amount = toAmount(Math.abs(n));
  const d = Dinero({ amount, currency: USD, precision: PRECISION });
  const formatted = d.toFormat('$0,0.00').replace(/^\$/, '');
  if (amount === 0) return '$0.00';
  if (n > 0) return '+' + '$' + formatted;
  if (n < 0) return '-' + '$' + formatted;
  return '$' + formatted;
}

/** Format signed number (for percentages or plain numbers): +0.52% or -1.96%. Zero displays without sign. */
export function formatSigned(value: number | string | undefined | null, opts?: { asPercent?: boolean; decimals?: number }): string {
  const raw = value ?? 0;
  const n = typeof raw === 'string'
    ? parseFloat(String(raw).replace(/[,\s\u2212\u2013]/g, (m) => (m === '\u2212' || m === '\u2013' ? '-' : '')))
    : Number(raw);
  const dec = opts?.decimals ?? 2;
  if (!Number.isFinite(n)) return '0.' + '0'.repeat(dec) + (opts?.asPercent ? '%' : '');
  const rounded = Math.round(Math.abs(n) * Math.pow(10, dec)) / Math.pow(10, dec);
  const absStr = rounded.toLocaleString('en-US', { minimumFractionDigits: dec, maximumFractionDigits: dec });
  const suffix = opts?.asPercent ? '%' : '';
  if (rounded === 0) return absStr + suffix;
  if (n > 0) return '+' + absStr + suffix;
  if (n < 0) return '-' + absStr + suffix;
  return absStr + suffix;
}

/** Get CSS color from formatted sign string (for consistent green/red display). */
export function colorFromSign(str: string): string {
  const s = String(str).trim();
  if (s.startsWith('+') || (s.startsWith('$') && s.includes('+'))) return 'var(--success)';
  if (s.startsWith('-') || (s.startsWith('$') && s.charAt(1) === '-') || s.startsWith('\u2212')) return 'var(--danger)';
  return 'var(--text-dim)';
}

/** Get color from numeric value - use this for guaranteed consistency (avoids string parsing). */
export function colorFromValue(value: number | undefined | null): string {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return 'var(--text-dim)';
  if (n > 0) return 'var(--success)';
  if (n < 0) return 'var(--danger)';
  return 'var(--text-dim)';
}

/** Format percent: 0.0523 -> "5.23%" */
export function formatPercent(value: number | undefined | null, decimals = 2): string {
  const n = Number(value ?? 0);
  if (!Number.isFinite(n)) return '0.' + '0'.repeat(decimals) + '%';
  return (n * 100).toFixed(decimals) + '%';
}


/** Format trade/quote price for UI: round noise while keeping useful precision. */
export function formatPrice(value: number | string | undefined | null, decimals = 2): string {
  const raw = value ?? 0;
  const n = typeof raw === 'string' ? Number(raw) : Number(value ?? 0);
  if (!Number.isFinite(n)) return (0).toFixed(decimals);
  return n.toLocaleString('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  });
}
