/**
 * Registry of every chart layer Brooks Studio knows about.
 *
 * Order is the order they appear in the `LayerToggle` panel and the order
 * they mount onto the chart. Adding a new layer = export a `ChartLayer`
 * from a sibling file and append it here.
 *
 * Phase S5 will append: channelsLayer, stopAdjLayer, htfOverlayLayer,
 * reasoningLayer.
 */

import type { ChartLayer } from './types';
import { regimeLayer } from './regime';
import { swingsLayer } from './swings';
import { ema20Layer, ema200Layer } from './ema';
import { signalsLayer } from './signals';
import { decisionsLayer } from './decisions';
import { fillsLayer } from './fills';

export const LAYERS: readonly ChartLayer[] = [
  regimeLayer,
  swingsLayer,
  ema20Layer,
  ema200Layer,
  signalsLayer,
  decisionsLayer,
  fillsLayer,
];

export const DEFAULT_VISIBLE: ReadonlySet<string> = new Set(
  LAYERS.filter((l) => l.defaultVisible).map((l) => l.id),
);

export function findLayer(id: string): ChartLayer | undefined {
  return LAYERS.find((l) => l.id === id);
}

export type { ChartLayer, LayerCtx, LayerHandle } from './types';
