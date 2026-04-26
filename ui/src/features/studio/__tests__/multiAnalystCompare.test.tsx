import { describe, expect, it } from 'vitest';
import { probabilityBucket } from '../components/side/MultiAnalystCompare';

describe('MultiAnalystCompare helpers', () => {
  it('buckets probabilities into low/med/high', () => {
    expect(probabilityBucket(0.1)).toBe('low');
    expect(probabilityBucket(0.39)).toBe('low');
    expect(probabilityBucket(0.4)).toBe('med');
    expect(probabilityBucket(0.59)).toBe('med');
    expect(probabilityBucket(0.6)).toBe('high');
    expect(probabilityBucket(1.0)).toBe('high');
  });

  it('returns em-dash for missing or invalid values', () => {
    expect(probabilityBucket(null)).toBe('—');
    expect(probabilityBucket(undefined)).toBe('—');
    expect(probabilityBucket(Number.NaN)).toBe('—');
  });
});
