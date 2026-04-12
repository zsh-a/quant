import { useState, useEffect, useCallback } from 'react';
import type { BenchmarkData, SessionSummary } from '../types';
import { API_BASE } from '../utils/api';

export const AVAILABLE_BENCHMARKS = [
  { code: 'sh.000300', name: 'HS300' },
  { code: 'sh.000905', name: 'ZZ500' },
  { code: 'sz.399006', name: 'ChiNext' },
];

export function useBenchmarks(primarySession: SessionSummary | undefined) {
  const [selectedBenchmarks, setSelectedBenchmarks] = useState<string[]>([]);
  const [benchmarksData, setBenchmarksData] = useState<Record<string, BenchmarkData[]>>({});

  const toggleBenchmark = useCallback((code: string) => {
    setSelectedBenchmarks((prev) =>
      prev.includes(code) ? prev.filter((c) => c !== code) : [...prev, code]
    );
  }, []);

  const fetchBenchmarks = useCallback(async () => {
    if (!primarySession?.start_date) {
      setBenchmarksData({});
      return;
    }

    const newData: Record<string, BenchmarkData[]> = {};
    await Promise.all(
      selectedBenchmarks.map(async (benchmarkCode) => {
        try {
          let url = `${API_BASE}/market/benchmark?symbol=${benchmarkCode}&start_date=${primarySession.start_date}`;
          if (primarySession.end_date) url += `&end_date=${primarySession.end_date}`;
          const resp = await fetch(url);
          if (resp.ok) {
            newData[benchmarkCode] = await resp.json();
          }
        } catch (err) {
          console.error(`Failed to fetch benchmark ${benchmarkCode}`, err);
        }
      })
    );
    setBenchmarksData(newData);
  }, [primarySession?.id, primarySession?.start_date, primarySession?.end_date, selectedBenchmarks]);

  useEffect(() => {
    if (selectedBenchmarks.length > 0) {
      fetchBenchmarks();
    } else {
      setBenchmarksData({});
    }
  }, [selectedBenchmarks, fetchBenchmarks]);

  return {
    selectedBenchmarks,
    benchmarksData,
    toggleBenchmark,
    availableBenchmarks: AVAILABLE_BENCHMARKS,
  };
}
