import { useState, useEffect, useCallback } from 'react';
import type { StrategyMeta } from '../types';
import { apiFetch } from '../utils/api';

export function useStrategies() {
  const [strategies, setStrategies] = useState<StrategyMeta[]>([]);

  const fetchStrategies = useCallback(async () => {
    try {
      const resp = await apiFetch('/strategies');
      const data = await resp.json();
      setStrategies(data);
    } catch (err) {
      console.error('Failed to fetch strategies', err);
    }
  }, []);

  useEffect(() => {
    fetchStrategies();
  }, [fetchStrategies]);

  return { strategies, refetchStrategies: fetchStrategies };
}
