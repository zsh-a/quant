import { afterEach, beforeEach, describe, expect, it } from 'vitest';

// jsdom in this project ships an empty `localStorage` shim. Provide an
// in-memory replacement before the store module is imported so its module-load
// `readVisibleLayersFromStorage` call sees a working API.
function installLocalStorage() {
  const map = new Map<string, string>();
  Object.defineProperty(window, 'localStorage', {
    configurable: true,
    value: {
      getItem: (k: string) => (map.has(k) ? map.get(k)! : null),
      setItem: (k: string, v: string) => map.set(k, String(v)),
      removeItem: (k: string) => map.delete(k),
      clear: () => map.clear(),
      key: (i: number) => [...map.keys()][i] ?? null,
      get length() {
        return map.size;
      },
    },
  });
}
installLocalStorage();

const { LAYERS_STORAGE_KEY, hasStoredVisibleLayers, useStudioStore } = await import('../store');

const reset = () => useStudioStore.getState().actions.reset();

function clearStudioStorage() {
  try {
    window.localStorage.removeItem(LAYERS_STORAGE_KEY);
  } catch {
    /* jsdom variants without write support — non-fatal for tests */
  }
}

beforeEach(() => {
  clearStudioStorage();
  reset();
});

afterEach(() => {
  clearStudioStorage();
  reset();
});

describe('visibleLayers store', () => {
  it('starts empty when no localStorage entry exists', () => {
    expect(useStudioStore.getState().visibleLayers.size).toBe(0);
    expect(hasStoredVisibleLayers()).toBe(false);
  });

  it('toggleLayer adds and removes ids and persists to localStorage', () => {
    const { actions } = useStudioStore.getState();
    actions.toggleLayer('regime');
    expect(useStudioStore.getState().visibleLayers.has('regime')).toBe(true);
    expect(JSON.parse(window.localStorage.getItem(LAYERS_STORAGE_KEY) ?? '[]')).toEqual([
      'regime',
    ]);

    actions.toggleLayer('regime');
    expect(useStudioStore.getState().visibleLayers.has('regime')).toBe(false);
    expect(JSON.parse(window.localStorage.getItem(LAYERS_STORAGE_KEY) ?? '[]')).toEqual([]);
  });

  it('setVisibleLayers replaces the entire set and persists', () => {
    const { actions } = useStudioStore.getState();
    actions.setVisibleLayers(['regime', 'swings', 'fills']);
    expect([...useStudioStore.getState().visibleLayers].sort()).toEqual([
      'fills',
      'regime',
      'swings',
    ]);
    expect(hasStoredVisibleLayers()).toBe(true);
  });

  it('rejects malformed localStorage payloads silently', () => {
    window.localStorage.setItem(LAYERS_STORAGE_KEY, 'not-json');
    // Forcing a fresh read by toggling something — the read happens at module load
    // so we exercise the helper directly:
    expect(hasStoredVisibleLayers()).toBe(false);
  });
});
