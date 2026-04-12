import { useSyncExternalStore } from 'react';

const root = () => document.documentElement;

/**
 * Read a CSS custom property from :root, reactively updating when the
 * `dark` class toggles (which is the only thing that changes our vars).
 */
export function useCssVar(name: string): string {
  const value = useSyncExternalStore(
    subscribe,
    () => getComputedStyle(root()).getPropertyValue(name).trim(),
  );
  return value;
}

/** Convenience: return multiple CSS vars at once. */
export function useCssVars<K extends string>(...names: K[]): Record<K, string> {
  // Re-read all vars on any theme change
  const snapshot = useSyncExternalStore(subscribe, () => {
    const style = getComputedStyle(root());
    return names.map((n) => style.getPropertyValue(n).trim()).join('\0');
  });

  const values = snapshot.split('\0');
  const result = {} as Record<K, string>;
  names.forEach((n, i) => { result[n] = values[i]; });
  return result;
}

// Shared subscription: watches for class changes on <html> (dark toggle)
const listeners = new Set<() => void>();

let observer: MutationObserver | null = null;

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (!observer) {
    observer = new MutationObserver(() => {
      listeners.forEach((fn) => fn());
    });
    observer.observe(root(), { attributes: true, attributeFilter: ['class'] });
  }
  return () => {
    listeners.delete(cb);
    if (listeners.size === 0 && observer) {
      observer.disconnect();
      observer = null;
    }
  };
}
