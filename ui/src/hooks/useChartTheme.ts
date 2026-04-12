import { useSyncExternalStore } from 'react';

const LIGHT = {
  stroke: '#4B52D5',
  grid: 'rgba(156, 163, 184, 0.25)',
  tooltipBg: 'rgba(255, 255, 255, 0.97)',
  tooltipBorder: '#D5D9E1',
  textDim: '#667085',
};

const DARK = {
  stroke: '#6B70E8',
  grid: 'rgba(40, 47, 63, 0.5)',
  tooltipBg: 'rgba(16, 20, 30, 0.97)',
  tooltipBorder: '#282F3F',
  textDim: '#7A828F',
};

function getSnapshot() {
  return document.documentElement.classList.contains('dark');
}

const listeners = new Set<() => void>();
let observer: MutationObserver | null = null;

function subscribe(cb: () => void) {
  listeners.add(cb);
  if (!observer) {
    observer = new MutationObserver(() => listeners.forEach((fn) => fn()));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
  }
  return () => {
    listeners.delete(cb);
    if (listeners.size === 0 && observer) {
      observer.disconnect();
      observer = null;
    }
  };
}

/** Returns resolved chart colors that react to dark/light toggle. */
export function useChartTheme() {
  const isDark = useSyncExternalStore(subscribe, getSnapshot);
  return isDark ? DARK : LIGHT;
}
