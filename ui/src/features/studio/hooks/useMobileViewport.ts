/**
 * useMobileViewport — single matchMedia subscription for the
 * `(max-width: 768px)` breakpoint used by the Studio mobile read-only mode.
 *
 * Returns `false` in SSR / pre-mount renders so the desktop layout is the
 * default (avoids a layout flash on hydration).
 */

import { useEffect, useState } from 'react';

export const MOBILE_MAX_WIDTH_PX = 768;
export const MOBILE_MEDIA_QUERY = `(max-width: ${MOBILE_MAX_WIDTH_PX}px)`;

export function useMobileViewport(): boolean {
  const [isMobile, setIsMobile] = useState<boolean>(() => {
    if (typeof window === 'undefined') return false;
    return window.matchMedia(MOBILE_MEDIA_QUERY).matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined') return;
    const mql = window.matchMedia(MOBILE_MEDIA_QUERY);
    const onChange = (event: MediaQueryListEvent) => setIsMobile(event.matches);
    setIsMobile(mql.matches);
    if (typeof mql.addEventListener === 'function') {
      mql.addEventListener('change', onChange);
      return () => mql.removeEventListener('change', onChange);
    }
    // Safari < 14 fallback
    mql.addListener(onChange);
    return () => mql.removeListener(onChange);
  }, []);

  return isMobile;
}

// Backwards-compatible alias — `useMobileViewport` predates the rename.
export const useIsMobileViewport = useMobileViewport;
