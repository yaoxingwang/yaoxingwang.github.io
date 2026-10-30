// What the browser offers this page.

/** Shared memory between the page and its workers (a cross-origin isolated page). */
export const sharedMemory = () => Boolean(globalThis.crossOriginIsolated) && typeof SharedArrayBuffer !== "undefined";

/** The system's reduced-motion setting is on (style.css has the same media query). */
export const reducedMotion = () => globalThis.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;

/** The processor's logical cores (4 when the browser does not say). */
export const cpuCores = () => navigator.hardwareConcurrency || 4;
