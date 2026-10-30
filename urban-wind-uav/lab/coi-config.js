// Settings for coi-serviceworker.js, which runs right after this file (build_lab.py).
// Reload only when the service worker is active, so the reloaded page gets the COOP/COEP headers.
window.coi = { doReload: () => navigator.serviceWorker.ready.then(() => window.location.reload()) };
