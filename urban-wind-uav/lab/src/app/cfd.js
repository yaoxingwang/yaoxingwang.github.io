// The CFD (ground-truth) wind fields of the current city, kept while the city is open: downloaded one at a time, the
// one a computation needs first (needCfd), the city's others after it in the background (queueCfd).
import { ErrorCode, hasCode } from "../shared/errors.js";
import { loadCfd, cfdBytes, downloadCancelled } from "../wind/cfd.js";
import { state, cfdEntry } from "./state.js";

const cfdCache = new Map();                   // file -> decoded grid
const cfdLoading = new Map();                 // file -> promise while it downloads
const cfdShare = new Map();                   // file -> share done (0..1), shown while a computation waits for it
const cfdAbort = new Map();                   // file -> AbortController of its download (cancelDownload)
// the share of the progress: the download 85 %, the decoding 15 % (decoding the large city's file takes seconds)
const DOWNLOAD_SHARE = 0.85, DECODE_SHARE = 0.15;
let active = null;                            // the download running: { entry, promise }
const queued = [];                            // the entries to download after it, in order

/** Whether the current setting has a CFD wind that is not here yet (downloaded and decoded). */
export const cfdPending = () => Boolean(cfdEntry()) && !cfdCache.has(cfdEntry().file);

/** The CFD wind grid of `file` (wind/cfd.js loadCfd), or undefined. */
export const cfdGrid = (file) => cfdCache.get(file);

/** The share done (0..1) of the CFD wind download of `file`. */
export const cfdShareDone = (file) => cfdShare.get(file) ?? 0;

/** Load (once) the CFD wind file of manifest entry `entry`. */
function ensureCfd(entry) {
  if (!entry || cfdCache.has(entry.file)) return Promise.resolve();
  if (!cfdLoading.has(entry.file)) {
    const controller = new AbortController();
    cfdAbort.set(entry.file, controller);
    const bytes = cfdBytes(entry);
    cfdShare.set(entry.file, 0);
    let got = 0, decoded = 0;
    const share = () => cfdShare.set(entry.file, DOWNLOAD_SHARE * got + DECODE_SHARE * decoded);
    const onBytes = (n) => { got = Math.min(1, n / bytes); share(); };
    const onDecode = (f) => { decoded = f; share(); };
    const signal = controller.signal;
    const loading = loadCfd(entry, state.manifest.cfd_encoding, { signal, onBytes, onDecode })
      .then((grid) => {
        if (signal.aborted) throw downloadCancelled(entry.file);    // a cancelled download is not kept
        cfdCache.set(entry.file, grid);
      })
      .finally(() => {
        if (cfdLoading.get(entry.file) !== loading) return;          // cancelled: the file may be loading anew
        cfdLoading.delete(entry.file);
        cfdAbort.delete(entry.file);
      });
    cfdLoading.set(entry.file, loading);
  }
  return cfdLoading.get(entry.file);
}

/** The download of `file` cancelled (its promise rejects with CANCELLED): the file is free to be loaded again. */
function cancelDownload(file) {
  cfdAbort.get(file)?.abort();
  cfdAbort.delete(file);
  cfdLoading.delete(file);
}

/** `entry` downloaded as the running download; the next queued one starts when it ends. */
function start(entry) {
  const promise = ensureCfd(entry);
  active = { entry, promise };
  promise
    .catch((error) => { if (!hasCode(error, ErrorCode.CANCELLED)) console.error(error); })
    .finally(() => {
      if (active?.entry !== entry) return;     // it gave way to another (needCfd) or its city was left
      active = null;
      startNext();
    });
  return promise;
}

/** The next queued download, unless one is running. */
function startNext() {
  if (active) return;
  const next = queued.shift();
  if (next) start(next);
}

/**
 * The CFD wind of `entry` wanted now (a computation waits for it): downloaded at once if it is not here; a download of
 * another one running gives way (stopped, and done again after this one).  Resolves when it is here; rejects when its
 * download failed.
 */
export function needCfd(entry) {
  if (!entry || cfdCache.has(entry.file)) return Promise.resolve();
  if (active?.entry === entry) return active.promise;
  if (queued.includes(entry)) queued.splice(queued.indexOf(entry), 1);
  if (active) {
    const { entry: other } = active;
    active = null;
    queued.unshift(other);
    cancelDownload(other.file);
  }
  return start(entry);
}

/** The CFD winds of `entries` downloaded in the background, one at a time, in this order, after the running one. */
export function queueCfd(entries) {
  for (const entry of entries) {
    if (!cfdCache.has(entry.file) && active?.entry !== entry && !queued.includes(entry)) queued.push(entry);
  }
  startNext();
}

/** A new city opens: the other cities' CFD winds are let go and their downloads still running or queued cancelled. */
export function forgetOtherCities(entry) {
  const ofCity = (file) => (entry.cfd || []).some((c) => c.file === file);
  for (const file of [...cfdCache.keys()]) if (!ofCity(file)) cfdCache.delete(file);
  for (const file of [...cfdShare.keys()]) if (!ofCity(file)) cfdShare.delete(file);
  for (const file of [...cfdAbort.keys()]) if (!ofCity(file)) cancelDownload(file);
  for (let k = queued.length - 1; k >= 0; k--) if (!ofCity(queued[k].file)) queued.splice(k, 1);
  if (active && !ofCity(active.entry.file)) active = null;
}
