// The particle worker (particles/worker.js) from the page's side: one worker for the page, the winds it holds, and
// the requests for a new particle system.  Random births in the corridor (or on the chosen plane), moved by the wind
// every frame, reborn at a new random point when their life ends - all in the worker, so the page never freezes.  The
// wind display's slices while the height slider moves are made there too (computeSlice): the worker holds the winds.
import { BAND_LOW_M, BAND_HIGH_M } from "../shared/units.js";
import { predictedDescriptor } from "../wind/samplers.js";
import { PARTICLE_LIMIT_MS } from "./config.js";
import { state, windScale } from "./state.js";
import { viewer } from "./scene.js";
import { cfdGrid } from "./cfd.js";
import { CityWorker } from "./city_worker.js";

const TRAIL_SPEED = 4;          // m/s: the background speed of the designed trail length (particles/system.js)
const MIN_TRAIL_SPEED = 0.5;    // m/s: the trail time scale of a (theoretical) still wind
// births: where the wind is slower than this share of the background speed, every birth point is taken; where it is
// faster, a point at speed s is taken with chance sqrt(share x background / s), so a slow region gets more particles
// and its short trails do not look sparse (particles/system.js spawn)
const SLOW_BIRTH_SHARE = 0.3;

// the worker: after a failure the next request starts a new one and the particles come back at once (onRestart), at
// most city_worker.js MAX_RESTARTS times in a row
const feed = new CityWorker(new URL("../particles/worker.js", import.meta.url), wire);
let particleId = 0, predictionCount = 0;
const predictionKeys = new WeakMap();    // prediction -> its number (the key of its wind in the worker)
const waiting = new Map();       // request id -> { resolve, timer }
// the slice being made { id, key, resolve, timer } and the one asked for after it (replaced by any later one), or null
let sliceRunning = null, sliceNext = null, sliceId = 0;
const held = new Map();          // the winds the worker holds: key -> kind
let onRestart = null;            // () -> draw the particles again after a restart

/** What to do after the worker failed and was restarted (the particles come back at once). */
export function setParticleRestart(handler) { onRestart = handler; }

/** A new wind is computed: particles still being set up for the previous wind are not drawn. */
export function invalidateParticles() { state.particleRequest = ++particleId; }

/** A new city: the particles, and the city and winds the worker holds, are let go. */
export function forgetParticleWinds() {
  endSlices({ replaced: true });
  if (!feed.worker) return;
  viewer.clearParticles();
  feed.worker.postMessage({ forget: true });
  held.clear();
  feed.forgetCity();             // the next request sends the new city
}

/**
 * The wind `source` for a request to the worker: { wind (windForWorker), evict }.  The worker keeps one wind per kind
 * (the newest prediction, the newest ground truth): an older one would keep its field (hundreds of MB on the large
 * city) alive, so it is told to evict it.
 */
function windRequest(source) {
  const wind = windForWorker(source);
  const evict = [];
  if (wind.kind) {
    for (const [key, kind] of held) if (kind === wind.kind) evict.push(key);
    for (const key of evict) held.delete(key);
    held.set(wind.key, wind.kind);
  }
  return { wind, evict };
}

/** The wind `source` ("ours" | "cfd") as a descriptor for the worker; its field only if the worker lacks this key. */
function windForWorker(source) {
  if (source === "cfd") {
    const file = state.cfdWindFile, grid = cfdGrid(file), key = `cfd:${file}`;     // the CFD wind shown
    return held.has(key) ? { key } : { kind: "cfd", key, grid };
  }
  const p = state.prediction, scale = windScale();
  if (!predictionKeys.has(p)) predictionKeys.set(p, ++predictionCount);
  const key = `predicted:${predictionKeys.get(p)}:${state.theta}:${scale}`;
  return held.has(key) ? { key } : { key, ...predictedDescriptor(p, state.city, state.theta, scale) };
}

/** The particle worker's replies: a system ready (or not), and each step's arrays. */
function wire(worker) {
  worker.onmessage = ({ data: m }) => {
    feed.answered();
    if ("sliceId" in m) {
      sliceAnswered(m);
    } else if ("ready" in m) {
      const request = waiting.get(m.id);
      if (request) {
        clearTimeout(request.timer);
        request.resolve(m);
      }
      waiting.delete(m.id);
    } else {
      viewer.receiveParticles(m.id, m.arrays);
    }
  };
  // (an error of a worker already replaced changes nothing)
  worker.onerror = (e) => {
    if (worker === feed.worker) failWorker(`particle worker failed: ${e.message || "it did not load"}`);
  };
}

/** The worker failed or stopped answering: every waiting request gets "no particles" (nothing waits for ever). */
function failWorker(reason) {
  console.warn(reason);
  for (const { resolve, timer } of waiting.values()) {
    clearTimeout(timer);
    resolve({ ready: false });
  }
  waiting.clear();
  endSlices(null);
  state.particleKey = null;      // the particles are gone: the next display of them makes them anew
  feed.endAfterFailure();        // the next request starts a new worker
  held.clear();
  viewer.clearParticles();
  if (feed.mayRestart) onRestart?.();
}

/**
 * A new particle system of `count` particles in the wind `source`, born in `polygon` (metres) or on `plane`.
 * Resolves to { id, count, arrays, ribbons, step } (viewer.setParticles; step(delta, arrays) asks the worker for the
 * next step, computed into `arrays`, which are transferred), or null when the worker could not make it.
 */
export function startParticles(source, polygon, count, seed, plane) {
  const worker = feed.ensure(state.city);
  const id = state.particleRequest = ++particleId;
  const { wind, evict } = windRequest(source);
  return new Promise((resolve) => {
    const step = (delta, arrays) => worker.postMessage({ step: { id, delta, arrays } }, arrays.map((a) => a.buffer));
    const timer = setTimeout(() => {
      if (waiting.has(id) && feed.worker === worker) {    // (not a worker since replaced)
        failWorker(`particle worker: no answer in ${PARTICLE_LIMIT_MS / 1000} s`);
      }
    }, PARTICLE_LIMIT_MS);
    // a system not made: the worker may not hold its wind either, so it is sent with its data next time
    const answer = (m) => {
      if (!m.ready) held.delete(wind.key);
      resolve(m.ready ? { id, count, arrays: m.arrays, ribbons: m.ribbons, step } : null);
    };
    waiting.set(id, { resolve: answer, timer });
    // the comets' length does not change with the background wind speed (particles/system.js)
    const timeScale = TRAIL_SPEED / Math.max(MIN_TRAIL_SPEED, state.speed);
    const slowSpeed = SLOW_BIRTH_SHARE * Math.max(MIN_TRAIL_SPEED, state.speed);
    const band = [BAND_LOW_M, BAND_HIGH_M];
    worker.postMessage({ init: { id, wind, evict, polygon, band, count, seed, plane, timeScale, slowSpeed } });
  });
}

/**
 * A slice of the wind `source` made in the worker (particles/slices.js): request = { kind: "h", windPoly, grid, z } or
 * { kind: "v", plane }.  One is made at a time: one asked for meanwhile waits, replaced by any later one, so a dragged
 * slider never queues work.  Resolves to { data[, ns, nz] }, { replaced: true } (a later request or another city), or
 * null when the worker failed or did not answer (the caller makes the slice itself).
 */
export function computeSlice(source, request) {
  return new Promise((resolve) => {
    const job = { source, request, resolve, prediction: state.prediction };
    if (!feed.mayRestart) {
      resolve(null);                   // the worker failed too often: no new one for slices (the caller makes them)
    } else if (!sliceRunning) {
      sendSlice(job);
    } else {
      sliceNext?.resolve({ replaced: true });
      sliceNext = job;
    }
  });
}

/**
 * Send a slice request to the worker (computeSlice); it fails the worker if no answer comes in time.  A request whose
 * prediction is gone meanwhile (a new computation started) is replaced.
 */
function sendSlice({ source, request, resolve, prediction }) {
  if (state.prediction !== prediction) { resolve({ replaced: true }); return; }
  const worker = feed.ensure(state.city), id = ++sliceId;
  const timer = setTimeout(() => {
    if (sliceRunning?.id === id && feed.worker === worker) {
      failWorker(`particle worker: no slice in ${PARTICLE_LIMIT_MS / 1000} s`);
    }
  }, PARTICLE_LIMIT_MS);
  const { wind, evict } = windRequest(source);
  sliceRunning = { id, key: wind.key, resolve, timer };
  worker.postMessage({ slice: { id, wind, evict, ...request } });
}

/** A slice from the worker: given to its request, then the waiting one is sent. */
function sliceAnswered(m) {
  if (sliceRunning?.id !== m.sliceId) return;          // a slice of a worker or city since let go
  clearTimeout(sliceRunning.timer);
  if (m.error) held.delete(sliceRunning.key);           // the worker may not hold this wind: its data goes next time
  sliceRunning.resolve(m.error ? null : m);
  sliceRunning = null;
  const next = sliceNext;
  sliceNext = null;
  if (next) sendSlice(next);
}

/** Every slice request ends with `answer` (another city: replaced; a failed worker: null). */
function endSlices(answer) {
  if (sliceRunning) {
    clearTimeout(sliceRunning.timer);
    sliceRunning.resolve(answer);
  }
  sliceNext?.resolve(answer);
  sliceRunning = sliceNext = null;
}
