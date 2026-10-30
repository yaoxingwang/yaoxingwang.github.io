// Wind particles in a worker, so moving thousands of particles every frame never freezes the page.  The worker owns
// the ParticleSystem (system.js) and also fills the arrays the viewer draws from (per trail point: position and
// speed, alpha; see fill), so the main thread only hands them to the GPU.  The wind arrives as a descriptor
// (wind/samplers.js windFromDescriptor); in a cross-origin isolated page its field is shared memory, not a copy, and
// a wind already sent is kept by its key.
// Messages: { city } (the buildings, sent once per city); { init: { id, wind, evict, polygon, band, count, seed, plane,
// timeScale, slowSpeed } } -> { id, ready, arrays, ribbons } (ribbons.js: the ribbons' arrays for the viewer);
// { step: { id, delta, arrays } } -> { id, arrays } (always answered); { forget } (a new city: its winds go);
// { slice: { id, wind, evict, kind: "h", windPoly, grid, z } or { ..., kind: "v", plane } } -> { sliceId, data[, ns,
// nz] } or { sliceId, error } (slices.js: the wind display's slices, made here because the winds are here).
import { windFromDescriptor } from "../wind/samplers.js";
import { ParticleSystem, TRAIL, POINT_TEXTURE_WIDTH } from "./system.js";
import { ribbonArrays } from "./ribbons.js";
import { horizontalSlice, verticalSlice } from "./slices.js";

const TRAIL_FADE = 1.6;        // a trail point's opacity: (1 - its age in the trail)^TRAIL_FADE
const winds = new Map();       // key -> sampler: the winds the page has sent and not yet evicted
let system = null, current = 0, city = null;

/**
 * The trail points of `sys` into the arrays `out` [points, alphas]: per trail point j, points[4 j ..] = position,
 * speed; alphas[j] = opacity.  The viewer's vertex shader takes the neighbours from the same texture, so each step
 * sends little data to the GPU.
 */
function fill(sys, out) {
  const [P, A] = out;
  const pos = sys.pos, sp = sys.speed;
  for (let i = 0; i < sys.count; i++) {
    const life = sys.lifeAlpha(i), filled = sys.filled[i];
    for (let k = 0; k < TRAIL; k++) {
      const j = i * TRAIL + k, o = j * 3;
      // trail points outside the corridor, the band or in a building are hidden (vis 0), the rest fade as usual
      A[j] = k < filled && sys.vis[j] ? life * Math.pow(1 - k / (TRAIL - 1), TRAIL_FADE) : 0;
      P[4 * j] = pos[o]; P[4 * j + 1] = pos[o + 1]; P[4 * j + 2] = pos[o + 2]; P[4 * j + 3] = sp[j];
    }
  }
}

/** Texture-sized arrays (whole rows of POINT_TEXTURE_WIDTH) for `count` particles. */
function pointArrays(count) {
  const texels = Math.ceil(count * TRAIL / POINT_TEXTURE_WIDTH) * POINT_TEXTURE_WIDTH;
  return [new Float32Array(texels * 4), new Float32Array(texels)];
}

/**
 * The sampler of `wind` (a descriptor with its data, built and kept by key; or { key } of one sent before), after the
 * winds the page no longer needs (`evict`, keys) are let go.
 */
function samplerOf(wind, evict) {
  for (const key of evict) winds.delete(key);
  if (wind.kind) winds.set(wind.key, windFromDescriptor(wind));
  const sampler = winds.get(wind.key);
  if (!sampler) throw new Error("no wind for this request");
  return sampler;
}

self.onmessage = (event) => {
  const m = event.data;
  if (m.city) { city = m.city; return; }
  if (m.forget) {              // a new city: the old city and its winds are not needed
    winds.clear(); system = null; current = 0; city = null;
    return;
  }
  if (m.init) {
    const { id, wind, evict, polygon, band, count, seed, plane, timeScale, slowSpeed } = m.init;
    let out, built, ribbons;
    try {
      const options = { plane, timeScale, slowSpeed };
      built = new ParticleSystem(samplerOf(wind, evict), city, polygon, band, count, seed, options);
      out = pointArrays(count);
      fill(built, out);
      ribbons = ribbonArrays(count);
    } catch (error) {             // answered, never left waiting; the system shown so far keeps moving
      console.warn("particles:", error.message);
      self.postMessage({ id, ready: false });
      return;
    }
    system = built; current = id;
    self.postMessage({ id, ready: true, arrays: out, ribbons },
                     [...out.map((a) => a.buffer), ribbons.side.buffer, ribbons.taper.buffer, ribbons.index.buffer]);
    return;
  }
  if (m.slice) {
    const { id, wind, evict, kind } = m.slice;
    try {
      const sampler = samplerOf(wind, evict);
      const slice = kind === "h" ? { data: horizontalSlice(city, sampler, m.slice.windPoly, m.slice.grid, m.slice.z) }
        : verticalSlice(city, sampler, m.slice.plane);
      self.postMessage({ sliceId: id, ...slice }, [slice.data.buffer]);
    } catch (error) {
      self.postMessage({ sliceId: id, error: error.message });
    }
    return;
  }
  if (m.step) {
    const { id, delta, arrays } = m.step;
    // a step for another system is answered with its arrays unchanged: the viewer waits for every answer
    if (id !== current || !system) { self.postMessage({ id, arrays }, arrays.map((a) => a.buffer)); return; }
    try {
      system.step(delta);
      fill(system, arrays);
    } catch (error) {
      console.warn("particles:", error.message);
    }
    self.postMessage({ id, arrays }, arrays.map((a) => a.buffer));     // always answered, or the particles would freeze
  }
};
