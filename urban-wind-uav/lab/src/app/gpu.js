// The WebGPU device and the wind predictor of the page: the model download, the half-precision fallback, and a new
// device after the graphics device was lost or stopped answering.  A device or predictor that is replaced is
// destroyed (its GPU memory and its worker threads are let go).
import { DemoError, ErrorCode, hasCode, withCode, modelMisfit } from "../shared/errors.js";
import { download, downloadJson } from "../shared/download.js";
import { CELL_M, LAYER_M } from "../shared/units.js";
import { loadManifest } from "../city/city.js";
import { GpuContext } from "../gpu/context.js";
import { Predictor } from "../gpu/predictor.js";
import { MODEL_DIR, PREDICTOR_OPTIONS, REFERENCE_SPEED, CITY_GEOMETRY_LIMIT_MS, DEVICE_LIMIT_MS } from "./config.js";
import { state, sameSpeed } from "./state.js";
import { withLimit } from "./dom.js";
import { fatal } from "./status.js";

/** The wind predictor (gpu/predictor.js) on the current graphics device. */
export let predictor = null;
let context = null;

/** Make `nextContext` (with its predictor) the page's device; the model's precision is the device's. */
function adopt(nextContext, nextPredictor) {
  context = nextContext;
  predictor = nextPredictor;
  state.precision = nextContext.useF16 ? "half" : "single";
  const info = nextContext.adapterInfo;
  state.gpuInfo = info ? { vendor: info.vendor, architecture: info.architecture } : null;
  watchDevice(nextContext);
}

/** A new graphics device within DEVICE_LIMIT_MS (a device that comes later is let go when it comes). */
async function newContext(options = {}) {
  const pending = GpuContext.create(options);
  try {
    return await withLimit(pending, DEVICE_LIMIT_MS, "graphics device");
  } catch (error) {
    if (hasCode(error, ErrorCode.TIMEOUT)) pending.then((late) => late.destroy(), () => {});
    throw error;
  }
}

/** The wind model (the downloaded metadata and weights) on graphics device `ctx`. */
const predictorOn = (ctx) => new Predictor(ctx, state.metadata, state.weights, PREDICTOR_OPTIONS);

/** A predictor on `ctx`; when it cannot be built, `ctx` is let go. */
function predictorOrLetGo(ctx) {
  try {
    return predictorOn(ctx);
  } catch (error) {
    ctx.destroy();
    throw error;
  }
}

/** Let a device and its predictor go (their GPU memory and the predictor's worker threads): { context, predictor }. */
function letGo({ context, predictor }) {
  predictor?.destroy();
  context?.destroy();
}

/**
 * Create the graphics device, download the model and the city manifest, and build the predictor.  Returns the
 * manifest, or null when the browser has no WebGPU (the page then says why).  Throws DemoError GPU when the model's
 * shaders do not build on this device, DOWNLOAD when a file does not load, MODEL when the model or the city data do
 * not fit the page (checkData), TIMEOUT when the device does not answer.
 */
export async function prepareModel() {
  let candidate;
  try {
    candidate = await newContext();
  } catch (error) {
    if (hasCode(error, ErrorCode.TIMEOUT)) throw error;     // a device that did not answer: the page's reload message
    fatal(`${error.message}\n\nThis demo computes the wind on your graphics card with WebGPU. ` +
          "Please use a recent desktop Chrome or Edge.");
    return null;
  }
  try {
    const [metadata, weights, manifest] = await Promise.all([
      downloadJson(`${MODEL_DIR}/metadata.json`),
      download(`${MODEL_DIR}/weights.bin`).then((bytes) => bytes.buffer),
      loadManifest(),
    ]);
    checkData(metadata, manifest);
    Object.assign(state, { metadata, weights });
    // half-precision storage cannot be tested on every machine: if its shaders fail validation, the model is built
    // in single precision
    let attempt = await validatedPredictor(candidate);
    if (attempt.error && candidate.useF16) {
      console.warn("The half-precision wind model did not build; using single precision.");
      attempt.built.destroy();
      candidate.destroy();
      candidate = null;                // (not destroyed again below if the next device cannot be made)
      candidate = await newContext({ allowF16: false });
      attempt = await validatedPredictor(candidate);
    }
    if (attempt.error) {
      attempt.built.destroy();
      throw new DemoError(ErrorCode.GPU, `GPU validation failed: ${attempt.error.message}`);
    }
    adopt(candidate, attempt.built);
    state.manifest = manifest;          // the cities can be opened from now on (the predictor exists)
    return manifest;
  } catch (error) {
    if (candidate !== context) candidate?.destroy();     // a device the page did not take is let go
    throw error;
  }
}

/** Throws DemoError MODEL when the model or the city data do not fit the page's reference speed and grid units. */
function checkData(metadata, manifest) {
  const mismatch = (what, value, page) => modelMisfit(`${what} ${value} differs from the page's ${page}`);
  const pageSpeed = `${REFERENCE_SPEED} m/s`;
  const modelSpeed = metadata.wind?.reference_speed_m_s, profileSpeed = manifest.profile?.reference_speed_m_s;
  if (!sameSpeed(modelSpeed, REFERENCE_SPEED)) throw mismatch("model reference speed", `${modelSpeed} m/s`, pageSpeed);
  if (!sameSpeed(profileSpeed, REFERENCE_SPEED)) {
    throw mismatch("height profile reference speed", `${profileSpeed} m/s`, pageSpeed);
  }
  const { pixel_metres: cell, layer_metres: layer } = manifest.encoding ?? {};
  if (cell !== CELL_M || layer !== LAYER_M) {
    throw mismatch("city grid", `${cell} m x ${layer} m`, `${CELL_M} m x ${LAYER_M} m`);
  }
}

/** The predictor on `ctx` and the validation error of its shaders (null when they built). */
async function validatedPredictor(ctx) {
  ctx.device.pushErrorScope("validation");
  let built;
  try {
    built = predictorOn(ctx);
  } catch (error) {
    await ctx.device.popErrorScope();       // the scope is closed also when the model does not fit (MODEL)
    throw error;
  }
  return { built, error: await ctx.device.popErrorScope() };
}

let switching = null;
let recreating = null;                // the running device re-creation (recreateGpu), or null
/**
 * Half precision gave non-finite winds on this device: a single-precision device and predictor replace the current
 * ones (the city is opened on it, its geometry computed by the next prediction).  One switch at a time; a computation
 * or city load started meanwhile waits for it (precisionSwitch) instead of using the predictor being replaced.
 */
export function useSinglePrecision() {
  switching ??= switchToSingle().finally(() => { switching = null; });
  return switching;
}

/** The running precision switch (a promise that never rejects), or null. */
export const precisionSwitch = () => switching?.catch(() => {}) ?? null;

/** The switch of useSinglePrecision: a single-precision device and predictor take over, the old ones are let go. */
async function switchToSingle() {
  // a device re-creation running meanwhile decides the device first
  if (recreating) await recreating.catch(() => {});
  if (context && !context.useF16) return;          // already single precision (adopt set state.precision)
  const nextPredictor = predictorOrLetGo(await newContext({ allowF16: false }));
  try {
    if (state.city) nextPredictor.setCity(state.city);
  } catch (error) {
    letGo({ context: nextPredictor.context, predictor: nextPredictor });
    throw error;
  }
  const previous = { context, predictor };
  adopt(nextPredictor.context, nextPredictor);   // first: the old device's lost report is then ignored
  letGo(previous);
}

const lostContexts = new WeakSet();   // contexts whose device was lost

/** A lost graphics device (driver reset, sleep, another tab) is replaced before the next computation. */
function watchDevice(watched) {
  watched.device.lost.then(() => {
    lostContexts.add(watched);
    if (context !== watched) return;
    state.deviceLost = true;
    console.warn("The graphics device was lost; a new one is made for the next computation.");
  });
}

/**
 * A new graphics device and predictor (one re-creation at a time); withCity: the current city is opened on it.
 * Until it has worked, state.deviceLost stays set, so every computation first waits for (or retries) it.
 */
export function recreateGpu(withCity = true) {
  recreating ??= makeGpu(withCity).finally(() => { recreating = null; });
  return recreating;
}

/** One re-creation (recreateGpu): the old device goes, then a new device and predictor (with the city: withCity). */
async function makeGpu(withCity) {
  // the old device is let go first (a hung device would otherwise keep running beside the new one); it is no longer
  // the page's device then, so its lost report is ignored (watchDevice)
  state.deviceLost = true;
  const previous = { context, predictor };
  context = null;
  predictor = null;
  letGo(previous);
  let next;
  try {
    next = await newContext({ allowF16: state.precision !== "single" });
    adopt(next, predictorOrLetGo(next));
  } catch (error) {
    // a device or model that does not build again, or a device that does not answer: a GPU failure (the browser's
    // own errors carry no code)
    if (hasCode(error, ErrorCode.TIMEOUT)) throw new DemoError(ErrorCode.GPU, error.message);
    throw withCode(error, ErrorCode.GPU);
  }
  // a city load opens its own city on the new predictor (no second geometry computation)
  if (withCity && state.city) {
    const opened = predictor.setCity(state.city);
    await withLimit(opened.ready, CITY_GEOMETRY_LIMIT_MS, "city geometry");
  }
  state.deviceLost = lostContexts.has(next);   // only once the new device has worked, and was not lost meanwhile
}
