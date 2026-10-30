// Route planning in four worker threads (planner/plan_worker.js), one per method: their start-up (the shared energy
// tables), the planning inputs of a mission, and one planning per computation.
import { CELL_M, BAND_LOW_M, BAND_HIGH_M, cellCentreM, cellOf } from "../shared/units.js";
import { DemoError, ErrorCode, errorText } from "../shared/errors.js";
import { sharedMemory, cpuCores } from "../shared/platform.js";
import { cfdWind, predictedDescriptor } from "../wind/samplers.js";
import { canonicalMoves, joinTables, TABLE_SPEED_M_S } from "../planner/lattice.js";
import { corridorShape } from "../planner/mission.js";
import { SAFE_CROP_MARGIN_M } from "../planner/clearance.js";
import { METHODS, PLAN_LIMIT_MS, PLANNER_START_LIMIT_MS, SPEED_MAX } from "./config.js";
import { state, cfdEntry, windScale } from "./state.js";
import { cfdGrid } from "./cfd.js";

// the planner's energy tables cover the winds of background speeds up to TABLE_SPEED_M_S (planner/lattice.js)
if (SPEED_MAX > TABLE_SPEED_M_S) throw new Error("the wind speed range exceeds the planner's energy tables");

let workers = [];                     // one planner worker per method, in METHODS order
const workerHelpers = [];             // helper threads of each planner worker
let plannerTables = null;             // the energy tables every worker holds (sent again to a replacement worker)
let planId = 0;
// shared with the planner workers: jobs with a smaller id are cancelled (cross-origin isolated pages only)
let planCancel = null;

/** A new planner worker (plan_worker.js). */
const newPlanner = () => new Worker(new URL("../planner/plan_worker.js", import.meta.url), { type: "module" });
/** The index of method `id`'s planner worker (one per config.js METHODS entry, in that order). */
const methodIndex = (id) => METHODS.findIndex((m) => m.id === id);

/**
 * The reply of `worker` to message id `id`: resolves to { data } (the reply), { error } (the worker failed: its
 * message) or { timeout: true } after `ms`; its listeners and timer go when it settles.  quiet: the worker's error is
 * not reported as an uncaught one as well (it is handled).
 */
function replyOf(worker, id, ms, quiet = false) {
  return new Promise((resolve) => {
    const finish = (outcome) => {
      clearTimeout(timer);
      worker.removeEventListener("message", handler);
      worker.removeEventListener("error", onError);
      resolve(outcome);
    };
    const handler = (event) => { if (event.data.id === id) finish({ data: event.data }); };
    const onError = (event) => {
      if (quiet) event.preventDefault();
      finish({ error: event.message || "unknown" });
    };
    const timer = setTimeout(() => finish({ timeout: true }), ms);
    worker.addEventListener("message", handler);
    worker.addEventListener("error", onError);
  });
}

/** A start-up reply of `worker` (message id `id`); rejects (DemoError PLANNER) on an error or after the time limit. */
async function startupReply(worker, id) {
  const reply = await replyOf(worker, id, PLANNER_START_LIMIT_MS);
  if (reply.data) return reply.data;
  const limit = PLANNER_START_LIMIT_MS / 1000;
  throw new DemoError(ErrorCode.PLANNER, reply.timeout ? `planner worker: no "${id}" in ${limit} s`
    : `planner worker error: ${reply.error}`);
}

let tablesBuild = null;               // the energy tables' build while it runs (startPlanners)

/** The planner workers' table build while it runs (it rejects when it failed), or null once they are ready. */
export const plannersPending = () => tablesBuild;

/**
 * The planner workers are created first, so their energy tables are built while the GPU, the model and the city
 * load.  Every worker builds a share of the moves in parallel (round robin); the page joins the shares and sends the
 * whole set to every worker (the same numbers as one full build).  plannersPending() is the build until every
 * worker holds them; a worker that fails or does not answer in time fails the preparation (state.plannerFailed).
 */
export function startPlanners() {
  workers = METHODS.map(() => newPlanner());
  tablesBuild = (async () => {
    // only the canonical moves are computed; the other tables are re-indexed copies (lattice.js joinTables, the
    // same numbers as a full build)
    const canon = canonicalMoves();
    // a parallel build on at most half of the processor's cores (the page must not take over the computer): the
    // planner workers themselves, plus temporary table builders when half the cores are more than the workers (from
    // 10 logical cores; they end after)
    const budget = Math.max(1, Math.min(canon.length, Math.floor(cpuCores() / 2)));
    const extraBuilders = Math.max(0, budget - workers.length);
    const temporary = Array.from({ length: extraBuilders }, () => newPlanner());
    const builders = [...workers.slice(0, budget), ...temporary];
    // builders.length <= budget <= canon.length: every builder has a share
    const shares = builders.map((_, w) => canon.filter((_, k) => k % builders.length === w));
    let parts;
    try {
      parts = await Promise.all(shares.map((share, w) => {
        const answer = startupReply(builders[w], "tables-part");
        builders[w].postMessage({ buildTables: share });
        return answer;
      }));
    } finally {
      temporary.forEach((w) => w.terminate());       // the temporary builders end in either case
    }
    const all = joinTables(parts);
    await Promise.all(workers.map((worker) => {
      const answer = startupReply(worker, "tables-set");
      worker.postMessage({ setTables: all });
      return answer;
    }));
    plannerTables = all;
  })().then(() => { tablesBuild = null; }, (error) => { state.plannerFailed = true; throw error; });
  // a failure is shown by the city load or the next computation (plannerFailed), so a rejection before either of
  // them awaits it is not reported as unhandled
  tablesBuild.catch(() => {});
}

/** The page cannot run (a start-up failure): the planner workers end (their helper threads end with them). */
export function stopPlanners() {
  for (const worker of workers) endWorker(worker, "the page stopped its route planners");
  workers = [];
}

/** End a planner worker: every reply still awaited from it (replyOf) ends now, not at its own time limit. */
function endWorker(worker, reason) {
  worker.terminate();
  worker.dispatchEvent(new ErrorEvent("error", { message: reason }));
}

// helper threads per planner (configurePlanners)
const MAX_OURS_HELPERS = 3, OTHER_HELPER_CORES = 8;

/**
 * Give the planners their helper threads and the shared cancel flag.  The predicted-wind route is the one the user
 * waits for after the prediction: it gets floor(cores / 4) - 1 helpers, one more from 12 logical cores, at most
 * MAX_OURS_HELPERS; the other three are planned while the wind is predicted and get 1 helper each from
 * OTHER_HELPER_CORES logical cores on (more helpers would oversubscribe the processor and slow the predicted-wind
 * route down).  The routes are the same for any number of helpers (one or more); with helpers the lattice search is
 * the faster one (planner/lattice.js latticeRoute: the same route as without helpers up to rounding).
 */
export function configurePlanners() {
  const cores = cpuCores();
  const oursHelpers = Math.max(0, Math.min(MAX_OURS_HELPERS, Math.floor(cores / 4) - 1 + (cores >= 12 ? 1 : 0)));
  const otherHelpers = cores >= OTHER_HELPER_CORES ? 1 : 0;
  workers.forEach((w, i) => {
    workerHelpers[i] = METHODS[i].id === "ours" ? oursHelpers : otherHelpers;
    w.postMessage({ helperThreads: workerHelpers[i] });
  });
  // superseded computations are cancelled in the planner workers (quick clicks would otherwise queue every old job)
  if (sharedMemory()) {
    planCancel = new Int32Array(new SharedArrayBuffer(4));
    workers.forEach((w) => w.postMessage({ cancelFlag: planCancel.buffer }));
  }
}

/** The planner workers cancel the jobs of every computation started so far. */
export const cancelPlanning = () => { if (planCancel) Atomics.store(planCancel, 0, planId + 1); };

/** The planning inputs that need no wind: the corridor geometry and the crop of the city's roof layers around it. */
function planGeometry() {
  const { city, mission } = state;
  const shape = corridorShape(mission.start, mission.goal);
  const xs = shape.polygon.map((p) => p[0]), ys = shape.polygon.map((p) => p[1]);
  const m = SAFE_CROP_MARGIN_M;
  const cx0 = Math.max(0, cellOf(Math.min(...xs) - m));
  const cy0 = Math.max(0, cellOf(Math.min(...ys) - m));
  const cx1 = Math.min(city.nx - 1, Math.ceil((Math.max(...xs) + m) / CELL_M));
  const cy1 = Math.min(city.ny - 1, Math.ceil((Math.max(...ys) + m) / CELL_M));
  const w = cx1 - cx0 + 1, h = cy1 - cy0 + 1;
  const layers = new Uint8Array(w * h);
  for (let y = 0; y < h; y++) {
    const from = (y + cy0) * city.nx + cx0;
    layers.set(city.layers.subarray(from, from + w), y * w);
  }
  return {
    start: mission.start, goal: mission.goal, band: [BAND_LOW_M, BAND_HIGH_M],
    // the planning space: the corridor's half-planes; halfMiddle = its widest half width, [sLo, sHi] along u from the
    // start
    corridor: { u: shape.u, n: shape.n, halfMiddle: shape.halfMiddle, sLo: shape.sLo, sHi: shape.sHi,
                planes: shape.planes },
    // the planning domain = the city's cell-centre range, as in the paper (trajectory positions bounded to the grid)
    domain: [cellCentreM(0), cellCentreM(city.nx - 1), cellCentreM(0), cellCentreM(city.ny - 1)],
    safe: { layers, x0: cx0, y0: cy0, w, h },
  };
}

/** The predicted wind as a descriptor for a worker (wind/samplers.js); its field is shared, not copied. */
function plannerPrediction() {
  const scale = windScale();
  return predictedDescriptor(state.prediction, state.city, state.theta, scale);
}

/** A planner worker that failed or did not answer is replaced by a new one with the same set-up. */
function replaceWorker(index, old) {
  if (workers[index] !== old) return;           // already replaced (several jobs waited on the same worker)
  endWorker(old, "the planner thread was replaced");
  const w = newPlanner();
  // (only runWorker replaces a worker, and a computation runs only after configurePlanners (main.js) and once the
  // tables exist (pipeline.js run waits for plannersPending))
  w.postMessage({ helperThreads: workerHelpers[index] });
  if (planCancel) w.postMessage({ cancelFlag: planCancel.buffer });
  w.postMessage({ setTables: plannerTables });
  workers[index] = w;
  console.warn("A route planner thread stopped or did not answer; it was replaced.");
}

/**
 * Plan in worker `index`: the mission geometry and the wind descriptor; the worker samples its wind grid.  A route that
 * takes longer than PLAN_LIMIT_MS, or a worker that fails, gives { error } instead of an endless wait.  Resolves to
 * the plan (plan_core.js: failed = NO_LATTICE_ROUTE | NO_SAFE_TRAJECTORY when the mission has no route), { error },
 * or { cancelled } (a newer computation took the worker).  A job of the current computation whose worker was replaced
 * because of another job (an older one that did not answer, or a worker that failed) runs once more on the new worker.
 */
function runWorker(index, geom, wind, id, again = false) {
  const worker = workers[index];
  const reply = replyOf(worker, id, PLAN_LIMIT_MS, true);
  worker.postMessage({ id, geom, wind });
  return reply.then(({ data, error, timeout }) => {
    if (data) return data.cancelled ? { cancelled: true } : data.error ? { error: data.error } : data.result;
    const replaced = workers[index] && workers[index] !== worker;
    if (!timeout && replaced && id === planId && !again) return runWorker(index, geom, wind, id, true);
    replaceWorker(index, worker);
    return { error: timeout ? `no answer in ${PLAN_LIMIT_MS / 1000} s` : `planner worker error: ${error}` };
  });
}

/**
 * Started before the wind prediction: the three routes that need no predicted wind (CFD wind, height profile, no
 * wind) are planned completely on their own workers while the graphics card predicts, and the predicted-wind worker
 * prepares everything that needs no wind (safe-height surfaces, lattice and its building clearance).  The setting's CFD
 * wind is here (pipeline.js run downloads it first).  Returns the planning: `cfd` = the CFD wind or null,
 * `finishedAt` = the finish time of each route.
 */
export function startPlanning(stale) {
  const id = ++planId;
  const geom = planGeometry();
  const scale = windScale();
  const oursIndex = methodIndex("ours");
  workers[oursIndex].postMessage({ id, geom, prepareOnly: true });
  const entry = cfdEntry();
  const cfd = entry ? cfdWind(cfdGrid(entry.file)) : null;
  const early = {}, finishedAt = {};
  const planning = { id, geom, oursIndex, cfd, early, finishedAt };
  const launch = (key, wind) => {
    // no wind for this method (e.g. no ground truth for this setting) or superseded: no route
    early[key] = routeOf(planning, key, Promise.resolve()
      .then(() => (!wind || stale() ? { points: null } : runWorker(methodIndex(key), geom, wind, id))));
  };
  launch("cfd", cfd && { kind: "cfd", grid: cfdGrid(entry.file) });
  launch("profile", { kind: "profile", profile: state.manifest.profile, theta: state.theta, scale });
  launch("agnostic", { kind: "none" });
  return planning;
}

/** Route `key` of `planning` from its plan's promise: a failure becomes { error }; its finish time. */
function routeOf(planning, key, plan) {
  return plan
    .catch((error) => ({ error: errorText(error) }))
    .then((r) => {
      planning.finishedAt[key] = performance.now();
      return r;
    });
}

/**
 * After the prediction: the predicted-wind route (its worker is already prepared).  Returns a function that waits for
 * all four routes: { ours, cfd, profile, agnostic }, or null when the computation was superseded.
 */
export function finishPlanning(planning, stale) {
  const oursPromise = routeOf(planning, "ours", Promise.resolve()
    .then(() => runWorker(planning.oursIndex, planning.geom, plannerPrediction(), planning.id)));
  return async () => {
    const ours = await oursPromise;
    if (stale()) return null;
    const keys = Object.keys(planning.early);
    const rest = await Promise.all(keys.map((k) => planning.early[k]));
    if (stale()) return null;
    const plans = { ours };
    keys.forEach((k, n) => { plans[k] = rest[n]; });
    return plans;
  };
}
