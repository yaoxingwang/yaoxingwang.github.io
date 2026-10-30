// One computation for a mission: the 3-D wind prediction in the corridor (WebGPU) with the prediction sweep, the wind
// display, four route plans in parallel workers, then every route is scored in the CFD wind when it exists for this
// city, speed and direction (otherwise the panel shows the routes' geometry, no energy).  A newer computation
// supersedes an older one.
import { DemoError, hasCode, ErrorCode } from "../shared/errors.js";
import { sharedMemory } from "../shared/platform.js";
import { predictedWind, profileWind, noWind } from "../wind/samplers.js";
import { trajectoryEnergy } from "../planner/energy.js";
import { randomMission } from "../planner/mission.js";
import { METHODS, SWEEP_RANGE_RATIO, SWEEP_EVERY, PREDICT_OVERLAP, PREDICT_LIMIT_MS } from "./config.js";
import { state, cfdEntry, windScale, windKey, predictionKey } from "./state.js";
import { nextFrame, withLimit, announce } from "./dom.js";
import { viewer, toCells, corridorCells, shownCorridor } from "./scene.js";
import { cfdPending, needCfd } from "./cfd.js";
import { status, progressStart, progressFinish, recordProgress, showCfdLoading, COMPUTING_MDDF } from "./status.js";
import { showSkeleton, showFailure, showRoutes, setRetry, resultsSummary } from "./results.js";
import { applyDisplay, drawWind } from "./wind_display.js";
import { invalidateParticles } from "./particle_feed.js";
import { drawSweep, setSweepOpen, endSweep } from "./sweep.js";
import { cancelPlanning, startPlanning, finishPlanning, plannersPending } from "./planning.js";
import { predictor, recreateGpu, useSinglePrecision, precisionSwitch } from "./gpu.js";
import { validatePlans } from "../local/validate_plans.js";

const NAN_CHECK_STRIDE = 97;      // one prediction value in this many is checked for non-finite values (half precision)
const MIN_PROGRESS_MS = 300;      // the shortest expected time of the next progress bar
const PREDICTING = "Predicting the corridor-localized 3D wind field…";

let runToken = 0;
let runMission = null;             // the mission the newest computation computes

/** The mission the newest computation computes (or computed), or null before the first. */
export const computedMission = () => runMission;

/** The running computation is superseded (a city is being loaded). */
export function supersedeRun() { runToken += 1; }

/** Compute the current mission: predict = a new wind prediction (else the stored one is reused when it fits). */
export async function run({ predict = true } = {}) {
  const token = ++runToken;
  const stale = () => token !== runToken;
  if (!beginRun()) return;
  // no wait at all in the usual case: the computation shows at once
  if ((precisionSwitch() || state.deviceLost) && !(await deviceReady(stale))) {
    if (!stale()) state.busy = false;  // the device could not be made: its failure card shows, nothing computes
    return;
  }
  // the planner workers' energy tables: still building only for a computation started after a failed first city
  const tables = plannersPending();
  if (tables) {
    await tables.catch(() => {});
    if (stale()) return;
    if (!beginRun()) return;           // the planner did not start: its failure card
  }
  // the ground truth of this setting not here yet (a speed whose download has not ended): downloaded first, and only
  // then computed, as every other computation
  if (cfdPending() && !(await groundTruthReady(stale))) return;
  const { city, mission } = state;    // read after the waits: what the user set meanwhile is computed
  runMission = mission;
  // the stored prediction is used only for the corridor and direction it was predicted for
  const predictFor = predictionKey(mission);
  const stored = state.prediction;
  const needPredict = predict || !stored || stored.city !== city || stored.predictedFor !== predictFor;
  // stage: for the failure message, "wind" or "planning"
  const job = { city, mission, stale, needPredict, started: performance.now(), predictMs: 0, stage: "wind",
                planning: null };
  try {
    showStart(mission, needPredict);
    // the routes that need no predicted wind start now, in parallel with the prediction
    job.planning = startPlanning(stale);
    if (needPredict) {
      const result = await predictWind(mission, stale);
      if (stale()) return;
      if (hasNonFinite(result.values) && state.precision === "half") return await runInSinglePrecision(stale);
      state.prediction = Object.assign(result, { city, predictedFor: predictFor });
      job.predictMs = result.milliseconds;
    }
    await planAndShow(job);
  } catch (error) {
    console.error(error);
    if (!stale()) showRunFailure(job.stage, job.stage === "wind" ? () => run({ predict: true }) : null);
  } finally {
    if (!stale()) state.busy = false;
    collectWorkerGarbage();
  }
}

/**
 * After the prediction of `job` (run): the predicted wind shown, the routes collected and checked, the results and
 * the routes shown.  Sets job.stage as it goes; returns early when the computation is superseded.
 */
async function planAndShow(job) {
  const { city, mission, stale, planning } = job;
  const scale = windScale();
  const predicted = predictedWind(state.prediction, city, state.theta, scale);
  // the predicted-wind route first: its worker was prepared during the prediction
  const collectPlans = finishPlanning(planning, stale);
  setWinds(predicted, planning.cfd, scale);
  await nextFrame();
  if (stale()) return;
  endSweep();                         // a slice still in the sweep worker is not drawn
  drawWind();
  job.stage = "planning";
  showPlanning(job.needPredict);
  await nextFrame();
  if (stale()) return;
  const plans = await collectPlans();
  if (!plans || stale()) return;
  const plansAt = performance.now();
  checkPlans(plans);
  validatePlans(plans, city, mission);
  storeResults(plans, planning.cfd, predicted, mission, shownTiming(planning, job.started, plansAt, job.predictMs));
  // the next progress bar's duration, from this computation
  recordProgress(job.predictMs > 0 ? "full" : "planOnly", Math.max(MIN_PROGRESS_MS, plansAt - job.started), city);
  await progressFinish();
  if (stale()) return;
  // no safe predicted-wind route: the mission failed
  if (!plans.ours?.points) {
    showFailure("route");
  } else {
    showRoutes();
    // for a screen reader (with Ground truth off there are no numbers)
    announce(resultsSummary() || "The routes are ready.", "results");
  }
}

/**
 * Before a computation: superseded jobs are cancelled and the last failure's retry is forgotten; false when there is
 * nothing to compute (no city or mission yet, a city being loaded - it starts its own mission) or nothing can finish
 * (the planner never started: its failure card).
 */
function beginRun() {
  cancelPlanning();                   // the planner workers cancel the jobs of superseded computations
  state.busy = false;                 // set again when this run computes (a superseded run may have left it set)
  state.results = null;              // a failed retry must never resurrect the previous run's export
  // nothing to compute: the failure card shown keeps its button's action
  if (!state.city || !state.mission || state.cityLoading) return false;
  if (state.plannerFailed) {
    setRetry(() => location.reload());
    showFailure("startup");
    return false;
  }
  setRetry(null);                     // a retry belongs to the failure it was made for
  return true;
}

/** The results of `mission`'s computation: the plans scored in the CFD wind if there is one, else the predicted. */
function storeResults(plans, cfd, predicted, mission, timing) {
  scorePlans(plans, cfd || predicted, mission);
  state.results = { plans, scoredIn: cfd ? "cfd" : "ours", mission, wind: windKey(), mode: state.mode, timing };
  window.dispatchEvent(new CustomEvent("geowind:results", { detail: state.results }));
}

/** Half precision gave non-finite winds: the computation again in single precision. */
async function runInSinglePrecision(stale) {
  console.warn("Half precision gave non-finite wind values; switching to single precision.");
  await useSinglePrecision();
  if (stale()) return;
  return run({ predict: true });
}

/** The routes are being optimized: the status, and the progress bar when there was no prediction (a speed change). */
function showPlanning(needPredict) {
  const routeCount = METHODS.filter((m) => m.id !== "cfd" || cfdEntry()).length;   // no CFD wind: no ground-truth route
  status(`Optimizing ${routeCount} trajectories…`, "progress");
  if (!needPredict) progressStart("planOnly");
}

/** A computation starts: the results skeleton, the wind display cleared (a new prediction), the corridor and pins. */
function showStart(mission, needPredict) {
  state.busy = true;
  showSkeleton();
  if (needPredict) clearWindDisplay();
  viewer.setCorridor(shownCorridor(mission));
  viewer.setPins(mission.start, mission.goal);
  if (state.frameNext) {
    viewer.frameCity();               // the whole city, not aimed at the start and goal
    state.frameNext = false;
  }
}

/** The winds of this computation (state.winds): the prediction, the CFD wind (or null), the height profile, none. */
function setWinds(predicted, cfd, scale) {
  const profile = profileWind(state.manifest.profile, state.theta, scale);
  state.winds = { ours: predicted, cfd, profile, agnostic: noWind() };
  state.cfdWindFile = cfdEntry()?.file ?? null;     // the setting of this computation (a newer one supersedes it)
}

/**
 * Before a computation whose ground truth is not here yet: its download first (cfd.js needCfd: ahead of any other),
 * its share shown; false when it failed (the "cfd" card: Try again downloads it again) or was superseded.
 */
async function groundTruthReady(stale) {
  state.busy = true;
  showSkeleton();
  const stopShowing = showCfdLoading(stale);
  try {
    await needCfd(cfdEntry());
  } catch (error) {
    if (!stale()) {
      state.busy = false;
      showRunFailure("cfd", () => run({ predict: false }));
    }
    return false;
  } finally {
    stopShowing();
  }
  return !stale();
}

/** An error or a cancelled job of the planner is thrown (a route the mission does not allow has points null). */
function checkPlans(plans) {
  for (const plan of Object.values(plans)) {
    if (plan.error || plan.cancelled) throw new DemoError(ErrorCode.PLANNER, `planner: ${plan.error ?? "cancelled"}`);
  }
}

/**
 * The computation stopped: failure card `kind` (results.js FAILURES: "wind", "planning", "cfd" - the stage it stopped
 * at - or "gpu"); retry: what its button does ("Try again" repeats a download or a prediction; null: the button
 * draws a new random mission, controls.js).
 */
function showRunFailure(kind, retry) {
  endSweep();                         // the half-built sweep goes too
  setRetry(retry);
  showFailure(kind);
}

/**
 * Before a computation while the predictor is being replaced (it shows as computing meanwhile) or after the device
 * was lost (a new one); false when superseded or failed.
 */
async function deviceReady(stale) {
  const switchingPrecision = precisionSwitch();
  if (switchingPrecision) {
    state.busy = true;
    showSkeleton();
    await switchingPrecision;
    if (stale()) return false;
  }
  return !state.deviceLost || renewDevice(stale);
}

/** A new graphics device before a computation, after the old one was lost; false when it failed or was superseded. */
async function renewDevice(stale) {
  showSkeleton();                     // the failure card goes first
  status("Preparing the graphics card…", "loading");
  try {
    await recreateGpu();
  } catch (error) {
    console.error(error);
    if (!stale()) showRunFailure("gpu", () => location.reload());
    return false;
  }
  if (stale()) return false;          // a newer run owns the status now
  status(null);
  return true;
}

/** A new prediction: the previous corridor's wind leaves the scene, and the sweep opens with its fixed range. */
function clearWindDisplay() {
  state.prediction = null;            // the previous corridor's wind is not used from now on
  invalidateParticles();              // particles still being set up for the previous wind are not drawn
  state.range = null;                 // the legend shows the sweep's range until the new wind's range exists
  state.sweepRange = [0, SWEEP_RANGE_RATIO * state.speed];
  applyDisplay();
  viewer.clearWind();
  viewer.clearRoutes();
  viewer.clearSweep();
  setSweepOpen(true);
}

/** The wind prediction of the mission's corridor with the sweep; bounded by PREDICT_LIMIT_MS. */
async function predictWind(mission, stale) {
  status(PREDICTING, "progress");
  progressStart();
  const used = predictor;
  const result = await withLimit(used.predict(toCells(mission.start), toCells(mission.goal), state.theta, {
    // a new wind direction first computes its along-wind geometry features (several seconds on the large city): the
    // status says so
    onStage: (stage) => { if (!stale()) status(stage === "geometry" ? COMPUTING_MDDF : PREDICTING, "progress"); },
    // the sweep shows every SWEEP_EVERY-th predicted layer; the sweep worker turns the sums into speeds
    onLayer: (layer, sums, layout, target) => { if (!stale()) drawSweep(layer, sums, layout, target, mission, stale); },
    sweepEvery: SWEEP_EVERY,
    // the predicted region is this corridor grown by the patch margin (gpu/tiling.js)
    overlap: PREDICT_OVERLAP, polygon: corridorCells(mission),
  }), PREDICT_LIMIT_MS, "wind prediction").catch((error) => {
    // a prediction that does not finish: the graphics card is made anew for the next computation - only for the
    // current computation on the current predictor (a superseded one's late timeout must not stop the current one)
    if (hasCode(error, ErrorCode.TIMEOUT) && !stale() && used === predictor) {
      used.cancel();
      state.deviceLost = true;
    }
    throw error;
  });
  if (!stale() && !result) throw new DemoError(ErrorCode.GPU, "the prediction stopped without a field");
  return result;
}

/** Whether the prediction has non-finite values (checked at every NAN_CHECK_STRIDE-th value). */
function hasNonFinite(values) {
  for (let i = 0; i < values.length; i += NAN_CHECK_STRIDE) if (!Number.isFinite(values[i])) return true;
  return false;
}

/** Every planned route's energy per km of the start-goal distance in the scoring wind (whPerKm). */
function scorePlans(plans, scoringWind, mission) {
  const distance = Math.hypot(mission.goal[0] - mission.start[0], mission.goal[1] - mission.start[1]);
  for (const plan of Object.values(plans)) {
    if (!plan.points) continue;
    const energy = trajectoryEnergy(plan.points, plan.speeds, scoringWind, plan.velocities);
    plan.whPerKm = energy / 3600 / (distance / 1000);
  }
}

/**
 * The shown time: wall clock from the start of this computation (the click, a new city's first mission, a wind
 * change) until the last route came back from the planner; everything in between counts, also waiting for the
 * planner workers (a ground-truth download comes before the start: run).
 */
function shownTiming(planning, started, plansAt, predictMs) {
  const ends = Object.values(planning.finishedAt);
  const shownEnd = ends.length ? Math.min(plansAt, Math.max(...ends)) : plansAt;
  return { wallMs: shownEnd - started, predictMs };
}

const GARBAGE_DELAY_MS = 1000;      // after a computation (a newer one starting meanwhile skips it)
let garbageTimer = null, garbagePending = false;
/**
 * After a computation, every thread's garbage is collected: worker threads free their temporary arrays (hundreds of
 * MB per mission on the large city) only when they happen to collect garbage, and idle workers rarely do.  The
 * browser's own memory measurement runs a collection in every thread of the page; its result is not used.  Once at a
 * time, GARBAGE_DELAY_MS after the computation, only where the browser offers it (cross-origin isolated Chrome / Edge).
 */
function collectWorkerGarbage() {
  if (!performance.measureUserAgentSpecificMemory || !sharedMemory()) return;
  clearTimeout(garbageTimer);
  garbageTimer = setTimeout(() => {
    if (garbagePending || state.busy) return;
    garbagePending = true;
    performance.measureUserAgentSpecificMemory().catch(() => {}).finally(() => { garbagePending = false; });
  }, GARBAGE_DELAY_MS);
}

/** A new mission; frame: the camera shows the whole city first (a mission alone never moves the view). */
export function setMission(mission, { predict = true, frame = false } = {}) {
  state.mission = mission;
  state.dragMission = null;           // a drag in progress does not bring back the mission it started on
  if (frame) state.frameNext = true;
  run({ predict });
}

/** A random mission of the paper's relative wind angle class `kind` ("tail", "head", "cross"; null: any). */
export function newRandomMission({ frame = false, kind = null } = {}) {
  if (!state.city) return;            // no city open (its failure card shows)
  setMission(randomMission(state.city, Math.random, state.altitude, kind, state.theta), { frame });
}
