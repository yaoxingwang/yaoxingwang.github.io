// Opening an urban block: its height map, the scene, the predictor's whole-city geometry, its CFD winds, and its first
// mission.
import { ErrorCode, hasCode, withCode } from "../shared/errors.js";
import { loadCity, cityBuildings } from "../city/city.js";
import { FIRST_MISSIONS, CITY_GEOMETRY_LIMIT_MS, GPU_WARM_UP_LIMIT_MS, PREDICT_OVERLAP } from "./config.js";
import { state, cfdEntry, blockName, cityShort } from "./state.js";
import { $, nextFrame, withLimit } from "./dom.js";
import { viewer } from "./scene.js";
import { needCfd, queueCfd, forgetOtherCities } from "./cfd.js";
import { status, PREPARING, COMPUTING_MDDF } from "./status.js";
import { showSkeleton, showFailure, setRetry } from "./results.js";
import { forgetParticleWinds, invalidateParticles } from "./particle_feed.js";
import { setSweepOpen } from "./sweep.js";
import { showWindDirection } from "./wind_display.js";
import { cancelPlanning, plannersPending } from "./planning.js";
import { predictor, recreateGpu, precisionSwitch } from "./gpu.js";
import { supersedeRun, setMission, newRandomMission } from "./pipeline.js";
import { enterMode, drawSpeedChoice } from "./wind_controls.js";
import { markPicker, openPicker } from "./picker.js";

const MDDF_SHARE_AFTER_MS = 1000;   // the geometry's share done is shown once it takes longer than this
const MDDF_SHARE_TICK_MS = 250;
// The GPU warm-up's throw-away prediction: two points this many cells apart around the city centre, in a direction
// this far (radians) from the mission's, so nothing of the mission is computed in advance.
const WARM_UP_HALF_SPAN_CELLS = 140;
const WARM_UP_TURN = 1.0;

let cityToken = 0;

/** The number of city loads started so far (a new one for every city opened, also the same city again). */
export const cityLoads = () => cityToken;

/**
 * The failure card of an error that stopped a city's loading (results.js FAILURES) and what its button does (null:
 * open the city again; a device taken for lost or hung is made anew first: loadCityAndMission).  The planner
 * workers did not start ("startup": only a reload helps); the city is too large for the demo ("toolarge": another
 * city); the graphics card did not answer ("gpuhang") or failed ("gpulost"); any other error ("open").  The city's
 * files failing is handled in loadCityAndMission ("loading"); no first mission: firstMissionFailed.
 */
function cityFailure(error) {
  if (hasCode(error, ErrorCode.PLANNER)) return { kind: "startup", retry: () => location.reload() };
  if (hasCode(error, ErrorCode.TOO_LARGE)) return { kind: "toolarge", retry: chooseAnotherCity };
  if (hasCode(error, ErrorCode.TIMEOUT)) return { kind: "gpuhang", retry: null };
  if (hasCode(error, ErrorCode.GPU)) return { kind: "gpulost", retry: null };
  return { kind: "open", retry: null };
}

/**
 * The toolarge card's button (every press): the city picker, with focus on another city's tile - opened after the
 * press's click has reached the page's close-the-picker-on-a-click-outside listener (controls.js).
 */
function chooseAnotherCity() {
  setRetry(chooseAnotherCity);
  setTimeout(() => {
    openPicker(true);
    $("city-tiles").querySelector('button:not([aria-current="true"])')?.focus();
  }, 0);
}

/** The open city's first random mission could not be drawn: its card; Try again draws again (no city reload). */
function firstMissionFailed() {
  state.cityLoading = false;
  setRetry(() => {
    try {
      newRandomMission({ frame: true });
    } catch (error) {
      if (!hasCode(error, ErrorCode.MISSION)) throw error;
      console.error(error);
      firstMissionFailed();
    }
  });
  showFailure("mission");
}

/** Load a city and start its first mission; any error ends the loading state (no computation waits for ever). */
export async function useCity(entry) {
  const token = ++cityToken;               // a newer load takes the next token: this one stops at its next check
  try {
    return await loadCityAndMission(entry, () => token === cityToken);
  } catch (error) {
    console.error(error);
    if (token === cityToken) {
      status(null);
      state.firstLoadDone = true;      // later loads say "Loading <city>…"
      state.mission = null;            // the previous city's mission is not computed in this one
      if (hasCode(error, ErrorCode.MISSION)) {
        firstMissionFailed();
      } else {
        const { kind, retry } = cityFailure(error);
        if (kind === "gpulost") state.deviceLost = true;     // its Try again makes the device anew first
        cityNotOpened(entry, kind, retry);
      }
    }
  }
}

/**
 * The city of `entry` did not open: the loading ends with failure card `kind`; its button: retry, else a new load.
 * No city is open meanwhile (nothing computes for one the predictor does not hold).
 */
function cityNotOpened(entry, kind, retry) {
  state.cityLoading = false;
  state.city = null;
  setRetry(retry ?? (() => useCity(entry)));
  showFailure(kind);
}

/** A city is being opened: the computation running for the previous one stops. */
function stopComputing() {
  supersedeRun();
  invalidateParticles();               // particles still being made for the previous city are not drawn
  cancelPlanning();
  predictor?.cancel();
  setSweepOpen(false);
  state.sweepRange = null;
  state.busy = false;
  showSkeleton();
}

/**
 * The opened city (with its buildings' surfaces) replaces the previous one: in the state (nothing of the previous
 * city's mission, wind or results stays), in the panel and in the scene.  The previous city's large data is let go:
 * its CFD winds, their downloads still running, and the winds the particle worker holds.
 */
function showCity(entry, city, buildings) {
  forgetOtherCities(entry);
  forgetParticleWinds();
  state.cityEntry = entry;
  $("city-name").textContent = cityShort();  // named at once, also when a later step fails
  markPicker();
  state.city = city;
  state.mission = null;
  state.prediction = null;
  state.winds = null;                         // (the large city's CFD grid is about 200 MB)
  state.cfdWindFile = null;
  state.results = null;
  viewer.setCity(city, buildings);
  viewer.clearMission();
  // the inflow arrows of the wind direction first: the opening view frames the city with their upwind side
  // (viewer.openingView), so the view while loading is already the one the first mission shows
  showWindDirection();
  viewer.frameCity();
}

/** The steps of useCity; current(): this load is still the newest. */
async function loadCityAndMission(entry, current) {
  const fixed = FIRST_MISSIONS[entry.id] ?? null;
  state.cityLoading = true;
  stopComputing();
  const label = state.firstLoadDone ? `Loading ${blockName(entry)}…` : PREPARING;   // one message for all preparation
  status(label, "loading");
  let city, buildings;
  try {
    city = await loadCity(entry);
    if (!current()) return;                   // another city was chosen meanwhile: no buildings built for this one
    buildings = await cityBuildings(city);
  } catch (error) {
    // handled here, not by useCity: the page's first open stays the first (its retry says "Preparing the demo…")
    console.error(error);
    if (current()) cityNotOpened(entry, "loading", null);
    return;
  }
  if (!current()) return;                     // another city was chosen meanwhile
  showCity(entry, city, buildings);
  // the page gets a turn between the loading steps (the scene's new city and the predictor's would otherwise run as
  // one long task)
  await nextFrame();
  if (!current()) return;
  const switchingPrecision = precisionSwitch();     // a predictor being replaced: the city is opened on the new one
  if (switchingPrecision) {
    await switchingPrecision;
    if (!current()) return;
  }
  if (state.deviceLost) await recreateGpu(false);  // the retry after a lost or hung device; the city is set below
  if (!current()) return;
  const opened = predictor.setCity(city);
  if (state.mode === "evaluation") enterMode("evaluation");    // a simulated speed of this city
  drawSpeedChoice();
  const cfdWinds = downloadCfdWinds(entry);
  await computeGeometry(current, opened, label);
  if (!current()) return;                     // another city was chosen meanwhile
  // what follows (planner, GPU warm-up) has no share: the plain text again
  status(label, "loading");
  const pending = plannersPending();
  if (pending) {
    status(PREPARING, "loading");
    await pending;
    if (!current()) return;
  }
  if (!state.gpuWarm) {
    await warmUp(city);
    if (!current()) return;
  }
  await cfdWinds;
  if (!current()) return;
  state.firstLoadDone = true;
  state.cityLoading = false;
  startFirstMission(fixed);
}

/**
 * The CFD winds of the city: the current speed's downloaded as part of opening it, while the city's geometry is
 * computed (loading is expected when a city is opened), so its first mission computes as every later one; the city's
 * other speeds after it in the background (cfd.js).  Resolves when the current speed's download has ended; one that
 * failed is tried again by the computation that needs it (pipeline.js), which reports it.
 */
function downloadCfdWinds(entry) {
  const first = cfdEntry();
  const current = needCfd(first).catch(() => {});      // (reported by cfd.js)
  queueCfd((entry.cfd || []).filter((c) => c !== first));
  return current;
}

/** The city's first mission: its own fixed one (config.js FIRST_MISSIONS), else a random one. */
function startFirstMission(fixed) {
  if (fixed) {
    setMission({ start: [...fixed.start, state.altitude], goal: [...fixed.goal, state.altitude] }, { frame: true });
  } else {
    newRandomMission({ frame: true });
  }
}

/**
 * The city's reusable whole-city geometry features (MDDF), computed before the first mission so the mission's shown
 * time starts after them; bounded (a GPU that stops answering ends the loading with a message).  While they are
 * computed on a city switch the status says so, with the share done after a second (the page's first open says
 * "Preparing the demo…" only).
 */
async function computeGeometry(current, opened, label) {
  const t0 = performance.now();
  const mddfText = state.firstLoadDone ? COMPUTING_MDDF : null;
  status(mddfText || label, "loading");
  const timer = mddfText && setInterval(() => {
    const f = opened.progress;
    if (current() && performance.now() - t0 > MDDF_SHARE_AFTER_MS && f > 0 && f < 1) {
      status(`${mddfText} ${Math.round(100 * f)} %`, "loading");
    }
  }, MDDF_SHARE_TICK_MS);
  try {
    await withLimit(opened.ready, CITY_GEOMETRY_LIMIT_MS, "city geometry");
  } catch (error) {
    noteHang(error);
    // the browser's own errors of the geometry computation (a lost device) carry no code: they are GPU failures
    throw withCode(error, ErrorCode.GPU);
  } finally {
    clearInterval(timer);
  }
}

/**
 * The GPU warm-up, once per page, so the first mission takes the same path and time as later ones: the browser
 * compiles the network's GPU pipelines on their first use (about 0.2 s of the first prediction otherwise).  A
 * tiny throw-away prediction near the city centre compiles them during loading.  Its errors are only reported.
 */
async function warmUp(city) {
  const c = [Math.floor(city.nx / 2), Math.floor(city.ny / 2)];
  try {
    const warmed = await withLimit(predictor.predict([c[0] - WARM_UP_HALF_SPAN_CELLS, c[1]],
                                                     [c[0] + WARM_UP_HALF_SPAN_CELLS, c[1]],
                                                     state.theta + WARM_UP_TURN, { overlap: PREDICT_OVERLAP }),
                                   GPU_WARM_UP_LIMIT_MS, "GPU warm-up");
    if (warmed) state.gpuWarm = true;      // null: cancelled (another city), nothing compiled yet
  } catch (error) {
    console.error(error);
    noteHang(error);
  }
}

/** A GPU step that did not answer in time: the device is taken for hung (made anew before the next computation). */
function noteHang(error) {
  if (hasCode(error, ErrorCode.TIMEOUT)) state.deviceLost = true;
}
