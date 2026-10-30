// The shared page state (what more than one module reads): every field is declared here, with what it holds.  Plain
// data, and the wind samplers of the last computation (functions of their fields, kept with them); no methods or
// callbacks.  Derived values are functions below.
import { BAND_LOW_M } from "../shared/units.js";
import { HEIGHT_MIN, REFERENCE_SPEED, METHODS } from "./config.js";

/** The one state object of the page. */
export const state = {
  // the city: the manifest of all cities, the open city's manifest entry and its height map (city/city.js)
  manifest: null, cityEntry: null, city: null,
  cityLoading: false,        // a city is being opened (its first mission starts when it is done)
  firstLoadDone: false,      // a city load got past its city file, then opened or failed (later loads say "Loading
                             // <city>…"; a city file that does not load keeps the first open's "Preparing the demo…")
  // the model files (kept for a new graphics device) and the device state
  metadata: null, weights: null,
  precision: null,           // "half" | "single": the wind model's float precision on this device
  gpuInfo: null,             // { vendor, architecture } of the graphics card in use (the introduction names it)
  deviceLost: false,         // the graphics device was lost or stopped answering: made anew before the next computation
  gpuWarm: false,            // the GPU pipelines have been compiled by a throw-away prediction
  // the route planner workers
  plannerFailed: false,
  // the wind: theta (radians, the direction it blows toward), speed (m/s); mode "evaluation" = Ground truth on
  // (CFD winds only: direction 0, the city's simulated speeds), "free" = off (any wind)
  theta: 0, speed: 4, mode: "evaluation",
  freeWind: null,            // { degrees, theta, speed } the user had with Ground truth off (restored by Off)
  dialTheta: null,           // the direction shown while the dial is dragged (radians)
  // the mission: { start, goal } in metres, z = altitude; dragMission: the mission a pin drag started on
  mission: null, dragMission: null, altitude: 75,
  frameNext: false,          // the next computation shows the whole city first
  // the computation
  busy: false,
  prediction: null,          // the last prediction (gpu/predictor.js result, plus city and predictedFor)
  winds: null,               // { ours, cfd, profile, agnostic } samplers of the last computation
  cfdWindFile: null,         // the CFD file winds.cfd comes from (the setting may have changed since)
  results: null,             // { plans, scoredIn, mission, wind (windKey), mode, timing } of the last computation
  failureShown: false,       // the failure card is shown
  // the wind display: view "particles" | "slice" | "none"; the type of each view (particles "3d" | "h" | "v",
  // slices "h" | "v"); plane: the plane of the current or last plane view ("h" | "v"); one height and one cross-wind
  // position for every plane; sourceChoice: the wind field the user chose ("ours" | "cfd", shownSource below)
  view: "particles", particlePlane: "h", slicePlane: "h", plane: "h", sliceHeight: 75, vOffset: 0,
  sourceChoice: "ours",
  range: null,               // the colour range of the wind display [0, max] (m/s)
  sweepRange: null,          // the colour range of the prediction sweep while it runs
  particleKey: null,         // which particle system is shown (wind_display.js)
  particleRequest: 0,        // the newest particle request (older answers are not drawn)
  // the results
  selected: "ours",          // the route shown on top (kept through computations; app/results.js showRoutes)
  flying: true,              // the drone flies along the selected route (the Fly button; kept through computations;
                             // on from the start: the opening shows a flight, the view following it, results.js)
  density: 0.75,             // the Density slider: the particle count times this (wind_display.js drawParticles)
  // the routes not drawn: their check boxes cleared (kept through computations and cities); at the opening all but
  // Ours
  hiddenRoutes: new Set(["cfd", "agnostic", "profile"]),
  tab: "energy",             // the chart tab
  evaluationTab: null,       // the tab to show again when Ground truth is back on
  profiles: null, profilesFor: null,   // the line charts' quantities and the results they belong to
};

/**
 * The display the wind view shows: "particles3d" (3-D particles), "particles2d" (particles born on a plane), "slice" or
 * "none".
 */
export function displayKind() {
  if (state.view === "particles") return state.particlePlane === "3d" ? "particles3d" : "particles2d";
  return state.view === "slice" ? "slice" : "none";
}

/** Whether the wind display shows particles (in the whole field or on a plane). */
export const particlesShown = () => (displayKind() === "particles3d" || displayKind() === "particles2d")
  && state.density > 0;

/** The wind field shown: the user's choice when that wind exists ("ours" | "cfd"). */
export const shownSource = () => (state.sourceChoice === "cfd" && state.winds?.cfd ? "cfd" : "ours");

/** The background wind speed as a factor of the model's reference speed (a predicted wind is scaled by it). */
export const windScale = () => state.speed / REFERENCE_SPEED;

const SPEED_TOLERANCE = 1e-6;   // m/s: speeds from the manifest and from the controls compared as numbers
const HEIGHT_TOLERANCE = 1e-6;  // m: a height this close to the band's bottom is the bottom
/** Whether two wind speeds are the same. */
export const sameSpeed = (a, b) => Math.abs(a - b) < SPEED_TOLERANCE;

/**
 * The CFD case of the current setting, or null.  With Ground truth on the wind is fixed at 0 degrees and the speed is
 * one of this city's simulated speeds.
 */
export function cfdEntry() {
  if (state.mode !== "evaluation") return null;
  return state.cityEntry?.cfd?.find((c) => sameSpeed(c.speed_m_s, state.speed)) || null;
}

/** Results must belong to the current mission, wind and evaluation mode. */
export const currentResults = () => Boolean(state.results && state.results.mission === state.mission &&
  state.results.wind === windKey() && state.results.mode === state.mode);

/** The background wind as a key (the wind results were computed in). */
export const windKey = () => `${state.theta},${state.speed}`;

/** The prediction as a key: the mission's start and goal and the wind direction (the speed only scales it). */
export const predictionKey = (mission) =>
  `${mission.start[0]},${mission.start[1]},${mission.goal[0]},${mission.goal[1]},${state.theta}`;

/** This city's simulated (CFD) wind speeds (m/s). */
export const cfdSpeeds = () => (state.cityEntry?.cfd || []).map((c) => c.speed_m_s);

/** The methods (config.js METHODS order) that have a route in `plans`. */
export const routedMethods = (plans) => METHODS.filter((m) => plans[m.id]?.points);

/**
 * A height (m) not below BAND_LOW_M, the centre of the lowest predicted layer and the lowest height the wind and the
 * planner work at: the sliders' HEIGHT_MIN (the paper's band starts there) is used as BAND_LOW_M and shown as it is.
 */
export const atLeastBand = (z) => Math.max(BAND_LOW_M, z);
/** An urban block's letter (the last word of its manifest label, "Urban block A") with its city: "A (Hong Kong)". */
export const shortName = (entry) => {
  const letter = entry.label.split(" ").pop();
  return entry.city ? `${letter} (${entry.city})` : letter;
};
/** The open urban block's short name, or "". */
export const cityShort = () => (state.cityEntry ? shortName(state.cityEntry) : "");
/** An urban block's name with its city, "Urban block A (Hong Kong)". */
export const blockName = (entry) => (entry.city ? `${entry.label} (${entry.city})` : entry.label);
/** The height shown for z (m): HEIGHT_MIN at the band's lowest working height, else whole metres. */
export const heightLabel = (z) => (z <= BAND_LOW_M + HEIGHT_TOLERANCE ? HEIGHT_MIN : Math.round(z));
