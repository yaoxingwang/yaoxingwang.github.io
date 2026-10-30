// The wind display: particles, horizontal and vertical slices, the display controls, the inflow arrows and the dial's
// view angle.
import { CELL_M, BAND_LOW_M, BAND_HIGH_M } from "../shared/units.js";
import { insidePolygon, twiceSignedArea } from "../shared/geometry.js";
import { horizontalSlice, verticalSlice } from "../particles/slices.js";
import { PARTICLE_M2_FULL, PARTICLE_M2_HORIZONTAL, PARTICLE_M2_VERTICAL, PARTICLE_MIN, PARTICLE_MAX,
         HEIGHT_MIN, HEIGHT_MAX, HEIGHT_STEP, DENSITY_MIN, DENSITY_MAX, DENSITY_STEP } from "./config.js";
import { state, displayKind, heightLabel, shownSource, windScale, particlesShown } from "./state.js";
import { $, showSliderValue } from "./dom.js";
import { viewer, shownCorridor, shownCorridorMetres } from "./scene.js";
import { fitPanel } from "./folds.js";
import { startParticles, setParticleRestart, computeSlice } from "./particle_feed.js";
import { drawLegend } from "./legend.js";

// The vertical plane's cross-wind position: whole cells, at least this far (m) inside the corridor's edges.
const PLANE_EDGE_M = 8;
const PLANE_SCAN_STEP_M = 2;     // the vertical plane's ends are found in steps of this along the wind

/**
 * The colour range of the wind display: 0 to the largest speed in the predicted corridor field (every predicted cell,
 * no percentile, no rounding).  The legend has ticks at whole steps below it; its top label is that maximum.
 */
function speedRange() {
  const scale = windScale();
  const top = Math.sqrt(state.prediction.maxSpeed2) * scale;
  return top > 0 ? [0, top] : [0, state.speed];
}

/** Which particle system the display needs: the 3-D one, or the one born on the current plane at its place. */
const particleKeyWanted = () => (displayKind() !== "particles2d" ? "3d"
  : `2d-${state.plane}-${state.plane === "h" ? state.sliceHeight : state.vOffset}`) + `-x${state.density}`;

// a particle worker that failed comes back with the particles at once
setParticleRestart(() => { if (state.prediction && particlesShown()) drawParticles(); });

/**
 * The particles for the current display: their count follows the area in metres (not on screen: zooming does not
 * change the density in space), between PARTICLE_MIN and PARTICLE_MAX, times the Density slider's value.
 */
export async function drawParticles(range = state.range) {
  const { mission } = state;
  if (!state.winds || !range || !state.prediction || !particlesShown()) return;
  const polygon = shownCorridorMetres(mission);
  const source = shownSource();
  const area = Math.abs(twiceSignedArea(polygon)) / 2;
  const planeArea = (pl) => (pl.s1 - pl.s0) * (BAND_HIGH_M - BAND_LOW_M);
  const onPlane = displayKind() === "particles2d";
  // particles on a plane: all born on the chosen plane (horizontal at the slider's height, or vertical along the
  // inflow at the slider's cross-wind offset) - the same plane as the slice
  const plane = !onPlane ? null : state.plane === "h" ? { kind: "h", z: state.sliceHeight } : verticalPlane();
  const want = !onPlane ? area / PARTICLE_M2_FULL
    : state.plane === "h" ? area / PARTICLE_M2_HORIZONTAL : planeArea(plane) / PARTICLE_M2_VERTICAL;
  const count = Math.round(state.density * Math.min(PARTICLE_MAX, Math.max(PARTICLE_MIN, want)));
  const seed = Math.round(mission.start[0] * 7 + mission.goal[1]);      // the same mission: the same particles
  const key = particleKeyWanted();    // the display these particles are made for (it may change while they are made)
  const system = await startParticles(source, polygon, count, seed, plane);
  if (!system || state.particleRequest !== system.id) return;     // a newer request meanwhile
  viewer.setParticles(system, range);
  state.particleKey = key;
  viewer.showParticles(particlesShown());
}

/** Draw the chosen wind display for the current prediction. */
export function drawWind() {
  if (!state.prediction) return;
  const range = speedRange();
  state.range = range;
  drawParticles(range);
  if (displayKind() === "slice") drawSlice();
  applyDisplay();
  showWindDirection();
}

/**
 * The vertical plane: along the inflow direction, through the mission midpoint moved by vOffset across the wind;
 * s0..s1 = where that line is inside the corridor; offsets limited to the corridor's cross-wind extent.
 */
export function verticalPlane() {
  const { mission } = state;
  const poly = shownCorridorMetres(mission);
  const u = [Math.cos(state.theta), Math.sin(state.theta)], n = [-u[1], u[0]];
  const c = [(mission.start[0] + mission.goal[0]) / 2, (mission.start[1] + mission.goal[1]) / 2];
  const across = poly.map(([x, y]) => (x - c[0]) * n[0] + (y - c[1]) * n[1]);
  const nmin = Math.ceil(Math.min(...across) / CELL_M) * CELL_M + PLANE_EDGE_M;
  const nmax = Math.floor(Math.max(...across) / CELL_M) * CELL_M - PLANE_EDGE_M;
  const off = Math.min(nmax, Math.max(nmin, state.vOffset));
  const origin = [c[0] + n[0] * off, c[1] + n[1] * off];
  const reach = poly.reduce((m, [x, y]) => Math.max(m, Math.hypot(x - origin[0], y - origin[1])), 0);
  let s0 = Infinity, s1 = -Infinity;
  for (let s = -reach; s <= reach; s += PLANE_SCAN_STEP_M) {
    if (insidePolygon(poly, origin[0] + u[0] * s, origin[1] + u[1] * s)) { s0 = Math.min(s0, s); s1 = Math.max(s1, s); }
  }
  return { kind: "v", origin, u, n, s0, s1, nmin, nmax, offset: off };
}

let sliceRequest = 0;             // the newest slice asked for (drawSlice)
let sliceShown = 0;               // the slice on screen (its request): one from the worker is shown only if newer

/**
 * The speed map at the slider's height (horizontal plane) or along the inflow (vertical plane) (particles/slices.js),
 * made here and shown at once; inWorker (the height slider moving, which must not hold up the page): made in the
 * particle worker (particle_feed.js computeSlice) and shown when it arrives if it is newer than the slice on screen
 * (also while later ones are on their way: a slider that keeps moving still sees the slice follow) and the display
 * still a slice (made here after all when the worker failed).
 */
export async function drawSlice({ inWorker = false } = {}) {
  const { city, prediction, mission } = state;
  if (!prediction || !state.range) return;
  const range = state.range, source = shownSource(), request = ++sliceRequest;
  let ask;
  if (state.plane === "v") {
    const pl = verticalPlane();
    if (!(pl.s1 > pl.s0)) { viewer.clearSlice(); sliceShown = request; return; }
    ask = { kind: "v", plane: { origin: pl.origin, u: pl.u, s0: pl.s0, s1: pl.s1 } };
  } else {
    const { gx, gy, layout } = prediction;
    const grid = { gx, gy, gridX0: layout.gridX0, gridY0: layout.gridY0, theta: layout.theta };
    ask = { kind: "h", windPoly: shownCorridor(mission).map(layout.toWind), grid, z: state.sliceHeight };
  }
  let slice = null;
  if (inWorker) {
    slice = await computeSlice(source, ask);
    if (slice?.replaced || request < sliceShown || state.prediction !== prediction || displayKind() !== "slice") {
      return;
    }
  }
  sliceShown = request;
  if (!slice) {
    const sampler = state.winds[source];
    slice = ask.kind === "h" ? { data: horizontalSlice(city, sampler, ask.windPoly, ask.grid, ask.z) }
      : verticalSlice(city, sampler, ask.plane);
  }
  if (ask.kind === "h") {
    viewer.setSlice(slice.data, prediction.layout, ask.z, range);
  } else {
    const plane = { ...ask.plane, z0: BAND_LOW_M, z1: BAND_HIGH_M };
    viewer.setVerticalSlice(slice.data, slice.ns, slice.nz, plane, range);
  }
}

/**
 * The display choice: view "particles" | "slice" | "none", and optionally the type ("3d" | "h" | "v").  The plane
 * (horizontal / vertical) is one choice for particles and slices: picking it in either view sets both; Full field
 * belongs to the particles only; each view keeps its own last type.
 */
export function setView(view, type = null) {
  if (type === "h" || type === "v") {
    state.slicePlane = type;
    if (view === "particles" || state.particlePlane !== "3d") state.particlePlane = type;
  } else if (type === "3d" && view === "particles") {
    state.particlePlane = "3d";
  }
  state.view = view;
  if (view === "particles" && state.particlePlane !== "3d") state.plane = state.particlePlane;
  else if (view === "slice") state.plane = state.slicePlane;
  // one height (horizontal plane) and one position (vertical plane) for the whole wind visualization: kept when
  // switching between particles and slices or between types, changed only by the slider
  if (displayKind() === "slice") drawSlice();
  // 3-D particles and particles born on a plane are different particle systems: rebuilt when the kind changes
  if (particlesShown() && state.particleKey !== particleKeyWanted()) drawParticles();
  applyDisplay();
}

/** The Density slider let go (or stepped with a key): the particles made anew at its value (none at 0). */
export function setDensity(density) {
  const stepped = +(Math.round(density / DENSITY_STEP) * DENSITY_STEP).toFixed(2);
  state.density = Math.min(DENSITY_MAX, Math.max(DENSITY_MIN, stepped));
  applyDisplay();
  if (particlesShown() && state.particleKey !== particleKeyWanted()) drawParticles();
}

/**
 * The type row and the slider: Height for the horizontal plane, cross-wind Position for the vertical one.  For the
 * vertical plane it also keeps state.vOffset inside the current corridor's cross-wind extent (a new mission can
 * narrow it), so the slider, the particle key and the plane agree.
 */
export function updatePlaneControls() {
  const type = state.view === "particles" ? state.particlePlane : state.slicePlane;
  $("plane-row").hidden = state.view === "none" || state.failureShown;
  for (const b of $("plane-kind").children) {
    b.hidden = b.dataset.plane === "3d" && state.view !== "particles";
    b.setAttribute("aria-pressed", String(b.dataset.plane === type));
  }
  const range = $("slice-range");
  if (state.plane === "h") {
    $("slider-name").textContent = "Height";
    Object.assign(range, { min: String(HEIGHT_MIN), max: String(HEIGHT_MAX), step: String(HEIGHT_STEP) });
    range.value = String(heightLabel(state.sliceHeight));
    showSliderValue(range, $("slice-value"), `${heightLabel(state.sliceHeight)} m`);
  } else if (state.mission) {
    const pl = verticalPlane();
    state.vOffset = pl.offset;                              // the offset limited to the corridor
    $("slider-name").textContent = "Position";
    Object.assign(range, { min: String(pl.nmin), max: String(pl.nmax), step: String(CELL_M) });
    range.value = String(pl.offset);
    showSliderValue(range, $("slice-value"), `${pl.offset > 0 ? "+" : ""}${Math.round(pl.offset)} m`);
  }
}

/**
 * The Wind field row: shown with Ground truth on and a wind display; while the ground-truth wind is not there its
 * Ground truth button cannot be pressed.
 */
export function showSourceRow() {
  $("wind-source-row").hidden = displayKind() === "none" || state.failureShown || state.mode !== "evaluation";
  $("wind-source").querySelector('[data-source="cfd"]').disabled = !state.winds?.cfd;
}

/** The display controls and layers as the state says. */
export function applyDisplay() {
  const display = displayKind();
  // Off (or a failure): the wind rows and the scale are not drawn at all, with no empty space left behind
  const windOff = display === "none" || state.failureShown;
  showSourceRow();
  // the buttons show the choice (also while its wind, the ground truth, is not there)
  for (const b of $("wind-source").children) {
    b.setAttribute("aria-pressed", String(b.dataset.source === state.sourceChoice));
  }
  for (const b of $("view-mode").children) b.setAttribute("aria-pressed", String(b.dataset.view === state.view));
  viewer.showParticles(particlesShown());
  viewer.showSlice(display === "slice");
  viewer.setCorridorLineVisible(display !== "slice");     // the corridor's outline does not show under a slice
  // the slider row only for slices and particles on a plane (not for the full field)
  $("height-row").hidden = !((display === "slice" || display === "particles2d") && !state.failureShown);
  $("density-row").hidden = state.view !== "particles" || state.failureShown;
  updatePlaneControls();
  $("legend").hidden = windOff;
  drawLegend(state.range);
  fitPanel();                // rows shown or hidden: fold what no longer fits
}

let directionDial = null;
/** The dial turned with the camera (app/dial.js), once it exists. */
export function setDirectionDial(dial) { directionDial = dial; }

/** The inflow arrows, and the dial turned with the camera (top-down from the camera azimuth). */
export function showWindDirection(theta = state.theta) {
  // the inflow arrows follow the wind direction; while the dial is dragged they preview its direction
  viewer.setInflowArrows(theta);
  if (!directionDial) return;
  const a = viewer.cameraState().azimuth * Math.PI / 180;
  // the screen angle of map 0 degrees seen top-down: right = -sin(azimuth), up = -cos(azimuth)
  const view = Math.atan2(-Math.cos(a), -Math.sin(a)) * 180 / Math.PI;
  directionDial.turnWithCamera(view);
}
