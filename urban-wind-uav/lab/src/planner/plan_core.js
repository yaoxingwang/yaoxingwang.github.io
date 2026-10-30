// Route planning for one wind: safe-height surfaces, the lattice route, trajectory optimization and the exact safety
// check.  The worker shell (plan_worker.js) handles the messages.
//   1. Lattice route (lattice.js): 8 m x 4.5 m corridor lattice, Dijkstra on steady-flight energy, clearance from
//      the conservative minimum-safe-height surface.
//   2. Trajectory (trajopt.js): positions, velocities, accelerations and one time step optimized by a barrier Newton
//      method on the exact energy model, clearance z >= H(x, y), airspeed, acceleration and thrust limits, corridor,
//      flight band and planning domain (the city).
//   3. Safety check every metre against the exact minimum safe height.
import { CELL_M, LAYER_M, cellCentreM } from "../shared/units.js";
import { DemoError, ErrorCode } from "../shared/errors.js";
import { alongSegment } from "../shared/geometry.js";
import { latticeClearance, latticeRoute, resample, nodePositionInto } from "./lattice.js";
import { optimizeTrajectory } from "./trajopt.js";
import { coefficientsFromSamples, coefficientsFromRegion } from "./spline3.js";
import { CLEARANCE_M, CAP_RADIUS_MAX_M, CAP_REACH_M, evalSafeHeight, exactSafeHeight } from "./clearance.js";
import { insideCorridor } from "./mission.js";
import { makePlanFields } from "./plan_fields.js";

export const LATTICE_DS = 8.0;         // lattice spacing (m), horizontal
export const LATTICE_DZ = 4.5;         // lattice spacing (m), vertical
const SEGMENT_M = 40.0;                // optimizer node spacing along the lattice route (m)
const BARRIER = { mus: [0.1, 0.01, 0.001], tracks: [0.1, 0.01, 0] };
// The ladder of conservative safe-height surfaces [cap radius (m), maximum filter radius (cells)]: provably
// conservative first (the clearance plus half a cell's diagonal), less conservative only if that fails; the exact
// check decides.
const LADDER = [[CAP_RADIUS_MAX_M, 2], [CAP_RADIUS_MAX_M, 1], [CLEARANCE_M, 0]];
// Only near the corridor is anything computed.  Every value at and next to the corridor is the same as from the whole
// crop: the B-spline taps of a point reach 2 cells (11.3 m), the maximum filter 2 cells, a cap 12.9 m - so surface
// values are kept within 16 m of the corridor and caps from buildings within 48 m of it.
const SURFACE_REACH_M = 16;
// the reaches above, checked: the taps (2 cells) within SURFACE_REACH_M; a cap's building within CAP_REACH_M after the
// surface reach, the widest maximum filter (LADDER) and the widest cap
const CELL_DIAGONAL_M = CELL_M * Math.SQRT2;
const reachesFit = SURFACE_REACH_M >= 2 * CELL_DIAGONAL_M
  && CAP_REACH_M >= SURFACE_REACH_M + LADDER[0][1] * CELL_DIAGONAL_M + CAP_RADIUS_MAX_M;
if (!reachesFit) throw new Error("the planner's reaches do not add up (SURFACE_REACH_M, CAP_REACH_M)");
const FLOOR_BELOW_BAND_M = 10;         // the surface where no building is near: this far below the flight band

// The energy-per-metre tables (lattice.js epmTables): the page has every worker build a share and sends them the
// joined set.
let tables = null;
/** The joined energy-per-metre tables of every move. */
export function setTables(t) { tables = t; }

// Why a mission has no route (a plan's `failed`): the mission's own failures, not errors of the planner.
const NO_LATTICE_ROUTE = "no lattice route";
const NO_SAFE_TRAJECTORY = "no safe trajectory found";

let prepared = null;
/** The mission prepared before its wind is known (prepare), or null. */
export function setPrepared(p) { prepared = p; }

let isCancelled = null;
/** (id) -> whether the page has cancelled job id (a newer computation started); set by the worker shell. */
export function setCancelCheck(check) { isCancelled = check; }
/** Stop plan `id` (DemoError CANCELLED) when a newer computation took this worker. */
const halt = (id) => { if (isCancelled && isCancelled(id)) throw new DemoError(ErrorCode.CANCELLED, "cancelled"); };

/**
 * The conservative safe-height surface of one ladder rung: at every crop cell centre near the corridor, caps of
 * radius capR (>= the exact H anywhere in the cell for capR = CLEARANCE_M + the half cell diagonal), then the maximum
 * over (2 maxR + 1)^2 cells (every B-spline tap around any point is then >= the exact H there).  The B-spline with
 * these samples as coefficients (a positive average of 16 taps) is >= the exact H everywhere.
 */
function safeSurface(safe, near, floor, capR, maxR) {
  const { w, h } = safe;
  const cap = new Float64Array(w * h).fill(floor);
  const reach = Math.ceil(capR / CELL_M);
  for (let cy = 0; cy < h; cy++) for (let cx = 0; cx < w; cx++) {
    const L = safe.layers[cy * w + cx];
    if (!L || !near.cap[cy * w + cx]) continue;
    const top = (L - 0.5) * LAYER_M;
    for (let dy = -reach; dy <= reach; dy++) for (let dx = -reach; dx <= reach; dx++) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= w || y >= h) continue;
      const d2 = CELL_M * CELL_M * (dx * dx + dy * dy);
      if (d2 >= capR * capR) continue;
      const v = top + Math.sqrt(capR * capR - d2);
      if (v > cap[y * w + x]) cap[y * w + x] = v;
    }
  }
  if (!maxR) return cap;
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let m = floor;
    if (!near.surface[y * w + x]) { out[y * w + x] = m; continue; }
    for (let dy = -maxR; dy <= maxR; dy++) for (let dx = -maxR; dx <= maxR; dx++) {
      const xx = Math.min(w - 1, Math.max(0, x + dx)), yy = Math.min(h - 1, Math.max(0, y + dy));
      if (cap[yy * w + xx] > m) m = cap[yy * w + xx];
    }
    out[y * w + x] = m;
  }
  return out;
}

/**
 * Everything that needs no wind: the lattice, the exact safe height, and the surfaces and lattice clearance of each
 * ladder rung (built when first needed).
 */
export function prepare(id, geom) {
  const { start, corridor, band, safe, domain } = geom;
  const { u, n, halfMiddle, planes, sLo, sHi } = corridor;
  // outside the planning domain (the city's cell-centre range) no height is safe
  const inside = (x, y) => x >= domain[0] && x <= domain[1] && y >= domain[2] && y <= domain[3];
  const { w: hw, h: hh } = safe;
  const near = { surface: new Uint8Array(hw * hh), cap: new Uint8Array(hw * hh) };
  for (let cy = 0; cy < hh; cy++) for (let cx = 0; cx < hw; cx++) {
    const x = cellCentreM(safe.x0 + cx), y = cellCentreM(safe.y0 + cy);
    near.surface[cy * hw + cx] = insideCorridor(planes, x, y, SURFACE_REACH_M) ? 1 : 0;
    near.cap[cy * hw + cx] = insideCorridor(planes, x, y, CAP_REACH_M) ? 1 : 0;
  }
  const floor = band[0] - FLOOR_BELOW_BAND_M;
  const hs = { coef: null, x0: safe.x0, y0: safe.y0, w: hw, h: hh };
  const h3 = new Float64Array(3);
  const surfaceAt = (x, y) => evalSafeHeight(hs, x, y, h3)[0];
  const exactH = (x, y) => (inside(x, y) ? exactSafeHeight(safe.layers, hw, hh, safe.x0, safe.y0, x, y) : Infinity);
  // the lattice box spans the corridor (along u from sLo to sHi, across +-halfMiddle; the start is a node); nodes
  // outside the corridor are masked: no clearance, no wind, never reached
  const iLo = Math.floor(sLo / LATTICE_DS), iHi = Math.ceil(sHi / LATTICE_DS), J0 = Math.floor(halfMiddle / LATTICE_DS);
  const nodeGrid = {
    origin: [start[0] + iLo * LATTICE_DS * u[0] - J0 * LATTICE_DS * n[0],
             start[1] + iLo * LATTICE_DS * u[1] - J0 * LATTICE_DS * n[1], band[0]],
    u, n, ds: LATTICE_DS, dz: LATTICE_DZ,
    I: iHi - iLo + 1, J: 2 * J0 + 1, K: Math.round((band[1] - band[0]) / LATTICE_DZ) + 1,
  };
  nodeGrid.mask = new Uint8Array(nodeGrid.I * nodeGrid.J);
  const p = [0, 0, 0];
  for (let j = 0; j < nodeGrid.J; j++) for (let i = 0; i < nodeGrid.I; i++) {
    nodePositionInto(nodeGrid, i, j, 0, p);
    nodeGrid.mask[j * nodeGrid.I + i] = insideCorridor(planes, p[0], p[1]) ? 1 : 0;
  }
  const rungs = [];
  /** Ladder rung r: its surface (made current in hs) and the lattice clearance on it. */
  const rung = (r) => {
    if (!rungs[r]) {
      hs.coef = safeSurface(safe, near, floor, ...LADDER[r]);
      const h = (x, y) => (inside(x, y) && insideCorridor(planes, x, y) ? surfaceAt(x, y) : Infinity);
      rungs[r] = { coef: hs.coef, clearance: latticeClearance(nodeGrid, h) };
    }
    hs.coef = rungs[r].coef;
    return rungs[r];
  };
  rung(0);
  return { id, nodeGrid, hs, exactH, rung };
}

/**
 * Exact check every metre along the straight segments between the trajectory points: clearance (minimum safe
 * height), corridor, flight band and the optimizer's limits.
 */
function isSafe(res, exactH, planes, band) {
  let minClear = Infinity, maxAcross = -Infinity, low = Infinity, high = -Infinity;
  const N = res.x.length / 3;
  for (let k = 0; k < N - 1; k++) {
    const a = res.x.subarray(3 * k, 3 * k + 3), b = res.x.subarray(3 * k + 3, 3 * k + 6);
    alongSegment(a, b, (x, y, z) => {
      const clear = z - exactH(x, y);
      if (clear < minClear) minClear = clear;
      // how far outside the corridor (largest edge excess; <= 0 inside)
      for (const [pa, pb, pc] of planes) maxAcross = Math.max(maxAcross, pa * x + pb * y - pc);
      low = Math.min(low, z); high = Math.max(high, z);
    });
  }
  return minClear >= -1e-6 && maxAcross <= 1e-6 && low >= band[0] - 1e-6 && high <= band[1] + 1e-6
    && res.limitViolation < 1e-3;
}

/**
 * Plan one route.  spec: the mission geometry (app/planning.js) and grid (wind_grid.js gridResult); pool: the started
 * helper threads (helper_pool.js) the whole plan uses, or null (this thread alone).  Returns
 * { points, velocities, speeds, planned (J, planning wind), dt, steps, retried, failed }; points is null when no
 * safe trajectory was found (failed: NO_LATTICE_ROUTE | NO_SAFE_TRAJECTORY).  planned, dt, steps, retried and failed
 * are diagnostics: the page uses only the route, the regression tests' fingerprint reads them.
 */
export function plan(id, spec, pool) {
  if (!(prepared && prepared.id === id)) prepared = prepare(id, spec);
  halt(id);
  const { start, goal, corridor: { planes }, band, grid, domain } = spec;
  const { nodeGrid, hs, exactH, rung } = prepared;
  // the wind spline on the planner's wind grid: of its region only for a grid with a region (the predicted and CFD
  // winds), built straight into the helper threads' shared buffer when there are helpers
  const alloc = pool ? (length) => pool.sharedCoef(length) : null;
  const { coef, dims } = grid.lo
    ? coefficientsFromRegion(grid.data, grid.dims, 3, grid.lo, grid.hi, alloc)   // 3: the wind's x, y, z
    : coefficientsFromSamples(grid.data, grid.dims, 3);
  const spline = { coef, dims, spacing: grid.spacing, origin: grid.origin, z0: grid.z0, u: grid.u, n: grid.n };
  const fields = makePlanFields(spline, hs, planes, band, domain);
  const wind = (p, out) => fields.wind(p, out, null);
  // every node's wind in advance, shared with the helper threads (the values the search would compute itself)
  const nodeWind = pool ? pool.latticeWinds(nodeGrid, spline, wind) : null;
  const route = (r) => {
    const linkClear = (p) => p[2] - exactH(p[0], p[1]);
    const lattice = latticeRoute(nodeGrid, tables, rung(r).clearance, wind, start, goal, linkClear, nodeWind);
    return lattice.points ? resample(lattice.points, SEGMENT_M) : null;
  };
  halt(id);
  let P = route(0);
  halt(id);
  let result = null, safe = false, retried = 0;
  for (let k = 0; k < LADDER.length && !safe; k++) {
    if (k > 0) {                  // rung 0's route and surface are current already (route(0) above)
      retried = k;
      P = route(k);
    }
    if (!P) continue;
    // the helper threads compute their share of the node terms with the same fields (plan_fields.js)
    const parallelNodes = Boolean(pool && P.length / 3 <= pool.maxN);
    if (parallelNodes) {
      const surface = { coef: hs.coef, x0: hs.x0, y0: hs.y0, w: hs.w, h: hs.h };
      pool.optStart({ spline, hs: surface, planes, band, domain, P });
    }
    const stop = isCancelled ? () => isCancelled(id) : null;
    result = optimizeTrajectory(P, fields, { ...BARRIER, pool: parallelNodes ? pool : null, halt: stop });
    safe = isSafe(result, exactH, planes, band);
  }
  if (!result) return { points: null, failed: NO_LATTICE_ROUTE };
  const N = result.x.length / 3;
  const points = [], velocities = [], speeds = [];
  for (let k = 0; k < N; k++) {
    points.push([result.x[3 * k], result.x[3 * k + 1], result.x[3 * k + 2]]);
    velocities.push([result.v[3 * k], result.v[3 * k + 1], result.v[3 * k + 2]]);
    speeds.push(Math.hypot(result.v[3 * k], result.v[3 * k + 1], result.v[3 * k + 2]));
  }
  return {
    points: safe ? points : null, velocities, speeds, planned: result.energy, dt: result.dt, steps: result.steps,
    retried, failed: safe ? undefined : NO_SAFE_TRAJECTORY,
  };
}
