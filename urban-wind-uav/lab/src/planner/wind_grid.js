// The planner's wind grid, sampled in the planner worker from a wind descriptor (wind/samplers.js).  Every wind has
// its own grid, so no wind is resampled: the predicted wind its wind-frame prediction grid, the CFD wind its city
// grid, the height profile one column, no wind one zero sample.  windGridRows fills the rows a in [aLo, aHi) only:
// helper threads share them, and every value is computed the same way whoever computes it.
import { CELL_M, LAYER_M, FIRST_FLIGHT_LAYER, LAST_FLIGHT_LAYER, WIND_GRID_STEP, WIND_BOX_ACROSS_M,
         CELL_VALUES, cellCentreM, cellIndexOf, layerCentreM } from "../shared/units.js";
import { windFrame } from "../shared/geometry.js";
import { windFromDescriptor } from "../wind/samplers.js";

// Prediction and CFD grids are taken at every WIND_GRID_STEP-th cell and every stored layer; only the region the
// planner reads is filled - the corridor's box (along the corridor BOX_ALONG_M beyond its ends, across
// WIND_BOX_ACROSS_M beyond its widest half width) grown by GROW samples (the spline's 4 taps and a margin from the
// region boundary).
const GROW = 4;
const BOX_ALONG_M = 24;
// The height profile's grid: one column (the planner grid's horizontal spacing), its own layers from the first to
// the last flight layer, PROFILE_STRIDE apart.
const PROFILE_LAYERS = [FIRST_FLIGHT_LAYER, LAST_FLIGHT_LAYER];
const PROFILE_STRIDE = 2;
const COLUMN_SPACING = [CELL_M * WIND_GRID_STEP, CELL_M * WIND_GRID_STEP, LAYER_M * PROFILE_STRIDE];

/**
 * The grid of wind descriptor `wind` for mission geometry `geom` (app/planning.js): { kind, dims [n0, n1, n2],
 * spacing [m], origin [x, y] of sample (0, 0), z0, gridU, gridN (the grid's horizontal axes), lo/hi (per a, the b
 * range of the region; absent: the whole box) and what windGridRows needs }.
 */
export function windGridShape(geom, wind) {
  if (wind.kind === "predicted") return predictedGridShape(geom, wind);
  if (wind.kind === "cfd") return cfdGridShape(geom, wind.grid);
  if (wind.kind === "profile") return profileGridShape(geom, wind);
  return noWindGridShape(geom);
}

/** Rows a in [aLo, aHi) of a grid of windGridShape into data (Float32Array, [a][b][k][3]). */
export function windGridRows(shape, data, aLo, aHi) {
  if (shape.kind === "predicted") {
    predictedGridRows(shape, data, aLo, aHi);
  } else if (shape.kind === "cfd") {
    cfdGridRows(shape, data, aLo, aHi);
  } else if (aLo <= 0 && aHi > 0) {        // one column: in row 0
    if (shape.kind === "profile") {
      for (let k = 0; k < shape.dims[2]; k++) data.set(shape.column(0, 0, shape.z0 + k * shape.spacing[2]), 3 * k);
    } else {
      data.fill(0, 0, 3);
    }
  }
}

/** The planner's wind grid of `wind` on one thread (no helpers). */
export function windGrid(geom, wind) {
  const shape = windGridShape(geom, wind);
  const [n0, n1, n2] = shape.dims;
  const data = new Float32Array(n0 * n1 * n2 * 3);
  windGridRows(shape, data, 0, n0);
  return gridResult(shape, data);
}

/** The grid as the planner takes it (plan_core.js): samples, shape and axes. */
export function gridResult(shape, data) {
  return { data, dims: shape.dims, spacing: shape.spacing, origin: shape.origin, z0: shape.z0,
           u: shape.gridU, n: shape.gridN, lo: shape.lo, hi: shape.hi };
}

/**
 * The predicted wind without resampling: its own wind-frame grid at every WIND_GRID_STEP-th cell and
 * every predicted layer (the layers between are interpolations and are left out).  The spline's axes are along and
 * across the wind; only the vector components are turned to the city frame (as predictedWind).  Cells outside the
 * predicted area take the nearest predicted cell of their row, or of the nearest predicted row.
 */
function predictedGridShape(geom, d) {
  const ls = d.layerStride;
  // (evenly spaced, the last layer one of them: gpu/predictor.js checks the stride)
  const f = frameGridShape(geom, d);
  return { ...f, kind: "predicted", dims: [f.n0, f.n1, (d.gz - 1) / ls + 1], ls, z0: layerCentreM(d.firstLayer),
           desc: d, spacing: [CELL_M * WIND_GRID_STEP, CELL_M * WIND_GRID_STEP, LAYER_M * ls] };
}

/**
 * The CFD wind on its own grid: the city grid (CFD is for wind toward 0 degrees, so the wind frame is the city frame)
 * at every WIND_GRID_STEP-th cell and every stored layer, as the predicted wind.  Stored order u, w, v.
 */
function cfdGridShape(geom, g) {
  const f = frameGridShape(geom, { theta: 0, nx: g.nx, ny: g.ny, gridX0: 0, gridY0: 0 });
  return { ...f, kind: "cfd", dims: [f.n0, f.n1, g.nz], z0: layerCentreM(g.firstLayer), desc: g,
           spacing: [CELL_M * WIND_GRID_STEP, CELL_M * WIND_GRID_STEP, LAYER_M * g.layerStride] };
}

/** The height profile does not change across the city: one column, its values taken at its own layers. */
function profileGridShape(geom, wind) {
  const levels = (PROFILE_LAYERS[1] - PROFILE_LAYERS[0]) / PROFILE_STRIDE + 1;
  return { kind: "profile", dims: [1, 1, levels], spacing: COLUMN_SPACING, origin: geom.start.slice(0, 2),
           z0: layerCentreM(PROFILE_LAYERS[0]), gridU: [1, 0], gridN: [0, 1], column: windFromDescriptor(wind) };
}

/** No wind (the wind-agnostic route): one zero sample; its spline is zero everywhere. */
function noWindGridShape(geom) {
  return { kind: "none", dims: [1, 1, 1], spacing: COLUMN_SPACING, origin: geom.start.slice(0, 2), z0: geom.band[0],
           gridU: [1, 0], gridN: [0, 1] };
}

/**
 * Grid geometry in a source's frame (rotated by theta about the city centre, its grid origin): the corridor box's
 * region.
 */
function frameGridShape(geom, d) {
  const { corridor: { u, n, halfMiddle, sLo, sHi }, start: [sx, sy] } = geom;
  const frame = windFrame(d.theta, d.nx, d.ny), { cos, sin } = frame;
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  const corners = [];
  const across = halfMiddle + WIND_BOX_ACROSS_M;
  const box = [[sLo - BOX_ALONG_M, -across], [sHi + BOX_ALONG_M, -across], [sHi + BOX_ALONG_M, across],
               [sLo - BOX_ALONG_M, across]];
  for (const [s, t] of box) {
    const ix = cellIndexOf(sx + s * u[0] + t * n[0]), iy = cellIndexOf(sy + s * u[1] + t * n[1]);
    const [fx, fy] = frame.toWind([ix, iy]), wx = fx - d.gridX0, wy = fy - d.gridY0;
    corners.push([wx, wy]);
    x0 = Math.min(x0, wx); x1 = Math.max(x1, wx); y0 = Math.min(y0, wy); y1 = Math.max(y1, wy);
  }
  // source grid cells c0 + a * WIND_GRID_STEP, r0 + b * WIND_GRID_STEP (GROW + 1 more on every side)
  const c0 = Math.floor(x0 / WIND_GRID_STEP) * WIND_GRID_STEP - (GROW + 1) * WIND_GRID_STEP;
  const c1 = Math.ceil(x1 / WIND_GRID_STEP) * WIND_GRID_STEP + (GROW + 1) * WIND_GRID_STEP;
  const r0 = Math.floor(y0 / WIND_GRID_STEP) * WIND_GRID_STEP - (GROW + 1) * WIND_GRID_STEP;
  const r1 = Math.ceil(y1 / WIND_GRID_STEP) * WIND_GRID_STEP + (GROW + 1) * WIND_GRID_STEP;
  const n0 = (c1 - c0) / WIND_GRID_STEP + 1, n1 = (r1 - r0) / WIND_GRID_STEP + 1;
  // the region: per a, the b range of the corridor box (a convex quadrilateral in (a, b)), grown by GROW samples
  const quad = corners.map(([wx, wy]) => [(wx - c0) / WIND_GRID_STEP, (wy - r0) / WIND_GRID_STEP]);
  const baseLo = new Float64Array(n0).fill(Infinity), baseHi = new Float64Array(n0).fill(-Infinity);
  for (let a = 0; a < n0; a++) for (let e = 0; e < 4; e++) {
    const [pa, pb] = quad[e], [qa, qb] = quad[(e + 1) % 4];
    if ((a < Math.min(pa, qa)) || (a > Math.max(pa, qa))) continue;
    if (pa === qa) {
      baseLo[a] = Math.min(baseLo[a], pb, qb); baseHi[a] = Math.max(baseHi[a], pb, qb);
    } else {
      const b = pb + (qb - pb) * (a - pa) / (qa - pa);
      baseLo[a] = Math.min(baseLo[a], b); baseHi[a] = Math.max(baseHi[a], b);
    }
  }
  const lo = new Int32Array(n0), hi = new Int32Array(n0);
  for (let a = 0; a < n0; a++) {
    let l = Infinity, h = -Infinity;
    for (let da = -GROW - 1; da <= GROW + 1; da++) {
      const j = a + da;
      if (j >= 0 && j < n0 && baseHi[j] >= baseLo[j]) { l = Math.min(l, baseLo[j]); h = Math.max(h, baseHi[j]); }
    }
    lo[a] = Number.isFinite(l) ? Math.max(0, Math.floor(l) - GROW) : 1;
    hi[a] = Number.isFinite(h) ? Math.min(n1 - 1, Math.ceil(h) + GROW) : 0;
  }
  // city position (metres) of source cell (c0, r0)
  const [ix, iy] = frame.toCity([d.gridX0 + c0, d.gridY0 + r0]);
  return {
    n0, n1, lo, hi, origin: [cellCentreM(ix), cellCentreM(iy)], gridU: [cos, sin], gridN: [-sin, cos], c0, r0,
  };
}

/** Rows a in [aLo, aHi) of the predicted wind's grid (a: along the wind). */
function predictedGridRows(shape, data, aLo, aHi) {
  const { dims: [, n1, n2], c0, r0, ls, desc: d, lo: bLo, hi: bHi } = shape;
  const { values, gx, gy, gz } = d, cos = Math.cos(d.theta), sin = Math.sin(d.theta), scale = d.scale;
  // predicted x range of every row (the predicted cells are the same on every layer; one layer is enough)
  const lo = new Int32Array(gy).fill(-1), hi = new Int32Array(gy).fill(-1);
  for (let y = 0; y < gy; y++) for (let x = 0; x < gx; x++) {
    if (values[((y * gz) * gx + x) * CELL_VALUES + 3] > 0) {
      if (lo[y] < 0) lo[y] = x;
      hi[y] = x;
    }
  }
  const nearestRow = (y) => {
    for (let r = 0; r < gy; r++) {
      if (y - r >= 0 && lo[y - r] >= 0) return y - r;
      if (y + r < gy && lo[y + r] >= 0) return y + r;
    }
    return -1;
  };
  for (let a = aLo; a < aHi; a++) {
    const x = Math.min(gx - 1, Math.max(0, c0 + a * WIND_GRID_STEP));
    for (let b = bLo[a]; b <= bHi[a]; b++) {
      let y = Math.min(gy - 1, Math.max(0, r0 + b * WIND_GRID_STEP));
      if (lo[y] < 0) y = nearestRow(y);
      const o = (a * n1 + b) * n2 * 3;
      if (y < 0) { data.fill(0, o, o + n2 * 3); continue; }
      const xx = Math.min(hi[y], Math.max(lo[y], x));
      for (let k = 0; k < n2; k++) {
        const i = ((y * gz + k * ls) * gx + xx) * CELL_VALUES;
        const wu = values[i], ww = values[i + 1], wv = values[i + 2];
        data[o + 3 * k] = scale * (wu * cos - wv * sin);
        data[o + 3 * k + 1] = scale * (wu * sin + wv * cos);
        data[o + 3 * k + 2] = scale * ww;
      }
    }
  }
}

/** Rows a in [aLo, aHi) of the CFD wind's grid: cells clamped to the city (edge values, as cfdWind). */
function cfdGridRows(shape, data, aLo, aHi) {
  const { dims: [, n1, n2], c0, r0, desc: g, lo: bLo, hi: bHi } = shape;
  const { values, nx, ny, nz } = g;
  for (let a = aLo; a < aHi; a++) {
    const x = Math.min(nx - 1, Math.max(0, c0 + a * WIND_GRID_STEP));
    for (let b = bLo[a]; b <= bHi[a]; b++) {
      const y = Math.min(ny - 1, Math.max(0, r0 + b * WIND_GRID_STEP)), o = (a * n1 + b) * n2 * 3;
      for (let k = 0; k < n2; k++) {
        const i = ((y * nz + k) * nx + x) * 3;
        data[o + 3 * k] = values[i]; data[o + 3 * k + 1] = values[i + 2]; data[o + 3 * k + 2] = values[i + 1];
      }
    }
  }
}
