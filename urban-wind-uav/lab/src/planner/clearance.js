// Clearance from buildings without a 3-D distance field.
//
// Buildings are vertical prisms: city cell (iy, ix) is solid up to `layers` 1.5 m layers.  Like the 3-D SDF of
// GeoWind2Plan (distance between voxel centres), the top solid voxel of a column has its centre at
// (layers - 0.5) x 1.5 m.  A point at height z is at least R from every building voxel exactly when, for every
// column c within horizontal distance d < R, z >= top_c + sqrt(R^2 - d^2).  The maximum of these lower bounds is the
// minimum safe height H(x, y): "SDF >= R" is the same as "z >= H(x, y)" (checked against the paper's SDF of a city
// block: identical at every grid point above 45 m).
import { CELL_M, LAYER_M, cellIndexOf } from "../shared/units.js";
import { bsplineWeights } from "../shared/numeric.js";

/** Local adaptation: 14 m to voxel centres leaves margin for a 10 m building-surface check. */
export const CLEARANCE_M = 14;
/** The largest cap radius of the safe-height surfaces (plan_core.js LADDER): the clearance and half a cell diagonal. */
export const CAP_RADIUS_MAX_M = CLEARANCE_M + CELL_M / 2 * Math.SQRT2;
/** Caps are made from buildings within this distance of the corridor (m; plan_core.js). */
export const CAP_REACH_M = 48;
/**
 * The safe-height crop of a mission (app/planning.js): the corridor's bounding box and this much around it (m), so it
 * holds every building within CAP_REACH_M of the corridor and the reach of its cap.
 */
export const SAFE_CROP_MARGIN_M = 68;
if (SAFE_CROP_MARGIN_M < CAP_REACH_M + CAP_RADIUS_MAX_M) throw new Error("the safe-height crop is too small for caps");
const NONE = -1e9;               // no building within reach: no lower bound
/**
 * Exact minimum safe height at city point (x, y) (metres, not interpolated) from a block of roof layers: `layers` is
 * a w x h grid whose cell (0, 0) is city cell (x0, y0) - the whole city or a crop of it: CLEARANCE_M above and
 * around every building voxel centre.  NONE when no building is within CLEARANCE_M.
 */
export function exactSafeHeight(layers, w, h, x0, y0, x, y) {
  const radius = CLEARANCE_M;
  const fx = cellIndexOf(x), fy = cellIndexOf(y);
  const r = Math.ceil(radius / CELL_M) + 1;
  const cx = Math.round(fx), cy = Math.round(fy);
  let best = NONE;
  for (let iy = Math.max(y0, cy - r); iy <= Math.min(y0 + h - 1, cy + r); iy++) {
    for (let ix = Math.max(x0, cx - r); ix <= Math.min(x0 + w - 1, cx + r); ix++) {
      const L = layers[(iy - y0) * w + ix - x0];
      if (!L) continue;
      const d2 = (CELL_M * (ix - fx)) ** 2 + (CELL_M * (iy - fy)) ** 2;
      if (d2 >= radius * radius) continue;
      const height = (L - 0.5) * LAYER_M + Math.sqrt(radius * radius - d2);
      if (height > best) best = height;
    }
  }
  return best;
}

/** Exact minimum safe height at any city point (exactSafeHeight on the whole city). */
export const safeHeightAt = (city, x, y) => exactSafeHeight(city.layers, city.nx, city.ny, 0, 0, x, y);

const WX = new Float64Array(4), DX = new Float64Array(4), WY = new Float64Array(4), DY = new Float64Array(4);

/**
 * The smooth safe-height surface and its gradient (per metre) at city point (x, y): the cubic B-spline with the
 * samples of `sp` (coef [h][w], its cell (0, 0) at city cell (x0, y0)) as coefficients, mirrored at the edges.
 * out = [H, dH/dx, dH/dy].
 */
export function evalSafeHeight(sp, x, y, out) {
  const fx = Math.min(Math.max(cellIndexOf(x) - sp.x0, 0), sp.w - 1);
  const fy = Math.min(Math.max(cellIndexOf(y) - sp.y0, 0), sp.h - 1);
  const bx = Math.floor(fx), by = Math.floor(fy);
  bsplineWeights(fx - bx, WX, DX);
  bsplineWeights(fy - by, WY, DY);
  let v = 0, gx = 0, gy = 0;
  const mirror = (m, n) => { while (m < 0 || m >= n) m = m < 0 ? -m : 2 * (n - 1) - m; return m; };
  for (let b = 0; b < 4; b++) {
    const iy = mirror(by - 1 + b, sp.h);
    for (let a = 0; a < 4; a++) {
      const c = sp.coef[iy * sp.w + mirror(bx - 1 + a, sp.w)];
      v += WX[a] * WY[b] * c; gx += DX[a] * WY[b] * c; gy += WX[a] * DY[b] * c;
    }
  }
  out[0] = v; out[1] = gx / CELL_M; out[2] = gy / CELL_M;
  return out;
}
