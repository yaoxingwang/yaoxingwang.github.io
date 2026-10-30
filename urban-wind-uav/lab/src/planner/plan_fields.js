// The trajectory optimizer's fields (trajopt.js).  The planner thread and its helper threads (helper_worker.js) build
// them with this one function, so they compute the same values.
//   wind(p, out3, jac9)   the wind spline on the planner's wind grid (the grid's horizontal axes u, n, then up), in
//                         city-frame components, and its Jacobian d w_i / d p_j (row-major) when jac9 is given
//   clear(p, grad3)       z - the conservative safe height (hs: the current ladder rung's surface) and its gradient
//   extra(y, m, k, eg)    the further margins of a node from m[k]: corridor edges, flight band, planning domain;
//                         their gradients (position only) into eg; returns their number
import { makeSpline3 } from "./spline3.js";
import { evalSafeHeight } from "./clearance.js";

/** wind(p, out3, jac9) above, of the planner's wind spline { coef, dims, spacing, origin, z0, u, n } (wind_grid.js). */
export function makeWindField({ coef, dims, spacing, origin, z0, u, n }) {
  const spline = makeSpline3(coef, dims, 3, spacing);   // 3 channels: the wind's x, y, z
  const local = new Float64Array(3), gradient = new Float64Array(9);
  return (p, out, J) => {
    const dx = p[0] - origin[0], dy = p[1] - origin[1];
    local[0] = dx * u[0] + dy * u[1]; local[1] = dx * n[0] + dy * n[1]; local[2] = p[2] - z0;
    spline(local, out, J ? gradient : null);
    if (J) for (let c = 0; c < 3; c++) {
      const gs = gradient[c * 3], gt = gradient[c * 3 + 1];
      J[c * 3] = gs * u[0] + gt * n[0]; J[c * 3 + 1] = gs * u[1] + gt * n[1]; J[c * 3 + 2] = gradient[c * 3 + 2];
    }
    return out;
  };
}

/** The optimizer's fields { wind, clear, extra } (above): the wind spline, the safe-height surface hs, the edges. */
export function makePlanFields(spline, hs, planes, band, domain) {
  const wind = makeWindField(spline);
  const h3 = new Float64Array(3);
  const clear = (p, grad) => {
    evalSafeHeight(hs, p[0], p[1], h3);
    grad[0] = -h3[1]; grad[1] = -h3[2]; grad[2] = 1;
    return p[2] - h3[0];
  };
  const extra = (y, m, k, eg) => {
    // the corridor: one margin per edge of the two trapezoids, c - (a x + b y) >= 0
    for (let e = 0; e < planes.length; e++) {
      const [a, b, c] = planes[e];
      m[k + e] = c - (a * y[0] + b * y[1]); eg[3 * e] = -a; eg[3 * e + 1] = -b; eg[3 * e + 2] = 0;
    }
    const q = planes.length, g = 3 * q;
    m[k + q] = y[2] - band[0]; eg[g] = 0; eg[g + 1] = 0; eg[g + 2] = 1;
    m[k + q + 1] = band[1] - y[2]; eg[g + 3] = 0; eg[g + 4] = 0; eg[g + 5] = -1;
    // the planning domain (the paper's position bounds)
    m[k + q + 2] = y[0] - domain[0]; eg[g + 6] = 1; eg[g + 7] = 0; eg[g + 8] = 0;
    m[k + q + 3] = domain[1] - y[0]; eg[g + 9] = -1; eg[g + 10] = 0; eg[g + 11] = 0;
    m[k + q + 4] = y[1] - domain[2]; eg[g + 12] = 0; eg[g + 13] = 1; eg[g + 14] = 0;
    m[k + q + 5] = domain[3] - y[1]; eg[g + 15] = 0; eg[g + 16] = -1; eg[g + 17] = 0;
    return q + 6;
  };
  return { wind, clear, extra };
}
