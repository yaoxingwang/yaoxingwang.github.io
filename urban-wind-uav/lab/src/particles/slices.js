// The wind display's speed slices from a wind sampler (wind/samplers.js): the speed at every cell of a horizontal
// plane in the corridor, or on a vertical plane along the wind through the flight band, in open air only (in a
// building or outside the city: none).  Made on the page (app/wind_display.js drawSlice), and in the particle worker,
// which holds the winds, while the height slider moves (app/particle_feed.js computeSlice), so that never holds up
// the page.
import { CELL_M, LAYER_M, BAND_LOW_M, BAND_HIGH_M } from "../shared/units.js";
import { windFrame } from "../shared/geometry.js";
import { openAir, forCellsInside } from "../city/cells.js";

/** The length of the wind vector w (as Math.hypot, written out: Math.hypot takes several times as long). */
const speedOf = (w) => Math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2]);

/**
 * The horizontal slice at height z (m): [speed, mask] per cell of the wind-frame grid { gx, gy, gridX0, gridY0, theta }
 * of `city` (Float32Array; mask 1, or 0 with speed 0), over the cells inside windPoly (wind-frame cells).
 */
export function horizontalSlice(city, sampler, windPoly, grid, z) {
  const data = new Float32Array(grid.gx * grid.gy * 2);
  const w = [0, 0, 0];
  forCellsInside(windPoly, grid, windFrame(grid.theta, city.nx, city.ny), (o, x, y) => {
    if (!openAir(city, x, y, z)) return;
    sampler(x, y, z, w);
    data[o * 2] = speedOf(w);
    data[o * 2 + 1] = 1;
  });
  return data;
}

/**
 * The vertical slice on the plane { origin, u, s0, s1 } (metres along u from origin): [speed, mask] per point, a
 * CELL_M apart along it by LAYER_M up through the flight band.  Returns { data, ns, nz }.
 */
export function verticalSlice(city, sampler, { origin, u, s0, s1 }) {
  const ns = Math.max(2, Math.round((s1 - s0) / CELL_M) + 1);
  const nz = Math.round((BAND_HIGH_M - BAND_LOW_M) / LAYER_M) + 1;
  const data = new Float32Array(ns * nz * 2);
  const w = [0, 0, 0];
  for (let k = 0; k < nz; k++) {
    const z = BAND_LOW_M + LAYER_M * k;
    for (let j = 0; j < ns; j++) {
      const s = s0 + (s1 - s0) * j / (ns - 1);
      const x = origin[0] + u[0] * s, y = origin[1] + u[1] * s;
      if (!openAir(city, x, y, z)) continue;
      sampler(x, y, z, w);
      data[(k * ns + j) * 2] = speedOf(w);
      data[(k * ns + j) * 2 + 1] = 1;
    }
  }
  return { data, ns, nz };
}
