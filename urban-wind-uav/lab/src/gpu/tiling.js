// Tiling of a mission corridor into the wind model's patches, in the wind frame (shared/geometry.js windFrame).
//
// The corridor polygon is grown by MARGIN cells (parallel offset of every edge) and clipped to the city square
// [-0.5, nx - 0.5] x [-0.5, ny - 0.5] (both in the wind frame).  Its bounding box is the predicted grid; it is covered
// by columns of patches at the requested stride, each column covering only the grown corridor's cross-wind extent
// within its cells.
import { CELL_M, WIND_GRID_STEP, WIND_BOX_ACROSS_M } from "../shared/units.js";
import { roundHalfEven } from "../shared/numeric.js";
import { clipConvex, twiceSignedArea, windFrame } from "../shared/geometry.js";

// The corridor is grown by this many cells before patches are chosen, so that the wind next to the corridor is
// predicted too: the planner's corridor box (planner/wind_grid.js) reaches WIND_BOX_ACROSS_M across beyond the
// corridor's widest half width, plus less than one sample spacing (WIND_GRID_STEP cells), and a trilinear sample
// (wind/samplers.js) reads the 4 horizontal neighbour cells of its point, each within sqrt(2) cells; the parallel
// offset by d contains exactly the points within d of every edge line.  The box's reach along the
// corridor (planner/wind_grid.js BOX_ALONG_M) is shorter, so covered too; the GROW samples beyond the box
// (planner/wind_grid.js) are only the spline's margin, read from the nearest predicted cells (wind/samplers.js).
const MARGIN = Math.ceil(WIND_BOX_ACROSS_M / CELL_M + WIND_GRID_STEP + Math.SQRT2);
// The default corridor when none is given (the GPU warm-up): start -> goal extended by this many cells, total width
// this share of the length.
const DEFAULT_EXTEND = 8;
const DEFAULT_WIDTH_FRACTION = 0.5;

/** Patch starts covering [start, start + size) at most `stride` apart, spread evenly (the model's training tiling). */
function axisStarts(start, size, patch, stride) {
  if (size <= patch) return [start];
  const count = Math.ceil((size - patch) / stride) + 1;
  const out = [];
  for (let k = 0; k < count; k++) out.push(start + k * patch - roundHalfEven(k * (count * patch - size) / (count - 1)));
  return out;
}

/** Cross-wind range [low, high] of a convex polygon ([x, y] corners) within the band x0 <= x <= x1, or null. */
function bandExtent(poly, x0, x1) {
  let low = Infinity, high = -Infinity;
  const consider = (y) => { low = Math.min(low, y); high = Math.max(high, y); };
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    if (a[0] >= x0 && a[0] <= x1) consider(a[1]);
    for (const edge of [x0, x1]) {
      if ((a[0] - edge) * (b[0] - edge) < 0) consider(a[1] + (edge - a[0]) / (b[0] - a[0]) * (b[1] - a[1]));
    }
  }
  return low <= high ? [low, high] : null;
}

/**
 * Parallel offset of a convex polygon by d: every edge moves outward by d along its normal and vertex i moves by
 * d (n1 + n2) / (1 + n1 . n2), where n1, n2 are the outward unit normals of its two edges (the intersection point of
 * the two offset edge lines).  The result contains the d-neighbourhood of the polygon.
 */
function growPolygon(poly, d) {
  const sign = twiceSignedArea(poly) > 0 ? 1 : -1;     // outward normal of edge p -> q is sign * (ty, -tx)
  const normals = poly.map((p, i) => {
    const q = poly[(i + 1) % poly.length];
    const l = Math.hypot(q[0] - p[0], q[1] - p[1]) || 1;
    return [sign * (q[1] - p[1]) / l, -sign * (q[0] - p[0]) / l];
  });
  return poly.map(([x, y], i) => {
    const n1 = normals[(i + poly.length - 1) % poly.length], n2 = normals[i];
    const k = d / (1 + n1[0] * n2[0] + n1[1] * n2[1]);
    return [x + k * (n1[0] + n2[0]), y + k * (n1[1] + n2[1])];
  });
}

/** The default corridor (city cells): the rectangle along start -> goal (see DEFAULT_EXTEND). */
export function defaultCorridor(start, goal) {
  const dx = goal[0] - start[0];
  const dy = goal[1] - start[1];
  const length = Math.hypot(dx, dy) || 1;
  const ux = dx / length, uy = dy / length;
  const nx = -uy, ny = ux;
  const half = 0.5 * DEFAULT_WIDTH_FRACTION * length;
  const a = [start[0] - ux * DEFAULT_EXTEND, start[1] - uy * DEFAULT_EXTEND];
  const b = [goal[0] + ux * DEFAULT_EXTEND, goal[1] + uy * DEFAULT_EXTEND];
  return [
    [a[0] + nx * half, a[1] + ny * half], [b[0] + nx * half, b[1] + ny * half],
    [b[0] - nx * half, b[1] - ny * half], [a[0] - nx * half, a[1] - ny * half],
  ];
}

/**
 * The tiling of corridor `polygon` (city cells) for wind direction theta in a city of nx x ny cells: the predicted
 * grid (gx x gy wind-frame cells from (gridX0, gridY0)), the patch origins [sy, sx], and the frame (toWind,
 * toCity).  Throws when the corridor does not touch the city.
 */
export function corridorTiling(polygon, theta, nx, ny, patch, stride) {
  const frame = windFrame(theta, nx, ny);
  const { toWind, toCity } = frame;
  const windPoly = polygon.map(toWind);
  const square = [[-0.5, -0.5], [nx - 0.5, -0.5], [nx - 0.5, ny - 0.5], [-0.5, ny - 0.5]].map(toWind);
  const grown = clipConvex(growPolygon(windPoly, MARGIN), square);
  if (grown.length < 3) throw new Error("the corridor does not touch the city");
  const gridX0 = Math.floor(Math.min(...grown.map((p) => p[0])));
  const gridY0 = Math.floor(Math.min(...grown.map((p) => p[1])));
  const gx = Math.ceil(Math.max(...grown.map((p) => p[0]))) - gridX0 + 1;
  const gy = Math.ceil(Math.max(...grown.map((p) => p[1]))) - gridY0 + 1;
  const patches = [];
  for (const x of axisStarts(gridX0, gx, patch, stride)) {
    // per column: the grown corridor's cross-wind extent within this column's cells x .. x + patch - 1, covered by
    // evenly spaced patches at no more than the stride
    const extent = bandExtent(grown, x, x + patch - 1);
    if (!extent) continue;
    const low = Math.floor(extent[0]);
    const span = Math.max(patch, Math.ceil(extent[1]) - low + 1);
    for (const y of axisStarts(low, span, patch, stride)) patches.push([y, x]);
  }
  return { theta, gx, gy, gridX0, gridY0, patches, toWind, toCity };
}
