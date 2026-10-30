// Plane geometry shared by the page and the workers: point in polygon, a polygon's cells row by row, a line in a
// box, polygon area, clipping, and the wind frame.

/** Whether (x, y) is inside the polygon [[x, y], ...] (even-odd rule, any orientation). */
export function insidePolygon(poly, x, y) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    // indexed reads, not destructuring: called for many points per frame (labels) and per particle step (worker)
    const xi = poly[i][0], yi = poly[i][1], xj = poly[j][0], yj = poly[j][1];
    if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

/**
 * The integer x of row y (an integer) inside the polygon, as runs: visit(x0, x1) for each run x0 <= x < x1, clipped to
 * lo <= x < hi.  The same points as insidePolygon(poly, x, y), a row at a time.
 */
export function polygonRuns(poly, y, lo, hi, visit) {
  const xs = [];
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i], [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y)) xs.push((xj - xi) * (y - yi) / (yj - yi) + xi);
  }
  xs.sort((p, q) => p - q);
  for (let k = 0; k + 1 < xs.length; k += 2) {
    const x0 = Math.max(lo, Math.ceil(xs[k])), x1 = Math.min(hi, Math.ceil(xs[k + 1]));
    if (x1 > x0) visit(x0, x1);
  }
}

/**
 * The parameter range [s0, s1] of the line o + s u inside the box lo <= p <= hi (2-D, per axis; the slab method);
 * s1 <= s0 when it misses.  An axis with |u| < eps counts as parallel: the line is inside its slab or never.
 */
export function lineInBox(o, u, lo, hi, eps) {
  let s0 = -Infinity, s1 = Infinity;
  for (let k = 0; k < 2; k++) {
    if (Math.abs(u[k]) < eps) {
      if (o[k] < lo[k] || o[k] > hi[k]) return [1, 0];
      continue;
    }
    const a = (lo[k] - o[k]) / u[k], b = (hi[k] - o[k]) / u[k];
    s0 = Math.max(s0, Math.min(a, b)); s1 = Math.min(s1, Math.max(a, b));
  }
  return [s0, s1];
}

/** Twice the signed area of a polygon (positive when counter-clockwise). */
export function twiceSignedArea(poly) {
  let sum = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i], b = poly[(i + 1) % poly.length];
    sum += a[0] * b[1] - b[0] * a[1];
  }
  return sum;
}

/** Sutherland-Hodgman clip of a polygon to the box x0 <= x <= x1, y0 <= y <= y1. */
export function clipToBox(poly, x0, x1, y0, y1) {
  for (const [axis, limit, keepBelow] of [[0, x0, false], [0, x1, true], [1, y0, false], [1, y1, true]]) {
    const inside = (p) => (keepBelow ? p[axis] <= limit : p[axis] >= limit);
    const out = [];
    for (let i = 0; i < poly.length; i++) {
      const a = poly[i], b = poly[(i + 1) % poly.length];
      if (inside(a)) out.push(a);
      if (inside(a) !== inside(b)) {
        const t = (limit - a[axis]) / (b[axis] - a[axis]);
        out.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      }
    }
    poly = out;
  }
  return poly;
}

/** Sutherland-Hodgman clip of a polygon by a convex polygon (any orientation). */
export function clipConvex(poly, clip) {
  const cx = clip.reduce((s, p) => s + p[0], 0) / clip.length;
  const cy = clip.reduce((s, p) => s + p[1], 0) / clip.length;
  let out = poly;
  for (let i = 0; i < clip.length && out.length; i++) {
    const a = clip[i], b = clip[(i + 1) % clip.length];
    const side = (p) => (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]);
    const inner = side([cx, cy]) > 0 ? 1 : -1;
    const input = out;
    out = [];
    for (let j = 0; j < input.length; j++) {
      const p = input[j], q = input[(j + 1) % input.length];
      const sp = inner * side(p), sq = inner * side(q);
      if (sp >= 0) out.push(p);
      if ((sp >= 0) !== (sq >= 0)) {
        const t = sp / (sp - sq);
        out.push([p[0] + t * (q[0] - p[0]), p[1] + t * (q[1] - p[1])]);
      }
    }
  }
  return out;
}

/**
 * The wind frame of direction theta (radians, the direction the wind blows toward) in a city of nx x ny cells: the
 * city rotated by theta about its centre, so the wind blows along +x.  Coordinates in cells:
 *   city  ix = cX + (sx - cX) cos - (sy - cY) sin,  iy = cY + (sx - cX) sin + (sy - cY) cos
 *   wind  sx = cX + (ix - cX) cos + (iy - cY) sin,  sy = cY - (ix - cX) sin + (iy - cY) cos
 * Hot loops write these expressions out with the frame's cX, cY, cos and sin (the same arithmetic).
 */
export function windFrame(theta, nx, ny) {
  const cX = (nx - 1) / 2, cY = (ny - 1) / 2, cos = Math.cos(theta), sin = Math.sin(theta);
  return {
    theta, cX, cY, cos, sin,
    toWind: ([ix, iy]) => {
      const dx = ix - cX, dy = iy - cY;
      return [cX + dx * cos + dy * sin, cY - dx * sin + dy * cos];
    },
    toCity: ([sx, sy]) => {
      const dx = sx - cX, dy = sy - cY;
      return [cX + dx * cos - dy * sin, cY + dx * sin + dy * cos];
    },
  };
}

/** The corners [x, y, z] of the vertical plane from origin + u s0 to origin + u s1 and from z0 to z1, in order. */
export function verticalPlaneCorners(origin, u, s0, s1, z0, z1) {
  const p = (s, z) => [origin[0] + u[0] * s, origin[1] + u[1] * s, z];
  return [p(s0, z0), p(s1, z0), p(s1, z1), p(s0, z1)];
}

/**
 * Points along segment a -> b at most 1 m apart, both ends included: visit(x, y, z) for each until it returns false.
 * Returns whether every point was visited.
 */
export function alongSegment(a, b, visit) {
  const L = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]), steps = Math.max(1, Math.ceil(L));
  for (let s = 0; s <= steps; s++) {
    const f = s / steps;
    if (visit(a[0] + f * (b[0] - a[0]), a[1] + f * (b[1] - a[1]), a[2] + f * (b[2] - a[2])) === false) return false;
  }
  return true;
}

/**
 * `values` (one per point) at distance `at` along a polyline whose cumulative lengths are `s`: linear between points,
 * clamped to the first and last segment.
 */
export function interpolateAlong(s, values, at) {
  let k = 0;
  while (k < s.length - 2 && s[k + 1] < at) k++;
  const f = Math.min(1, Math.max(0, (at - s[k]) / ((s[k + 1] - s[k]) || 1)));
  return values[k] + f * (values[k + 1] - values[k]);
}
