// Mission geometry: the corridor of a start-goal pair, the rules for usable start and goal points, and the paper's
// random missions.  City frame and units: shared/units.js.
import { CELL_M, cellCentreM, cellOf } from "../shared/units.js";
import { DemoError, ErrorCode } from "../shared/errors.js";
import { roundHalfEven } from "../shared/numeric.js";
import { lineInBox } from "../shared/geometry.js";
import { safeHeightAt } from "./clearance.js";

// The corridor: two trapezoids joined at the middle of start and goal, along their direction - 300 m wide at both
// ends, 50 m beyond start and goal, and in the middle 0.6 x the start-goal distance wide, at least 600 m and at most
// 800 m.  Below 500 m between start and goal the shape stays that of 500 m (600 m long, centred on the middle of the
// two points; only the points move towards the middle).  The same polygon is the drawn corridor, the predicted region
// (grown by the patch margin) and the planning space (lattice, optimizer constraints, the safe-height surfaces).
const CORRIDOR_END_WIDTH_M = 300;
const CORRIDOR_MID_MIN_M = 600;
const CORRIDOR_MID_MAX_M = 800;
const CORRIDOR_MID_FRACTION = 0.6;    // middle width / start-goal distance
const CORRIDOR_EXTEND_M = 50;
const CORRIDOR_FIXED_BELOW_M = 500;

/**
 * The corridor of start and goal (metres): its 6 corners (counter-clockwise), its edges as half-planes [a, b, c] with
 * a x + b y <= c inside ((a, b) the outward unit normal), the along-track range [sLo, sHi] measured from the start
 * along u, and its half width in the middle (halfMiddle); u: along the track, n: across it.
 */
export function corridorShape(start, goal) {
  const dx = goal[0] - start[0], dy = goal[1] - start[1], L = Math.hypot(dx, dy);
  const u = L > 1e-9 ? [dx / L, dy / L] : [1, 0], n = [-u[1], u[0]];
  const span = Math.max(L, CORRIDOR_FIXED_BELOW_M);
  const A = span / 2 + CORRIDOR_EXTEND_M;
  const halfEnd = CORRIDOR_END_WIDTH_M / 2;
  const halfMiddle = Math.min(Math.max(CORRIDOR_MID_FRACTION * span, CORRIDOR_MID_MIN_M), CORRIDOR_MID_MAX_M) / 2;
  const mid = [(start[0] + goal[0]) / 2, (start[1] + goal[1]) / 2];
  const at = (s, t) => [mid[0] + s * u[0] + t * n[0], mid[1] + s * u[1] + t * n[1]];
  const polygon = [at(-A, -halfEnd), at(0, -halfMiddle), at(A, -halfEnd), at(A, halfEnd), at(0, halfMiddle),
                   at(-A, halfEnd)];
  const planes = polygon.map((p, i) => {
    const q = polygon[(i + 1) % polygon.length], ex = q[0] - p[0], ey = q[1] - p[1], l = Math.hypot(ex, ey);
    const a = ey / l, b = -ex / l;              // outward for a counter-clockwise polygon
    return [a, b, a * p[0] + b * p[1]];
  });
  return { u, n, halfMiddle, polygon, planes, sLo: L / 2 - A, sHi: L / 2 + A };
}

/** Inside the corridor's half-planes (grown by d metres: every edge moved outward by d). */
export function insideCorridor(planes, x, y, d = 0) {
  for (const [a, b, c] of planes) if (a * x + b * y > c + d) return false;
  return true;
}

// Dragged start and goal have no distance limit; they are only kept MIN_SEPARATION_M apart so the mission is not
// degenerate.
const MIN_SEPARATION_M = 40;
const PARALLEL_EPS = 1e-12;           // a sampled chord this close to axis-parallel counts as parallel
const SNAP_RINGS = 200;               // snapToValid searches this many rings of cells around the point
const MISSION_DRAWS = 50;             // randomMission: pairs drawn at most until one is of the asked wind class
// The paper's mission sampling settings: start and goal at least 5 cells inside the city's edge, start-goal chord at
// least 0.9 x the side, at most 10000 draws.
const INSET_CELLS = 5;
const MIN_CHORD_RATIO = 0.9;
const MAX_TRIES = 10000;

/**
 * The paper's clearance test at a point: the distance from (x, y, z) to every building voxel centre exceeds
 * CLEARANCE_M (its sdf is the distance between voxel centres; mission sampling rejects sdf <= clearance).  The exact
 * minimum safe height (clearance.js safeHeightAt) is the lowest z with no voxel centre closer than CLEARANCE_M, so the
 * test is z > H.
 */
function clearOfBuildings(city, x, y, z) {
  return z > safeHeightAt(city, x, y);
}

/** The city's size [x, y] (m) and the inset (m) a start or goal keeps from its edges. */
const insetBox = (city) => ({ span: [city.nx * CELL_M, city.ny * CELL_M], edge: cellCentreM(INSET_CELLS) });

/** Whether a start or goal can be at (x, y, z): inside the city by the paper's inset, CLEARANCE_M from buildings. */
function usablePoint(city, x, y, z) {
  const { span, edge } = insetBox(city);
  if (x < edge || y < edge || x > span[0] - edge || y > span[1] - edge) return false;
  return clearOfBuildings(city, x, y, z);
}

/**
 * The usable point nearest to (x, y) at height z (a start or goal dragged to an unusable place snaps to the nearest
 * usable one instead of failing), at least MIN_SEPARATION_M from `other`.  Searched over cell centres, ring by ring;
 * null if none within SNAP_RINGS cells.
 */
export function snapToValid(city, x, y, z, other = null) {
  const { span, edge } = insetBox(city);
  const apart = (px, py) => !other || Math.hypot(px - other[0], py - other[1]) >= MIN_SEPARATION_M;
  const ok = (px, py) => usablePoint(city, px, py, z) && apart(px, py);
  x = Math.min(span[0] - edge, Math.max(edge, x)); y = Math.min(span[1] - edge, Math.max(edge, y));
  if (ok(x, y)) return [x, y];
  const ci = cellOf(x), cj = cellOf(y);
  for (let r = 1; r <= SNAP_RINGS; r++) {
    let best = null, bestD = Infinity;
    for (let j = cj - r; j <= cj + r; j++) for (let i = ci - r; i <= ci + r; i++) {
      if (Math.max(Math.abs(i - ci), Math.abs(j - cj)) !== r) continue;
      const px = cellCentreM(i), py = cellCentreM(j);
      if (!ok(px, py)) continue;
      const d = Math.hypot(px - x, py - y);
      if (d < bestD) { bestD = d; best = [px, py]; }
    }
    if (best) return best;
  }
  return null;
}

/**
 * Relative wind angle class of a mission, as the paper (Section 3.1): psi = angle between the horizontal start-to-goal
 * direction and the direction the wind blows toward (theta); tailwind |psi| <= 45 deg, headwind |psi - 180| <= 45
 * deg, crosswind otherwise.
 */
function windClass(start, goal, theta) {
  const psi = (Math.atan2(goal[1] - start[1], goal[0] - start[0]) - theta) * 180 / Math.PI;
  const a = Math.abs(((psi % 360) + 540) % 360 - 180);
  return a <= 45 ? "tail" : a >= 135 ? "head" : "cross";
}

/**
 * A random start and goal at `altitude`: the paper's mission sampling (inset from the border, with the settings
 * above).  `kind` ("tail" | "head" | "cross" | null) restricts the direction to that relative wind angle class: it
 * is drawn uniformly inside the class's angle range instead of the full circle - the paper's distribution restricted
 * to the class.  Snapping the end points to border cells can turn the direction by a few degrees; such a pair is
 * drawn again (rare).
 */
export function randomMission(city, random, altitude, kind, theta) {
  for (let k = 0; k < MISSION_DRAWS; k++) {
    let direction = null;
    if (kind) {
      const centre = kind === "tail" ? 0 : kind === "head" ? Math.PI : (random() < 0.5 ? 1 : -1) * Math.PI / 2;
      direction = () => theta + centre + (random() - 0.5) * Math.PI / 2;        // page direction start -> goal
    }
    const m = samplePair(city, random, altitude, direction);
    if (!kind || windClass(m.start, m.goal, theta) === kind) return m;
  }
  throw new DemoError(ErrorCode.MISSION, `no ${kind}wind mission found`);
}

/**
 * One pair as the paper draws it.  Paper axes: its "x" is array axis 0 = city rows (y here), its "y" axis 2 = city
 * columns (x here); rng.uniform(a, b) -> a + (b - a) random().  `direction`: () -> page direction start -> goal, or
 * null (uniform on the circle).
 */
function samplePair(city, random, altitude, direction) {
  const X = city.ny, Y = city.nx;
  const xmin = INSET_CELLS, xmax = X - 1 - INSET_CELLS;
  const ymin = INSET_CELLS, ymax = Y - 1 - INSET_CELLS;
  const cx = 0.5 * (xmin + xmax), cy = 0.5 * (ymin + ymax);
  const ax = 0.5 * (xmax - xmin), ay = 0.5 * (ymax - ymin);
  const a = Math.min(ax, ay), side = 2 * a;
  const minChord = MIN_CHORD_RATIO * side;
  const snap = (p) => {
    let x = Math.min(Math.max(roundHalfEven(p[0]), xmin), xmax);
    let y = Math.min(Math.max(roundHalfEven(p[1]), ymin), ymax);
    const dx = Math.min(Math.abs(x - xmin), Math.abs(x - xmax));
    const dy = Math.min(Math.abs(y - ymin), Math.abs(y - ymax));
    if (dx <= dy) x = Math.abs(x - xmin) <= Math.abs(x - xmax) ? xmin : xmax;
    else y = Math.abs(y - ymin) <= Math.abs(y - ymax) ? ymin : ymax;
    return [x, y];
  };
  for (let attempt = 0; attempt < MAX_TRIES; attempt++) {
    // a page direction phi from start to goal is the paper's theta = pi/2 - phi
    const paperTheta = direction ? Math.PI / 2 - direction() : 2 * Math.PI * random();
    const u = [Math.cos(paperTheta), Math.sin(paperTheta)], n = [-u[1], u[0]];
    const s = Math.abs(u[1]), c = Math.abs(u[0]);
    const dMax = a * (s + c) - minChord * (s * c);
    if (dMax <= 0) continue;
    const d = -dMax + 2 * dMax * random();
    const p0 = [d * n[0], d * n[1]];
    const [tMin, tMax] = lineInBox(p0, u, [-a, -a], [a, a], PARALLEL_EPS);     // the square |x|, |y| <= a
    if (tMax <= tMin) continue;
    const p1 = snap([p0[0] + tMin * u[0] + cx, p0[1] + tMin * u[1] + cy]);
    const p2 = snap([p0[0] + tMax * u[0] + cx, p0[1] + tMax * u[1] + cy]);
    // paper index (row, column) -> page metres (cell centre)
    const start = [cellCentreM(p1[1]), cellCentreM(p1[0]), altitude];
    const goal = [cellCentreM(p2[1]), cellCentreM(p2[0]), altitude];
    if (!clearOfBuildings(city, start[0], start[1], altitude)) continue;
    if (!clearOfBuildings(city, goal[0], goal[1], altitude)) continue;
    return { start, goal };
  }
  throw new DemoError(ErrorCode.MISSION, "no start and goal clear of buildings found");
}
