// The wind-aware global route on a coarse 3-D lattice (plan_core.js LATTICE_DS x LATTICE_DZ), refined afterwards by
// the trajectory optimizer.
//
// Lattice: nodes p(i, j, k) = origin + i ds u + j ds n + k dz z_hat (u, n horizontal unit vectors).  From each node
// 50 moves: 16 horizontal directions (unit and knight-like steps) combined with climb / level / descend, plus straight
// up and down.  A node is usable when it is at or above the minimum safe height there; an edge also needs its points
// at 1/4, 1/2 and 3/4 of its length to be (latticeClearance).  Edge cost: steady-flight energy at the ground speed
// minimizing energy per metre (energy.js power model), in the mean wind of the edge's two nodes.  Energy per metre
// comes from a table over the wind in the lattice frame (trilinear lookup), built once per move.
import { alongSegment } from "../shared/geometry.js";
import { power, VEHICLE, DRAG, EPS } from "./energy.js";
import { Heap, IndexedHeap } from "./heap.js";

/** Greatest common divisor (Euclid). */
const gcd = (a, b) => (b ? gcd(b, a % b) : a);

/** The 50 moves [di, dj, dk]. */
const MOVES = (() => {
  const horizontal = [];
  for (let a = -2; a <= 2; a++) for (let b = -2; b <= 2; b++) {
    if ((a || b) && gcd(Math.abs(a), Math.abs(b)) === 1) horizontal.push([a, b]);
  }
  const moves = [];
  for (const [a, b] of horizontal) for (const c of [-1, 0, 1]) moves.push([a, b, c]);
  moves.push([0, 0, 1], [0, 0, -1]);
  return moves;
})();

// ---------- energy-per-metre tables ----------
// Wind range of the tables (m/s: along the move, across it, vertical; first, last, step).  It covers the largest wind
// the planner sees at background speeds up to TABLE_SPEED_M_S (app/planning.js checks the page's maximum against
// it): predicted and profile winds scale with speed / 4 m/s, and the largest wind of the 4 m/s CFD fields of the
// page's cities (the flight band, fluid cells) is 7.715 m/s horizontal and 5.764 m/s vertical; x 10 / 4 = 19.29 and
// 14.41 m/s, rounded up to the table steps.  Values beyond the range are clamped to the edge.
export const TABLE_SPEED_M_S = 10;
const TABLE = { along: [-20, 20, 1], across: [-20, 20, 1], up: [-14.5, 14.5, 0.5] };
const TA0 = TABLE.along[0], TAS = TABLE.along[2], TC0 = TABLE.across[0], TCS = TABLE.across[2];
const TZ0 = TABLE.up[0], TZS = TABLE.up[2];

// the best ground speed: a scan of 1 .. SCAN_TOP_SPEED m/s in 1 m/s steps, then GOLDEN_STEPS golden-section steps
// within 1 m/s of the scan's best (not below MIN_GROUND_SPEED)
const SCAN_TOP_SPEED = 25, MIN_GROUND_SPEED = 0.5, GOLDEN_STEPS = 18;
const GOLDEN = 0.6180339887;          // (sqrt 5 - 1) / 2 to 10 digits

/** Energy per metre (J/m) flying direction (dx, dy, dz) in wind (wx, wy, wz) at the best feasible ground speed. */
function steadyEnergyPerMetre(dx, dy, dz, wx, wy, wz) {
  const cost = (v) => {
    const ax = v * dx - wx, ay = v * dy - wy, az = v * dz - wz;
    const a2 = ax * ax + ay * ay + az * az;
    if (a2 > VEHICLE.maxAirspeed * VEHICLE.maxAirspeed) return Infinity;
    const s = Math.sqrt(a2 + EPS);
    const tx = DRAG * s * ax, ty = DRAG * s * ay, tz = VEHICLE.mass * VEHICLE.g + DRAG * s * az;
    if (Math.sqrt(tx * tx + ty * ty + tz * tz) > VEHICLE.maxThrust) return Infinity;
    return power(ax, ay, az) / v;
  };
  // a coarse scan, then a golden-section search around its best speed (robust to the infeasible band at high airspeed)
  let best = Infinity, bestSpeed = 1;
  for (let v = 1; v <= SCAN_TOP_SPEED; v += 1) {
    const c = cost(v);
    if (c < best) { best = c; bestSpeed = v; }
  }
  if (!isFinite(best)) return Infinity;
  let lo = Math.max(MIN_GROUND_SPEED, bestSpeed - 1), hi = bestSpeed + 1;
  let x1 = hi - GOLDEN * (hi - lo), x2 = lo + GOLDEN * (hi - lo), f1 = cost(x1), f2 = cost(x2);
  for (let i = 0; i < GOLDEN_STEPS; i++) {
    if (f1 <= f2) {
      hi = x2; x2 = x1; f2 = f1; x1 = hi - GOLDEN * (hi - lo); f1 = cost(x1);
    } else {
      lo = x1; x1 = x2; f1 = f2; x2 = lo + GOLDEN * (hi - lo); f2 = cost(x2);
    }
  }
  return Math.min(best, f1, f2);
}

/**
 * Energy-per-metre tables, one per move, over the wind in the lattice frame: { tables: [{ t, len }], na, nc, nz }.
 * `only`: compute just these move indices (the others stay null) - the planner workers each build a share and the page
 * joins them, so the numbers are exactly those of a full build.
 */
export function epmTables(ds, dz, only) {
  const [a0, a1, as] = TABLE.along, [c0, c1, cs] = TABLE.across, [z0, z1, zs] = TABLE.up;
  const na = Math.round((a1 - a0) / as) + 1, nc = Math.round((c1 - c0) / cs) + 1, nz = Math.round((z1 - z0) / zs) + 1;
  const tables = [];
  for (let m = 0; m < MOVES.length; m++) {
    const [a, b, c] = MOVES[m];
    const len = Math.hypot(a * ds, b * ds, c * dz);
    if (!only.includes(m)) { tables.push(null); continue; }
    const ua = a * ds / len, uc = b * ds / len, uz = c * dz / len;
    const t = new Float32Array(na * nc * nz);
    for (let i = 0; i < na; i++) for (let j = 0; j < nc; j++) for (let k = 0; k < nz; k++) {
      t[(i * nc + j) * nz + k] = steadyEnergyPerMetre(ua, uc, uz, a0 + i * as, c0 + j * cs, z0 + k * zs);
    }
    tables.push({ t, len });
  }
  return { tables, na, nc, nz };
}

/**
 * Symmetry of the tables.  steadyEnergyPerMetre is unchanged when a horizontal direction component and the same wind
 * component change sign together, and when the two horizontal components are swapped together with the two
 * horizontal wind components (the power model has only squares, products of equal-signed pairs and commutative
 * sums), and the along and across wind axes are the same and symmetric about 0.  So every move's table is a
 * re-indexed copy of the table of a canonical move (|a| >= |b| >= 0): flip = mirror an index, swap = exchange the two
 * horizontal indices.  Derived tables hold the same numbers as computed ones.
 */
function moveSymmetry(m) {
  const [a, b, c] = MOVES[m];
  const A = Math.abs(a), B = Math.abs(b);
  const swap = B > A;
  const p = swap ? B : A, q = swap ? A : B;
  const canon = MOVES.findIndex(([x, y, z]) => x === p && y === q && z === c);
  return { canon, flipA: a < 0, flipB: b < 0, swap };
}

/** Indices of the moves whose tables are computed (the others are derived). */
export function canonicalMoves() {
  return [...new Set(MOVES.map((_, m) => moveSymmetry(m).canon))].sort((x, y) => x - y);
}

/** The table of a move from its canonical move's table (moveSymmetry). */
function deriveTable(canonTable, sym, na, nc, nz) {
  if (!sym.flipA && !sym.flipB && !sym.swap) return canonTable;
  const out = new Float32Array(na * nc * nz);
  for (let i = 0; i < na; i++) for (let j = 0; j < nc; j++) {
    // wind of this move (wa_i, wc_j); canonical wind: not swapped (fa wa, fb wc), swapped (fb wc, fa wa)
    const ia = sym.flipA ? na - 1 - i : i, jb = sym.flipB ? nc - 1 - j : j;
    const ci = sym.swap ? jb : ia, cj = sym.swap ? ia : jb;
    const o = (i * nc + j) * nz, co = (ci * nc + cj) * nz;
    for (let k = 0; k < nz; k++) out[o + k] = canonTable[co + k];
  }
  return out;
}

/**
 * Every move's table from the builders' parts (each { moves: [{ m, t, len }], na, nc, nz }; together they hold the
 * canonical moves): the other moves' tables are derived by symmetry.  Returns { tables, na, nc, nz }.
 */
export function joinTables(parts) {
  const all = { tables: new Array(MOVES.length), na: parts[0].na, nc: parts[0].nc, nz: parts[0].nz };
  for (const part of parts) for (const { m, t, len } of part.moves) all.tables[m] = { t, len };
  for (let m = 0; m < MOVES.length; m++) {
    if (all.tables[m]) continue;
    const sym = moveSymmetry(m), c = all.tables[sym.canon];
    all.tables[m] = { t: deriveTable(c.t, sym, all.na, all.nc, all.nz), len: c.len };
  }
  return all;
}

// table coordinates are clamped this far below the last index, so a cell and its upper neighbour exist
const TABLE_EDGE = 1.000001;

/** Trilinear table value at table coordinates (fa, fc, fz), clamped to the table (the search's edge cost). */
function tableValue(tabs, t, fa, fc, fz) {
  const na = tabs.na, nc = tabs.nc, nz = tabs.nz;
  fa = fa < 0 ? 0 : fa > na - TABLE_EDGE ? na - TABLE_EDGE : fa;
  fc = fc < 0 ? 0 : fc > nc - TABLE_EDGE ? nc - TABLE_EDGE : fc;
  fz = fz < 0 ? 0 : fz > nz - TABLE_EDGE ? nz - TABLE_EDGE : fz;
  const i = fa | 0, j = fc | 0, k = fz | 0;
  const ta = fa - i, tc = fc - j, tz = fz - k;
  const o = (i * nc + j) * nz + k, oj = nz, oi = nc * nz;
  const v00 = t[o] + tz * (t[o + 1] - t[o]);
  const v01 = t[o + oj] + tz * (t[o + oj + 1] - t[o + oj]);
  const v10 = t[o + oi] + tz * (t[o + oi + 1] - t[o + oi]);
  const v11 = t[o + oi + oj] + tz * (t[o + oi + oj + 1] - t[o + oi + oj]);
  const v0 = v00 + tc * (v01 - v00), v1 = v10 + tc * (v11 - v10);
  return v0 + ta * (v1 - v0);
}

// ---------- clearance ----------
/**
 * Building clearance of the whole lattice when clearance is z >= h(x, y) (minimum safe height): it depends on the
 * horizontal position only, so it needs no wind and is built before the wind is known.
 *   nodeH[j I + i]: node (i, j, k) is free when its height z_k >= nodeH;
 *   edgeZ[(j I + i) M + m]: move m from (i, j, k) has its 1/4, 1/2, 3/4 points clear when z_k >= edgeZ (= max over f
 *   of h(point f) - f c dz); Infinity when the move leaves the lattice.
 */
export function latticeClearance(nodeGrid, h) {
  const { dz, I, J } = nodeGrid, M = MOVES.length;
  const nodeH = new Float64Array(I * J), edgeZ = new Float64Array(I * J * M);
  const at = [0, 0, 0];         // fractional (fi, fj): the quarter points of a move
  const hAt = (fi, fj) => (nodePositionInto(nodeGrid, fi, fj, 0, at), h(at[0], at[1]));
  for (let j = 0; j < J; j++) for (let i = 0; i < I; i++) nodeH[j * I + i] = hAt(i, j);
  const mid = new Float64Array(3);
  for (let j = 0; j < J; j++) for (let i = 0; i < I; i++) {
    let lastA = null, lastB = null;
    for (let m = 0; m < M; m++) {
      const [a, b, c] = MOVES[m], o = (j * I + i) * M + m;
      const i2 = i + a, j2 = j + b;
      if (i2 < 0 || j2 < 0 || i2 >= I || j2 >= J) { edgeZ[o] = Infinity; continue; }
      if (!a && !b) { edgeZ[o] = -Infinity; continue; }   // vertical: both end nodes free is enough
      if (a !== lastA || b !== lastB) {                   // the three moves of one direction share h
        for (let s = 0; s < 3; s++) mid[s] = hAt(i + 0.25 * (s + 1) * a, j + 0.25 * (s + 1) * b);
        lastA = a; lastB = b;
      }
      let req = -Infinity;
      for (let s = 0; s < 3; s++) { const v = mid[s] - 0.25 * (s + 1) * c * dz; if (v > req) req = v; }
      edgeZ[o] = req;
    }
  }
  return { nodeH, edgeZ };
}

// ---------- route ----------
/**
 * The lowest-energy lattice route.  nodeGrid: { origin [x, y, z], u, n, ds, dz, I, J, K, mask }; clearance:
 * latticeClearance(nodeGrid, h); wind(p, out3) -> wind at p; start, goal: [x, y, z]; linkClear(p) -> exact clearance
 * margin (the start and goal links are checked with it every metre); nodeWind: every node's wind computed in advance
 * (helper threads, latticeNodeWinds), or null (computed when first needed).
 * Returns { points (start, lattice nodes, goal) }, points null when there is no route.
 *
 * The two searches give the same route up to rounding: with the node winds in advance the edge's table coordinates are
 * the mean of its nodes' float32 coordinates and tiles of nodes are pruned by lower bounds (searchWithNodeWinds, the
 * faster one), otherwise they are the coordinates of the nodes' mean wind (searchLazy, winds computed where it looks).
 * Where several routes cost the same up to rounding (a wind uniform in the horizontal, the height profile), the two
 * can return different ones of them.
 */
export function latticeRoute(nodeGrid, tabs, clearance, wind, start, goal, linkClear, nodeWind) {
  const s = connect(nodeGrid, clearance, start, linkClear), g = connect(nodeGrid, clearance, goal, linkClear);
  if (s < 0 || g < 0) return { points: null };
  const { dist, pred } = nodeWind
    ? searchWithNodeWinds(nodeGrid, tabs, clearance, nodeWind, s, g)
    : searchLazy(nodeGrid, tabs, clearance, wind, s, g);
  if (!isFinite(dist[g])) return { points: null };
  const { I, J } = nodeGrid, IJ = I * J;
  const chain = [];
  for (let id = g; id >= 0; id = pred[id]) {
    chain.push(id);
    if (id === s) break;
  }
  chain.reverse();
  const points = [start.slice()];
  for (const id of chain) {
    const k = Math.floor(id / IJ), rem = id - k * IJ, j = Math.floor(rem / I), i = rem - j * I;
    points.push(nodePosition(nodeGrid, i, j, k));
  }
  points.push(goal.slice());
  return { points };
}

/** The position [x, y, z] of lattice node (i, j, k), written into `out`. */
export function nodePositionInto({ origin, u, n, ds, dz }, i, j, k, out) {
  out[0] = origin[0] + i * ds * u[0] + j * ds * n[0];
  out[1] = origin[1] + i * ds * u[1] + j * ds * n[1];
  out[2] = origin[2] + k * dz;
  return out;
}

/** The position [x, y, z] of lattice node (i, j, k). */
const nodePosition = (nodeGrid, i, j, k) => nodePositionInto(nodeGrid, i, j, k, [0, 0, 0]);

const CONNECT_NODES = 6;        // a start or goal is linked to free nodes within this many nodes along each axis

/** The node a start or goal connects to: the nearest free node whose straight link is clear; -1 if none. */
function connect(nodeGrid, clearance, point, linkClear) {
  const { origin, u, n, ds, dz, I, J, K } = nodeGrid;
  const linkOk = (a, b) => alongSegment(a, b, (x, y, z) => !(linkClear([x, y, z]) < 0));
  const dx = point[0] - origin[0], dy = point[1] - origin[1];
  const ci = Math.round((dx * u[0] + dy * u[1]) / ds), cj = Math.round((dx * n[0] + dy * n[1]) / ds);
  const ck = Math.round((point[2] - origin[2]) / dz);
  const candidates = [];
  const R = CONNECT_NODES;
  for (let k = Math.max(0, ck - R); k <= Math.min(K - 1, ck + R); k++)
    for (let j = Math.max(0, cj - R); j <= Math.min(J - 1, cj + R); j++)
      for (let i = Math.max(0, ci - R); i <= Math.min(I - 1, ci + R); i++) {
        const id = (k * J + j) * I + i;
        if (!(origin[2] + k * dz >= clearance.nodeH[j * I + i])) continue;
        const q = nodePosition(nodeGrid, i, j, k);
        candidates.push([Math.hypot(q[0] - point[0], q[1] - point[1], q[2] - point[2]), id, q]);
      }
  candidates.sort((x, y) => x[0] - y[0]);
  for (const [, id, q] of candidates) if (linkOk(point, q)) return id;
  return -1;
}

/** Dijkstra from s to g with every node's wind computed when it is first reached. */
function searchLazy(nodeGrid, tabs, clearance, wind, s, g) {
  const { origin, u, n, dz, I, J, K } = nodeGrid;
  const N = I * J * K, M = MOVES.length, IJ = I * J;
  const nodeWind = new Float32Array(N * 3).fill(NaN);
  const position = [0, 0, 0], wv = new Float64Array(3);
  const windAt = (id, i, j, k, out) => {
    if (Number.isNaN(nodeWind[id * 3])) {
      wind(nodePositionInto(nodeGrid, i, j, k, position), wv);
      nodeWind[id * 3] = wv[0]; nodeWind[id * 3 + 1] = wv[1]; nodeWind[id * 3 + 2] = wv[2];
    }
    out[0] = nodeWind[id * 3]; out[1] = nodeWind[id * 3 + 1]; out[2] = nodeWind[id * 3 + 2];
    return out;
  };
  const dist = new Float64Array(N).fill(Infinity), pred = new Int32Array(N).fill(-1), visited = new Uint8Array(N);
  const heap = new Heap();
  dist[s] = 0; heap.push(0, s);
  const wa = new Float64Array(3), wb = new Float64Array(3);
  while (heap.size) {
    const id = heap.pop();
    if (visited[id]) continue;
    visited[id] = 1;
    if (id === g) break;
    const k = Math.floor(id / IJ), rem = id - k * IJ, j = Math.floor(rem / I), i = rem - j * I;
    const pz = origin[2] + k * dz;
    windAt(id, i, j, k, wa);
    for (let m = 0; m < M; m++) {
      const [a, b, c] = MOVES[m];
      const i2 = i + a, j2 = j + b, k2 = k + c;
      if (i2 < 0 || j2 < 0 || k2 < 0 || i2 >= I || j2 >= J || k2 >= K) continue;
      const id2 = (k2 * J + j2) * I + i2;
      if (visited[id2]) continue;
      if (!(origin[2] + k2 * dz >= clearance.nodeH[id2 - k2 * IJ])) continue;
      if (pz < clearance.edgeZ[(j * I + i) * M + m]) continue;
      windAt(id2, i2, j2, k2, wb);
      const ex = 0.5 * (wa[0] + wb[0]), ey = 0.5 * (wa[1] + wb[1]), ez = 0.5 * (wa[2] + wb[2]);
      const along = ex * u[0] + ey * u[1], across = ex * n[0] + ey * n[1];
      const table = tabs.tables[m];
      const epm = tableValue(tabs, table.t, (along - TA0) / TAS, (across - TC0) / TCS, (ez - TZ0) / TZS);
      if (!isFinite(epm)) continue;
      const nd = dist[id] + epm * table.len;
      if (nd < dist[id2]) { dist[id2] = nd; pred[id2] = id; heap.push(nd, id2); }
    }
  }
  return { dist, pred };
}

/**
 * Dijkstra from s to g with every node's wind given (helper threads): each node's table coordinates are computed
 * once, an edge's are the mean of its nodes', and a relaxation that cannot improve its target even at the smallest
 * cost possible near its tile is skipped (exact: it would not change anything).
 */
function searchWithNodeWinds(nodeGrid, tabs, clearance, nodeWind, s, g) {
  const { origin, u, n, dz, I, J, K } = nodeGrid;
  const N = I * J * K, M = MOVES.length, IJ = I * J;
  const F = new Float32Array(4 * N);      // per node: along, across, up table coordinates (one cache line per node)
  for (let id = 0; id < N; id++) {
    const x = nodeWind[3 * id], y = nodeWind[3 * id + 1];
    F[4 * id] = ((x * u[0] + y * u[1]) - TA0) / TAS;
    F[4 * id + 1] = ((x * n[0] + y * n[1]) - TC0) / TCS;
    F[4 * id + 2] = (nodeWind[3 * id + 2] - TZ0) / TZS;
  }
  const MA = new Int32Array(M), MB = new Int32Array(M), MC = new Int32Array(M);
  const OFF = new Int32Array(M), LEN = new Float64Array(M);
  const TT = [];
  for (let m = 0; m < M; m++) {
    MA[m] = MOVES[m][0]; MB[m] = MOVES[m][1]; MC[m] = MOVES[m][2];
    OFF[m] = MC[m] * IJ + MB[m] * I + MA[m]; LEN[m] = tabs.tables[m].len; TT.push(tabs.tables[m].t);
  }
  const lowerBounds = tileLowerBounds(nodeGrid, tabs, F, TT, LEN);
  const { nti, ntj, TI, TJ, TK, bound } = lowerBounds;
  const nodeH = clearance.nodeH, edgeZ = clearance.edgeZ;
  const dist = new Float64Array(N).fill(Infinity), pred = new Int32Array(N).fill(-1), visited = new Uint8Array(N);
  const heap = new IndexedHeap(N);
  dist[s] = 0; heap.set(0, s);
  while (heap.size) {
    const id = heap.pop();
    if (visited[id]) continue;
    visited[id] = 1;
    if (id === g) break;
    const k = Math.floor(id / IJ), rem = id - k * IJ, j = Math.floor(rem / I), i = rem - j * I;
    const pz = origin[2] + k * dz, eBase = rem * M, d0 = dist[id];
    const fa1 = F[4 * id], fc1 = F[4 * id + 1], fz1 = F[4 * id + 2];
    const lbBase = (((k / TK | 0) * ntj + (j / TJ | 0)) * nti + (i / TI | 0)) * M;
    const interior = i >= 2 && j >= 2 && k >= 1 && i < I - 2 && j < J - 2 && k < K - 1;
    for (let m = 0; m < M; m++) {
      const k2 = k + MC[m];
      if (!interior) {
        const i2 = i + MA[m], j2 = j + MB[m];
        if (i2 < 0 || j2 < 0 || k2 < 0 || i2 >= I || j2 >= J || k2 >= K) continue;
      }
      const id2 = id + OFF[m];
      if (visited[id2]) continue;
      if (d0 + bound[lbBase + m] >= dist[id2]) continue;
      if (!(origin[2] + k2 * dz >= nodeH[id2 - k2 * IJ])) continue;
      if (pz < edgeZ[eBase + m]) continue;
      const f2 = 4 * id2;
      const fa = 0.5 * (fa1 + F[f2]), fc = 0.5 * (fc1 + F[f2 + 1]), fz = 0.5 * (fz1 + F[f2 + 2]);
      const epm = tableValue(tabs, TT[m], fa, fc, fz);           // clamped to the table there
      if (!(epm < Infinity)) continue;
      const nd = d0 + epm * LEN[m];
      if (nd < dist[id2]) { dist[id2] = nd; pred[id2] = id; heap.set(nd, id2); }
    }
  }
  return { dist, pred };
}

const TI = 8, TJ = 8, TK = 4;          // tileLowerBounds: a tile's nodes along i, j, k
/**
 * Per tile of TI x TJ x TK nodes and per move: the smallest table value over the coordinate range of the tile's and its
 * neighbour tiles' corridor nodes, times the move length.  A move from a node of the tile ends in the tile or a
 * neighbour (moves reach 2 nodes across, 1 up), its edge coordinates (the mean of the two nodes') lie in that range,
 * and the trilinear value is at least the smallest entry of the covering cells: no edge costs less.
 */
function tileLowerBounds(nodeGrid, tabs, F, TT, LEN) {
  const { I, J, K } = nodeGrid, IJ = I * J, M = MOVES.length;
  const na = tabs.na, nc = tabs.nc, nz = tabs.nz;
  const faMax = na - TABLE_EDGE, fcMax = nc - TABLE_EDGE, fzMax = nz - TABLE_EDGE;
  const nti = Math.ceil(I / TI), ntj = Math.ceil(J / TJ), ntk = Math.ceil(K / TK), NT = nti * ntj * ntk;
  const box = new Float64Array(NT * 6);
  for (let t = 0; t < NT; t++) {
    box[6 * t] = Infinity; box[6 * t + 1] = -Infinity; box[6 * t + 2] = Infinity;
    box[6 * t + 3] = -Infinity; box[6 * t + 4] = Infinity; box[6 * t + 5] = -Infinity;
  }
  for (let k = 0; k < K; k++) for (let j = 0; j < J; j++) for (let i = 0; i < I; i++) {
    const rem = j * I + i;
    if (!nodeGrid.mask[rem]) continue;
    const id = k * IJ + rem, t = ((k / TK | 0) * ntj + (j / TJ | 0)) * nti + (i / TI | 0), b = 6 * t;
    const a = F[4 * id], c = F[4 * id + 1], z = F[4 * id + 2];
    if (a < box[b]) box[b] = a; if (a > box[b + 1]) box[b + 1] = a;
    if (c < box[b + 2]) box[b + 2] = c; if (c > box[b + 3]) box[b + 3] = c;
    if (z < box[b + 4]) box[b + 4] = z; if (z > box[b + 5]) box[b + 5] = z;
  }
  const bound = new Float64Array(NT * M);
  const clamp = (f, max) => (f < 0 ? 0 : f > max ? max : f);
  for (let tk = 0; tk < ntk; tk++) for (let tj = 0; tj < ntj; tj++) for (let ti = 0; ti < nti; ti++) {
    let aLo = Infinity, aHi = -Infinity, cLo = Infinity, cHi = -Infinity, zLo = Infinity, zHi = -Infinity;
    for (let dk = -1; dk <= 1; dk++) for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
      const a = ti + di, b2 = tj + dj, c = tk + dk;
      if (a < 0 || b2 < 0 || c < 0 || a >= nti || b2 >= ntj || c >= ntk) continue;
      const b = 6 * ((c * ntj + b2) * nti + a);
      if (box[b] < aLo) aLo = box[b]; if (box[b + 1] > aHi) aHi = box[b + 1];
      if (box[b + 2] < cLo) cLo = box[b + 2]; if (box[b + 3] > cHi) cHi = box[b + 3];
      if (box[b + 4] < zLo) zLo = box[b + 4]; if (box[b + 5] > zHi) zHi = box[b + 5];
    }
    const t = (tk * ntj + tj) * nti + ti;
    if (!(aLo <= aHi)) continue;                      // no corridor node near: bound 0 (never used)
    const i0 = Math.floor(clamp(aLo - 1e-3, faMax)), i1 = Math.min(na - 1, Math.floor(clamp(aHi + 1e-3, faMax)) + 1);
    const j0 = Math.floor(clamp(cLo - 1e-3, fcMax)), j1 = Math.min(nc - 1, Math.floor(clamp(cHi + 1e-3, fcMax)) + 1);
    const k0 = Math.floor(clamp(zLo - 1e-3, fzMax)), k1 = Math.min(nz - 1, Math.floor(clamp(zHi + 1e-3, fzMax)) + 1);
    for (let m = 0; m < M; m++) {
      let e = Infinity;
      const tt = TT[m];
      for (let a = i0; a <= i1; a++) for (let c = j0; c <= j1; c++) {
        const o = (a * nc + c) * nz;
        for (let z = k0; z <= k1; z++) { const x = tt[o + z]; if (x < e) e = x; }
      }
      bound[t * M + m] = Number.isFinite(e) && e > 0 ? e * LEN[m] * (1 - 1e-6) : 0;
    }
  }
  return { nti, ntj, TI, TJ, TK, bound };
}

/**
 * Node winds of a lattice for the ids in [idLo, idHi) into nodeWind (Float32, 3 per node): the same position and wind
 * evaluation as the lazy search, so the values are the ones it would compute when needed.  Nodes outside the corridor
 * (nodeGrid.mask) are never reached and get no wind.
 */
export function latticeNodeWinds(nodeGrid, wind, nodeWind, idLo, idHi) {
  const { I, J } = nodeGrid;
  const position = [0, 0, 0], wv = new Float64Array(3);
  for (let id = idLo; id < idHi; id++) {
    const k = Math.floor(id / (I * J)), rem = id - k * I * J, j = Math.floor(rem / I), i = rem - j * I;
    if (!nodeGrid.mask[rem]) {
      nodeWind[id * 3] = 0; nodeWind[id * 3 + 1] = 0; nodeWind[id * 3 + 2] = 0;
      continue;
    }
    wind(nodePositionInto(nodeGrid, i, j, k, position), wv);
    nodeWind[id * 3] = wv[0]; nodeWind[id * 3 + 1] = wv[1]; nodeWind[id * 3 + 2] = wv[2];
  }
}

/** Points evenly spaced by arc length along a polyline, about segM apart (flat Float64Array of [x, y, z]). */
export function resample(points, segM) {
  const pts = [points[0]];
  for (let i = 1; i < points.length; i++) {
    const a = pts[pts.length - 1], b = points[i];
    if (Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) > 1e-9) pts.push(b);
  }
  const s = [0];
  for (let i = 1; i < pts.length; i++) {
    s.push(s[i - 1] + Math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1], pts[i][2] - pts[i - 1][2]));
  }
  const total = s[s.length - 1];
  const n = Math.max(Math.ceil(total / segM), 2) + 1;
  const out = new Float64Array(n * 3);
  let seg = 0;
  for (let m = 0; m < n; m++) {
    const t = total * m / (n - 1);
    while (seg < pts.length - 2 && s[seg + 1] < t) seg++;
    const f = (t - s[seg]) / Math.max(s[seg + 1] - s[seg], 1e-12);
    for (let c = 0; c < 3; c++) out[3 * m + c] = pts[seg][c] + f * (pts[seg + 1][c] - pts[seg][c]);
  }
  return out;
}
