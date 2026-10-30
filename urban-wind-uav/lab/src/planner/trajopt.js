// The energy-optimal trajectory along a given route: the GeoWind2Plan trajectory problem (trapezoidal kinematics, one
// time step dt, the power model of energy.js, clearance, airspeed, acceleration and thrust limits) solved without
// IPOPT.
//
// The kinematics are eliminated exactly: with node accelerations a_k and dt as the only variables, the initial
// velocity follows from the goal condition and
//   x_k = S + k/(n-1) (G - S) + dt^2 sum_j M_kj a_j
//   v_k = (G - S) / ((n-1) dt)  + dt   sum_j N_kj a_j
// so only inequalities remain.  They are handled by a log barrier with a C2 quadratic extension (infeasible starts
// allowed), minimized by damped Newton steps; a route-tracking term w |x - P|^2 with w -> 0 keeps early iterates on
// the feasible side of buildings.  In a research configuration (4 m wind grid with Gaussian smoothing, the paper's
// SDF, no corridor, 11 barrier levels) the solver's results agree with IPOPT's (64 missions in each of two city
// blocks); this page's configuration (8 m wind samples, the safe-height surface ladder of plan_core.js, corridor,
// flight band and planning domain, 3 barrier levels) has no comparison of its own with IPOPT.
//
// Frame: metres, z up (gravity along -z).  fields (plan_fields.js): wind(p, out3, jac9), clear(p, grad3),
// extra(y, m, k, eg).
import { DemoError, ErrorCode } from "../shared/errors.js";
import { VEHICLE, DRAG, EPS } from "./energy.js";
import { structBuffers, structReset, structRecord, structAdd, mmtv, nmtv } from "./hessian.js";
import { TASK, NODE_STATE, NODE_TERM, VALUE_TERMS, MAX_MARGINS, TERM_HESSIAN, TERM_DT_CROSS, TERM_DT,
         shareOf, nodeStateOf, MAX_EXTRA_MARGINS } from "./pool_protocol.js";
import { cholesky, choleskyBlocked, choleskySolve } from "./linalg.js";

const { mass: M, rho: RHO, rotorArea: AR, profilePowerHover: PP, g: G0,
        maxAirspeed: VMAX, maxThrust: TMAX, maxAcceleration: AMAX } = VEHICLE;

const ENERGY_SCALE = 1e-3;           // the objective is energy (J) x this
const FD_STEP = 1e-5;                // relative finite-difference step of the second derivatives
const NEWTON_STEPS = 60;             // Newton steps per barrier weight at most
const CRUISE_SPEED = 12.0;           // m/s: the initial time step from the route length
const SEGMENT_POINTS = [1 / 3, 2 / 3];   // clearance samples between consecutive nodes (fractions)
const MIN_DT = 1e-3;                 // s: dt is kept above this by a barrier

/** y = the point (1 - al) x_k + al x_{k+1} between nodes k and k + 1 of the flat positions X. */
function segmentPoint(X, k, al, y) {
  for (let c = 0; c < 3; c++) y[c] = (1 - al) * X[3 * k + c] + al * X[3 * k + 3 + c];
}

/** Power (W) and its gradients w.r.t. airspeed q and acceleration a (the optimizer's form of energy.js power). */
function powerGrad(q, a, out) {
  const s = Math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + EPS);
  const T0 = M * a[0] + DRAG * s * q[0], T1 = M * a[1] + DRAG * s * q[1], T2 = M * a[2] + M * G0 + DRAG * s * q[2];
  const Tn = Math.sqrt(T0 * T0 + T1 * T1 + T2 * T2 + EPS);
  const den = Tn + EPS;
  const qT = q[0] * T0 + q[1] * T1 + q[2] * T2;
  const vc = qT / den;
  const R = Math.sqrt(0.25 * vc * vc + Tn / (2 * RHO * AR) + EPS);
  const vi = -0.5 * vc + R;
  const ratio = Tn / (M * G0);
  const P = qT + Tn * vi + PP * ratio * Math.sqrt(ratio);
  out.P = P; out.Tn = Tn; out.s = s;
  out.T[0] = T0; out.T[1] = T1; out.T[2] = T2;
  if (!out.gq) return out;
  const dviDvc = -0.5 + vc / (4 * R), dviDTn = 1 / (4 * RHO * AR * R);
  const A1 = Tn * dviDvc;
  const A2 = vi + Tn * dviDTn + 1.5 * PP * Math.sqrt(ratio) / (M * G0);
  const cT = A2 - A1 * qT / (den * den);
  const gT0 = q[0] + A1 * q[0] / den + cT * T0 / Tn;
  const gT1 = q[1] + A1 * q[1] / den + cT * T1 / Tn;
  const gT2 = q[2] + A1 * q[2] / den + cT * T2 / Tn;
  const qg = q[0] * gT0 + q[1] * gT1 + q[2] * gT2;
  out.gq[0] = T0 + A1 * T0 / den + DRAG * (s * gT0 + q[0] * qg / s);
  out.gq[1] = T1 + A1 * T1 / den + DRAG * (s * gT1 + q[1] * qg / s);
  out.gq[2] = T2 + A1 * T2 / den + DRAG * (s * gT2 + q[2] * qg / s);
  out.ga[0] = M * gT0; out.ga[1] = M * gT1; out.ga[2] = M * gT2;
  // the thrust norm's gradient w.r.t. q and a (the thrust limit)
  const u0 = T0 / Tn, u1 = T1 / Tn, u2 = T2 / Tn;
  const qu = q[0] * u0 + q[1] * u1 + q[2] * u2;
  out.tq[0] = DRAG * (s * u0 + q[0] * qu / s);
  out.tq[1] = DRAG * (s * u1 + q[1] * qu / s);
  out.tq[2] = DRAG * (s * u2 + q[2] * qu / s);
  out.ta[0] = M * u0; out.ta[1] = M * u1; out.ta[2] = M * u2;
  return out;
}

const POWER_WITH_GRADIENT = { T: new Float64Array(3), gq: new Float64Array(3), ga: new Float64Array(3),
                              tq: new Float64Array(3), ta: new Float64Array(3) };
const POWER_VALUE = { T: new Float64Array(3) };

/**
 * Node quantities for y = (x, v, a) (9): the power P and the margins [clear, acc, air, thrust, ...extra], each with its
 * gradient w.r.t. y when wantGrad.  Values in m[], gradients in gm[i * 9 .. i * 9 + 8], the power gradient in gp.
 * Returns { P, nm } (nm: the number of margins).
 * pos: the position-only parts (wind and its Jacobian, clearance, extra margins) of the base point, stored by the base
 * evaluation and restored when only v or a is perturbed (pos.use): the same values, computed once instead of seven
 * times.
 */
function nodeEval(fields, y, wantGrad, gp, m, gm, scratch, pos = null) {
  const { w, J, cg, eg } = scratch;
  const use = pos && pos.use;
  if (use) { w.set(pos.w); J.set(pos.J); } else { fields.wind(y, w, J); }
  if (pos && !use) { pos.w.set(w); pos.J.set(J); }
  const q = scratch.q;
  q[0] = y[3] - w[0]; q[1] = y[4] - w[1]; q[2] = y[5] - w[2];
  const a = scratch.a;
  a[0] = y[6]; a[1] = y[7]; a[2] = y[8];
  const pw = wantGrad ? POWER_WITH_GRADIENT : POWER_VALUE;
  powerGrad(q, a, pw);
  let nm = 0;
  // clearance
  if (use) { m[nm] = pos.clear; cg.set(pos.cg); } else { m[nm] = fields.clear(y, cg); }
  if (pos && !use) { pos.clear = m[nm]; pos.cg.set(cg); }
  if (wantGrad) { gm.fill(0, nm * 9, nm * 9 + 9); gm[nm * 9] = cg[0]; gm[nm * 9 + 1] = cg[1]; gm[nm * 9 + 2] = cg[2]; }
  nm++;
  // acceleration
  const an = Math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2] + EPS);
  m[nm] = AMAX - an;
  if (wantGrad) { gm.fill(0, nm * 9, nm * 9 + 9); for (let c = 0; c < 3; c++) gm[nm * 9 + 6 + c] = -a[c] / an; }
  nm++;
  // airspeed
  m[nm] = VMAX - pw.s;
  if (wantGrad) {
    gm.fill(0, nm * 9, nm * 9 + 9);
    for (let c = 0; c < 3; c++) {
      const d = -q[c] / pw.s;
      gm[nm * 9 + 3 + c] = d;
      for (let r = 0; r < 3; r++) gm[nm * 9 + r] -= d * J[c * 3 + r];      // dq/dx = -J
    }
  }
  nm++;
  // thrust
  m[nm] = TMAX - pw.Tn;
  if (wantGrad) {
    gm.fill(0, nm * 9, nm * 9 + 9);
    for (let c = 0; c < 3; c++) {
      gm[nm * 9 + 3 + c] = -pw.tq[c];
      gm[nm * 9 + 6 + c] = -pw.ta[c];
      for (let r = 0; r < 3; r++) gm[nm * 9 + r] += pw.tq[c] * J[c * 3 + r];
    }
  }
  nm++;
  // corridor, band, domain
  let k;
  if (use) {
    k = pos.k;
    for (let i = 0; i < k; i++) m[nm + i] = pos.em[i];
    eg.set(pos.eg.subarray(0, 3 * k));
  } else {
    k = fields.extra(y, m, nm, eg);
  }
  if (pos && !use) { pos.k = k; for (let i = 0; i < k; i++) pos.em[i] = m[nm + i]; pos.eg.set(eg.subarray(0, 3 * k)); }
  if (wantGrad) for (let i = 0; i < k; i++) {
    gm.fill(0, (nm + i) * 9, (nm + i) * 9 + 9);
    gm[(nm + i) * 9] = eg[i * 3]; gm[(nm + i) * 9 + 1] = eg[i * 3 + 1]; gm[(nm + i) * 9 + 2] = eg[i * 3 + 2];
  }
  nm += k;
  if (wantGrad) {
    for (let c = 0; c < 3; c++) {
      gp[3 + c] = pw.gq[c];
      gp[6 + c] = pw.ga[c];
      let gx = 0;
      for (let r = 0; r < 3; r++) gx -= pw.gq[r] * J[r * 3 + c];
      gp[c] = gx;
    }
  }
  return { P: pw.P, nm };
}

/** The barrier with a C2 quadratic extension below tau = mu: value, first and second derivative into out. */
function barrier(s, mu, out) {
  const tau = mu;
  if (s >= tau) {
    out[0] = -mu * Math.log(s); out[1] = -mu / s; out[2] = mu / (s * s);
  } else {
    const d = s - tau;
    out[0] = -mu * Math.log(tau) - mu * d / tau + mu * d * d / (2 * tau * tau);
    out[1] = -mu / tau + mu * d / (tau * tau);
    out[2] = mu / (tau * tau);
  }
  return out;
}

/** The trapezoid weight of node k of n (energy = dt sum omega_k P_k). */
const omegaOf = (k, n) => (k === 0 || k === n - 1 ? 0.5 : 1.0);

/** Work arrays of nodeEval (one set per user: the optimizer's own, and each node environment's). */
const nodeScratch = () => ({ w: new Float64Array(3), J: new Float64Array(9), cg: new Float64Array(3),
                             eg: new Float64Array(3 * MAX_EXTRA_MARGINS), q: new Float64Array(3),
                             a: new Float64Array(3) });

/**
 * Buffers for the node functions below, shared by the optimizer and the helper threads (helper_worker.js), which
 * compute a share of the nodes with the same code.  P: route points (tracking term).
 */
export function makeNodeEnv(fields, n, P) {
  return {
    fields, n, P,
    scratch: nodeScratch(),
    yp: new Float64Array(NODE_STATE), gp: new Float64Array(NODE_STATE), gpp: new Float64Array(NODE_STATE),
    mm: new Float64Array(MAX_MARGINS), gm: new Float64Array(MAX_MARGINS * NODE_STATE),
    mmp: new Float64Array(MAX_MARGINS), gmp: new Float64Array(MAX_MARGINS * NODE_STATE),
    bb: new Float64Array(3),
    posCache: { use: false, w: new Float64Array(3), J: new Float64Array(9), clear: 0, cg: new Float64Array(3), k: 0,
                em: new Float64Array(MAX_EXTRA_MARGINS),
                eg: new Float64Array(3 * MAX_EXTRA_MARGINS) },
  };
}

/**
 * Node k's term of the objective: gradient gyOut (9) and Hessian HyOut (81) in y, the dt-y cross terms HdtyOut (9);
 * returns the explicit dt derivative.
 */
export function nodeTermEnv(env, k, y, mu, wt, dtv, gyOut, HyOut, HdtyOut) {
  const { fields, n, P, scratch, yp, gp, gpp, mm, gm, mmp, gmp, bb, posCache } = env;
  const wgt = ENERGY_SCALE * dtv * omegaOf(k, n);
  posCache.use = false;
  const r = nodeEval(fields, y, true, gp, mm, gm, scratch, posCache);
  gyOut.fill(0); HyOut.fill(0);
  for (let i = 0; i < 9; i++) gyOut[i] = wgt * gp[i];
  const d1 = new Float64Array(r.nm);
  const ends = k === 0 || k === n - 1;
  for (let i = 0; i < r.nm; i++) {
    if (ends && i === 0) { d1[i] = 0; continue; }   // end-point clearance: constant (see nodeValueTerms)
    barrier(mm[i], mu, bb);
    d1[i] = bb[1];
    for (let p = 0; p < 9; p++) {
      gyOut[p] += bb[1] * gm[i * 9 + p];
      for (let q = 0; q < 9; q++) HyOut[p * 9 + q] += bb[2] * gm[i * 9 + p] * gm[i * 9 + q];
    }
  }
  // second derivatives of P and of the margins by finite differences of the analytic gradients
  for (let q = 0; q < 9; q++) {
    for (let p = 0; p < 9; p++) yp[p] = y[p];
    const step = FD_STEP * Math.max(1, Math.abs(y[q]));
    yp[q] += step;
    posCache.use = q >= 3;          // q >= 3: only v or a perturbed
    nodeEval(fields, yp, true, gpp, mmp, gmp, scratch, q >= 3 ? posCache : null);
    for (let p = 0; p < 9; p++) {
      let v = wgt * (gpp[p] - gp[p]) / step;
      for (let i = 0; i < r.nm; i++) v += d1[i] * (gmp[i * 9 + p] - gm[i * 9 + p]) / step;
      HyOut[p * 9 + q] += v;
    }
  }
  for (let p = 0; p < 9; p++) for (let q = p + 1; q < 9; q++) {
    const s = 0.5 * (HyOut[p * 9 + q] + HyOut[q * 9 + p]); HyOut[p * 9 + q] = s; HyOut[q * 9 + p] = s;
  }
  // explicit dt: d/ddt (s dt omega P) = s omega P (s = ENERGY_SCALE); cross with y: s omega gp
  for (let i = 0; i < 9; i++) HdtyOut[i] = ENERGY_SCALE * omegaOf(k, n) * gp[i];
  if (wt) {
    for (let c = 0; c < 3; c++) { gyOut[c] += 2 * wt * (y[c] - P[3 * k + c]) / n; HyOut[c * 9 + c] += 2 * wt / n; }
  }
  return ENERGY_SCALE * omegaOf(k, n) * r.P;
}

/**
 * The numbers node k adds to the objective value, in the order the objective adds them: out[o] = count, then the
 * numbers.  End points: every margin except clearance (index 0), as the paper constrains all nodes; the end positions
 * are fixed, so their clearance is a constant, and the conservative surface may lie above an end point the exact test
 * accepts.
 */
export function nodeValueTerms(env, k, y, mu, wt, dtv, out, o) {
  const { fields, n, P, mm, scratch, bb } = env;
  const r = nodeEval(fields, y, false, null, mm, null, scratch);
  let count = 0;
  out[o + 1 + count++] = ENERGY_SCALE * dtv * omegaOf(k, n) * r.P;
  for (let i = k > 0 && k < n - 1 ? 0 : 1; i < r.nm; i++) out[o + 1 + count++] = barrier(mm[i], mu, bb)[0];
  if (wt) for (let c = 0; c < 3; c++) { const d = y[c] - P[3 * k + c]; out[o + 1 + count++] = wt * d * d / n; }
  out[o] = count;
  return count;
}

/** The kinematic maps M (positions) and N (velocities), n x n row-major, and the partial sums U and Y. */
function kinematicMaps(n) {
  const u = new Float64Array(n * n), y = new Float64Array(n * n);
  for (let k = 1; k < n; k++) {
    for (let j = 0; j < n; j++) u[k * n + j] = u[(k - 1) * n + j];
    u[k * n + k - 1] += 0.5; u[k * n + k] += 0.5;
  }
  for (let k = 1; k < n; k++) for (let j = 0; j < n; j++) {
    y[k * n + j] = y[(k - 1) * n + j] + 0.5 * (u[(k - 1) * n + j] + u[k * n + j]);
  }
  const Mm = new Float64Array(n * n), Nm = new Float64Array(n * n);
  for (let k = 0; k < n; k++) for (let j = 0; j < n; j++) {
    Mm[k * n + j] = y[k * n + j] - (k / (n - 1)) * y[(n - 1) * n + j];
    Nm[k * n + j] = u[k * n + j] - y[(n - 1) * n + j] / (n - 1);
  }
  return { Mm, Nm, U: u, Y: y };
}

/**
 * The optimization problem of one route: variables z = (a_0 .. a_{n-1}, dt) (3 n + 1), its state (positions and
 * velocities), objective value, gradient and Hessian (structured assembly, hessian.js), and the damped Newton solver.
 * With helper threads (opts.pool) the node terms are computed in shares by them.
 */
class TrajectoryProblem {
  constructor(P, fields, opts) {
    const n = this.n = P.length / 3;
    const mVar = this.mVar = 3 * n + 1;
    Object.assign(this, { P, fields, opts });
    this.S = [P[0], P[1], P[2]];
    const goal = [P[3 * n - 3], P[3 * n - 2], P[3 * n - 1]];
    this.D = [goal[0] - this.S[0], goal[1] - this.S[1], goal[2] - this.S[2]];
    const { Mm, Nm, U, Y } = kinematicMaps(n);
    Object.assign(this, { Mm, Nm, U, Y });
    this.z = new Float64Array(mVar);
    this.X = new Float64Array(3 * n); this.V = new Float64Array(3 * n);
    this.Xa = new Float64Array(3 * n); this.Va = new Float64Array(3 * n);   // sum_j M a, sum_j N a
    this.scratch = nodeScratch();
    this.y = new Float64Array(NODE_STATE);
    this.mm = new Float64Array(MAX_MARGINS);
    this.bb = new Float64Array(3);
    this.gz = new Float64Array(mVar);
    this.Hz = new Float64Array(mVar * mVar);
    this.env = makeNodeEnv(fields, n, P);
    this.pool = opts.pool;             // the helper threads sharing the node terms (plan_core.js), or null
    this.valueTerms = new Float64Array(VALUE_TERMS);
    // structured Hessian assembly
    this.hctx = { n, mVar, Hz: this.Hz, Mm, Nm, dtv: 0 };
    this.acc = structBuffers(n);
    this.eRow = Y.subarray((n - 1) * n, n * n);
    const perAxis = () => [0, 1, 2].map(() => new Float64Array(n));
    this.GX = perAxis(); this.GV = perAxis(); this.AL = perAxis(); this.BE = perAxis();
    this.dtcol = new Float64Array(3 * n); this.tvec = new Float64Array(n);
    this.dydt = new Float64Array(9); this.d2ydt = new Float64Array(9); this.hdt = new Float64Array(9);
    this.gyb = new Float64Array(9); this.Hyb = new Float64Array(81); this.Hdtyb = new Float64Array(9);
  }

  /** The initial point: dt from the cruise speed, accelerations by smoothed least squares along the route. */
  initialPoint() {
    const { n, P, Mm, S, D, z, mVar } = this;
    let length = 0;
    for (let k = 0; k < n - 1; k++) {
      length += Math.hypot(P[3 * k + 3] - P[3 * k], P[3 * k + 4] - P[3 * k + 1], P[3 * k + 5] - P[3 * k + 2]);
    }
    const seg = length / (n - 1);
    const dt = seg / CRUISE_SPEED;
    const lam = 1e-2 * seg * seg;
    const A = new Float64Array(n * n), L = new Float64Array(n * n);
    const rhs = new Float64Array(n), sol = new Float64Array(n);
    const d2 = dt * dt;
    for (let j = 0; j < n; j++) for (let l = 0; l < n; l++) {
      let s = 0;
      for (let k = 0; k < n; k++) s += Mm[k * n + j] * Mm[k * n + l];
      A[j * n + l] = d2 * d2 * s;
    }
    for (let j = 0; j < n - 1; j++) {
      A[j * n + j] += lam; A[(j + 1) * n + j + 1] += lam; A[j * n + j + 1] -= lam; A[(j + 1) * n + j] -= lam;
    }
    for (let j = 0; j < n; j++) A[j * n + j] += 1e-9 * (1 + A[j * n + j]);
    // positive definite for finite route points: a failure means non-finite points
    if (!cholesky(A, n, L)) throw new Error("the route's initial point could not be computed (non-finite points)");
    for (let c = 0; c < 3; c++) {
      for (let j = 0; j < n; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) s += Mm[k * n + j] * (P[3 * k + c] - S[c] - (k / (n - 1)) * D[c]);
        rhs[j] = d2 * s;
      }
      choleskySolve(L, n, rhs, sol);
      for (let j = 0; j < n; j++) z[3 * j + c] = sol[j];
    }
    z[mVar - 1] = dt;
  }

  /** Positions X and velocities V (and the partial sums Xa, Va) of the variables zz. */
  state(zz) {
    const { n, mVar, Mm, Nm, S, D, X, V, Xa, Va } = this;
    const dtv = zz[mVar - 1];
    for (let k = 0; k < n; k++) for (let c = 0; c < 3; c++) {
      let sm = 0, sn = 0;
      for (let j = 0; j < n; j++) { const aj = zz[3 * j + c]; sm += Mm[k * n + j] * aj; sn += Nm[k * n + j] * aj; }
      Xa[3 * k + c] = sm; Va[3 * k + c] = sn;
      X[3 * k + c] = S[c] + (k / (n - 1)) * D[c] + dtv * dtv * sm;
      V[3 * k + c] = D[c] / ((n - 1) * dtv) + dtv * sn;
    }
  }

  /** y = (x, v, a) of node k at the current state and variables zz. */
  nodeState(k, zz, y) {
    const { X, V } = this;
    for (let c = 0; c < 3; c++) { y[c] = X[3 * k + c]; y[3 + c] = V[3 * k + c]; y[6 + c] = zz[3 * k + c]; }
  }

  /** The node states of every node into the helpers' shared buffer. */
  shareNodeStates(zz) {
    const ys = this.pool.ybuf;
    for (let k = 0; k < this.n; k++) this.nodeState(k, zz, nodeStateOf(ys, k));
  }

  /** The objective value at zz. */
  value(zz, mu, wt) {
    const { n, mVar, X, y, env, fields, bb } = this;
    const dtv = zz[mVar - 1];
    if (!(dtv > 0)) return Infinity;
    this.state(zz);
    let F = 0;
    // the nodes' numbers (nodeValueTerms), added in node order; with helpers they compute a share
    if (this.pool) {
      const ys = this.pool.ybuf, valOut = this.pool.valOut;
      this.shareNodeStates(zz);
      this.pool.optRun(TASK.NODE_VALUES, n, mu, wt, dtv, (parts) => {
        const [, hi] = shareOf(0, parts, n);       // this thread's share (from node 0)
        for (let k = 0; k < hi; k++) {
          nodeValueTerms(env, k, nodeStateOf(ys, k), mu, wt, dtv, valOut, VALUE_TERMS * k);
        }
      });
      for (let k = 0; k < n; k++) {
        const o = VALUE_TERMS * k, count = valOut[o];
        for (let i = 1; i <= count; i++) F += valOut[o + i];
      }
    } else {
      const terms = this.valueTerms;
      for (let k = 0; k < n; k++) {
        this.nodeState(k, zz, y);
        nodeValueTerms(env, k, y, mu, wt, dtv, terms, 0);
        for (let i = 1; i <= terms[0]; i++) F += terms[i];
      }
    }
    for (let k = 0; k < n - 1; k++) for (const al of SEGMENT_POINTS) {
      segmentPoint(X, k, al, y);
      F += barrier(fields.clear(y, this.scratch.cg), mu, bb)[0];
    }
    F += barrier(dtv - MIN_DT, mu, bb)[0];
    return F;
  }

  /** Start the structured Hessian assembly of one Newton step (hessian.js): every accumulator cleared. */
  structStart() {
    structReset(this.acc, this.n);
    for (let c = 0; c < 3; c++) { this.GX[c].fill(0); this.GV[c].fill(0); this.AL[c].fill(0); this.BE[c].fill(0); }
    this.dtcol.fill(0);
  }

  /**
   * One objective term at node k (or at the segment point (1 - al) x_k + al x_{k+1} when isMix) with gradient gy and
   * Hessian Hy in y = (x, v, a), explicit dt derivative gdtExplicit and dt-y cross terms Hdty: O(1) here (81 + small),
   * the O(n^2) part in structFinish.  A node term T(y, dt) with y_x = S + .. + dt^2 M a, y_v = D/((n-1)dt) + dt N a,
   * y_a = a_k.
   */
  addTerm(k, gy, Hy, gdtExplicit, Hdty, isMix = false, al = 0) {
    const { n, mVar, z, Xa, Va, D, GX, GV, AL, BE, gz, Hz, dtcol, dydt, d2ydt, hdt } = this;
    const dtv = z[mVar - 1], d2 = dtv * dtv, last = mVar - 1;
    for (let c = 0; c < 3; c++) {          // dy/ddt and d2y/ddt2 of the term's y
      const xa = isMix ? (1 - al) * Xa[3 * k + c] + al * Xa[3 * k + 3 + c] : Xa[3 * k + c];
      dydt[c] = 2 * dtv * xa; d2ydt[c] = 2 * xa;
      if (isMix) {
        dydt[3 + c] = 0; d2ydt[3 + c] = 0;
      } else {
        dydt[3 + c] = -D[c] / ((n - 1) * d2) + Va[3 * k + c]; d2ydt[3 + c] = 2 * D[c] / ((n - 1) * d2 * dtv);
      }
      dydt[6 + c] = 0; d2ydt[6 + c] = 0;
    }
    for (let c = 0; c < 3; c++) {          // gradient: rows of the node (segment: its two nodes' mix)
      if (isMix) {
        GX[c][k] += (1 - al) * gy[c]; GX[c][k + 1] += al * gy[c];
      } else {
        GX[c][k] += gy[c]; GV[c][k] += gy[3 + c]; gz[3 * k + c] += gy[6 + c];
      }
    }
    let gdt = gdtExplicit;
    for (let r = 0; r < 9; r++) gdt += gy[r] * dydt[r];
    gz[last] += gdt;
    structRecord(this.acc, n, k, isMix, al, Hy);
    for (let r = 0; r < 9; r++) {
      let sum = Hdty ? Hdty[r] : 0;
      for (let q = 0; q < 9; q++) sum += Hy[r * 9 + q] * dydt[q];
      hdt[r] = sum;
    }
    for (let c = 0; c < 3; c++) {          // dt row / column
      const a = d2 * hdt[c] + 2 * dtv * gy[c];
      if (isMix) {
        AL[c][k] += (1 - al) * a; AL[c][k + 1] += al * a;
      } else {
        AL[c][k] += a; BE[c][k] += dtv * hdt[3 + c] + gy[3 + c]; dtcol[3 * k + c] += hdt[6 + c];
      }
    }
    let hdd = 0;
    for (let r = 0; r < 9; r++) hdd += dydt[r] * hdt[r] + gy[r] * d2ydt[r];
    if (Hdty) for (let r = 0; r < 9; r++) hdd += Hdty[r] * dydt[r];
    Hz[last * mVar + last] += hdd;
  }

  /** Finish the assembly: the accumulated node terms into the gradient gz and the Hessian Hz. */
  structFinish() {
    const { n, mVar, z, GX, GV, AL, BE, eRow, tvec, gz, Hz, dtcol } = this;
    const dtv = z[mVar - 1], d2 = dtv * dtv, last = mVar - 1;
    this.hctx.dtv = dtv;
    for (let c = 0; c < 3; c++) {
      mmtv(GX[c], eRow, tvec, n);
      for (let j = 0; j < n; j++) gz[3 * j + c] += d2 * tvec[j];
      nmtv(GV[c], eRow, tvec, n);
      for (let j = 0; j < n; j++) gz[3 * j + c] += dtv * tvec[j];
      mmtv(AL[c], eRow, tvec, n);
      for (let j = 0; j < n; j++) dtcol[3 * j + c] += tvec[j];
      nmtv(BE[c], eRow, tvec, n);
      for (let j = 0; j < n; j++) dtcol[3 * j + c] += tvec[j];
    }
    for (let r = 0; r < 3 * n; r++) { Hz[r * mVar + last] += dtcol[r]; Hz[last * mVar + r] += dtcol[r]; }
    structAdd(this.hctx, this.acc, this.Y, this.U);
  }

  /** Gradient gz and Hessian Hz of the objective at the current variables z. */
  gradHess(mu, wt) {
    const { n, mVar, z, X, y, env, fields, bb, gz, Hz, gyb, Hyb, Hdtyb } = this;
    gz.fill(0); Hz.fill(0);
    this.structStart();
    const dtv = z[mVar - 1];
    this.state(z);
    if (this.pool) {
      // node terms: a share each on the helpers and here (nodeTermEnv), then added in node order
      const ys = this.pool.ybuf, nodeOut = this.pool.nodeOut;
      this.shareNodeStates(z);
      this.pool.optRun(TASK.NODE_TERMS, n, mu, wt, dtv, (parts) => {
        const [, hi] = shareOf(0, parts, n);       // this thread's share (from node 0)
        for (let k = 0; k < hi; k++) {
          const o = NODE_TERM * k;
          nodeOut[o + TERM_DT] = nodeTermEnv(env, k, nodeStateOf(ys, k), mu, wt, dtv,
                                             nodeOut.subarray(o, o + TERM_HESSIAN),
                                             nodeOut.subarray(o + TERM_HESSIAN, o + TERM_DT_CROSS),
                                             nodeOut.subarray(o + TERM_DT_CROSS, o + TERM_DT));
        }
      });
      for (let k = 0; k < n; k++) {
        const o = NODE_TERM * k;
        this.addTerm(k, nodeOut.subarray(o, o + TERM_HESSIAN), nodeOut.subarray(o + TERM_HESSIAN, o + TERM_DT_CROSS),
                     nodeOut[o + TERM_DT], nodeOut.subarray(o + TERM_DT_CROSS, o + TERM_DT));
      }
    } else {
      for (let k = 0; k < n; k++) {
        this.nodeState(k, z, y);
        const gdt = nodeTermEnv(env, k, y, mu, wt, dtv, gyb, Hyb, Hdtyb);
        this.addTerm(k, gyb, Hyb, gdt, Hdtyb);
      }
    }
    // segment clearance samples: point p = (1 - al) x_k + al x_{k+1}
    const gc = new Float64Array(3), gc2 = new Float64Array(3), pp = new Float64Array(3);
    for (let k = 0; k < n - 1; k++) for (const al of SEGMENT_POINTS) {
      segmentPoint(X, k, al, y);
      const cv = fields.clear(y, gc);
      barrier(cv, mu, bb);
      gyb.fill(0); Hyb.fill(0);
      for (let c = 0; c < 3; c++) gyb[c] = bb[1] * gc[c];
      for (let p = 0; p < 3; p++) for (let q = 0; q < 3; q++) Hyb[p * 9 + q] = bb[2] * gc[p] * gc[q];
      for (let q = 0; q < 3; q++) {
        pp[0] = y[0]; pp[1] = y[1]; pp[2] = y[2];
        const step = FD_STEP * Math.max(1, Math.abs(y[q]));
        pp[q] += step;
        fields.clear(pp, gc2);
        for (let p = 0; p < 3; p++) Hyb[p * 9 + q] += bb[1] * (gc2[p] - gc[p]) / step;
      }
      for (let p = 0; p < 3; p++) for (let q = p + 1; q < 3; q++) {
        const s = 0.5 * (Hyb[p * 9 + q] + Hyb[q * 9 + p]); Hyb[p * 9 + q] = s; Hyb[q * 9 + p] = s;
      }
      this.addTerm(k, gyb, Hyb, 0, null, true, al);
    }
    this.structFinish();
    barrier(dtv - MIN_DT, mu, bb);
    gz[mVar - 1] += bb[1];
    Hz[(mVar - 1) * mVar + mVar - 1] += bb[2];
  }

  /**
   * Damped Newton steps per barrier weight (opts.mus, with route-tracking weights opts.tracks).  The step uses the
   * smallest damping of the sequence 0, d1, 4 d1, 16 d1, ... (d1 = 1e-8 max(1, max |diag|)) whose matrix factorizes.
   * A larger damping only adds to the diagonal, so above the smallest level every level factorizes and below it none:
   * the search starts one level below the previous step's and goes up or down only as far as needed - the same level
   * as trying 0, d1, ... in turn, with fewer factorizations.  A matrix that no damping makes positive definite
   * (non-finite values) ends the optimization with an error.  Returns the number of steps taken.
   */
  solve() {
    const { mVar, z, gz, Hz, opts } = this;
    const L = new Float64Array(mVar * mVar), A = new Float64Array(mVar * mVar);
    const dir = new Float64Array(mVar), trial = new Float64Array(mVar), negg = new Float64Array(mVar);
    const L2 = new Float64Array(mVar * mVar);
    let lastLevel = 0, steps = 0;
    for (let st = 0; st < opts.mus.length; st++) {
      const mu = opts.mus[st], wt = opts.tracks[st];
      for (let it = 0; it < NEWTON_STEPS; it++) {
        // a superseded computation stops here (between steps the helper threads are idle)
        if (opts.halt && opts.halt()) throw new DemoError(ErrorCode.CANCELLED, "cancelled");
        this.gradHess(mu, wt);
        const F0 = this.value(z, mu, wt);
        let maxDiag = 0;
        for (let i = 0; i < mVar; i++) maxDiag = Math.max(maxDiag, Math.abs(Hz[i * mVar + i]));
        const levelDamp = (j) => {
          let d = 0;
          for (let t = 1; t <= j; t++) d = Math.max(4 * d, 1e-8 * Math.max(1, maxDiag));
          return d;
        };
        const tryLevel = (j, out) => {
          const d = levelDamp(j);
          if (j > 200 || !Number.isFinite(d)) throw new Error("no positive definite Newton step (non-finite values)");
          A.set(Hz);
          for (let i = 0; i < mVar; i++) A[i * mVar + i] += d;
          return choleskyBlocked(A, mVar, out);
        };
        let j = Math.max(0, lastLevel - 1);
        if (tryLevel(j, L)) {
          while (j > 0 && tryLevel(j - 1, L2)) { L.set(L2); j--; }
        } else {
          do j++; while (!tryLevel(j, L));
        }
        lastLevel = j;
        for (let i = 0; i < mVar; i++) negg[i] = -gz[i];
        choleskySolve(L, mVar, negg, dir);
        let dec = 0;
        for (let i = 0; i < mVar; i++) dec -= gz[i] * dir[i];
        if (dec < 1e-12) break;
        let step = 1, accepted = false;
        for (;;) {
          for (let i = 0; i < mVar; i++) trial[i] = z[i] + step * dir[i];
          const F1 = this.value(trial, mu, wt);
          if (F1 <= F0 - 1e-4 * step * dec) { accepted = true; break; }
          if (step < 1e-12) break;
          step *= 0.5;
        }
        if (!accepted) break;          // no decrease possible: stop this barrier weight
        z.set(trial);
        steps++;
        if (dec < 1e-10) break;
      }
    }
    return steps;
  }

  /** The trajectory at the current variables: { x, v, dt, energy, limitViolation }. */
  result() {
    const { n, mVar, z, X, V, y, mm, fields, scratch } = this;
    this.state(z);
    let limitViolation = 0;
    const power = new Float64Array(n);
    for (let k = 0; k < n; k++) {
      this.nodeState(k, z, y);
      const r = nodeEval(fields, y, false, null, mm, null, scratch);
      power[k] = r.P;
      // acceleration, airspeed, thrust, extras (clearance, i = 0, is checked exactly by the caller)
      for (let i = 1; i < r.nm; i++) limitViolation = Math.max(limitViolation, -mm[i]);
    }
    const dtv = z[mVar - 1];
    let energy = 0;
    for (let k = 0; k < n; k++) energy += dtv * omegaOf(k, n) * power[k];
    return { x: Float64Array.from(X), v: Float64Array.from(V),
             dt: dtv, energy, limitViolation };
  }
}

/**
 * Optimize a trajectory along route points P (n x 3, flat), first = start, last = goal.  opts: mus (barrier
 * weights), tracks (route-tracking weight per barrier weight), pool (helper_pool.js, started and large enough for
 * this route, or null: this thread computes every node term), halt (() -> true when the computation is cancelled).
 * Returns { x, v (flat n x 3), dt, energy (J, planning wind), limitViolation, steps }.
 */
export function optimizeTrajectory(P, fields, opts) {
  const problem = new TrajectoryProblem(P, fields, opts);
  problem.initialPoint();
  const steps = problem.solve();
  return { ...problem.result(), steps };
}
