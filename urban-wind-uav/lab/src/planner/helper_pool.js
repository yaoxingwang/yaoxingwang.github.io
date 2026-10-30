// Helper threads of one planner worker (helper_worker.js).  In shared memory they share the planner's wind grid rows,
// the lattice's node winds and the trajectory optimizer's node terms (pool_protocol.js TASK): the planner thread does
// the first share, each helper one of the others, and every value is computed by one thread with the same code as one
// thread alone, so the results are the same for any number of helpers (one or more).  Needs shared memory (a
// cross-origin isolated page); without it, or before every helper has started, the planner computes everything itself
// and the lattice search is the lazy one (lattice.js latticeRoute: the same route up to rounding).  No unbounded wait:
// a helper that does not take a plan's data or finish a task in time makes the planning fail with an error, and the
// helpers are then ended (later plans are computed by the planner thread alone).
// Shared memory: ctrl (pool_protocol.js CTRL); ack Int32 [helpers]: the sequence number of the last plan data each
// helper has taken (minus it: it could not take them); prm Float64 [mu, wt, dt]; ybuf Float64 [node][NODE_STATE]:
// node states; nodeOut / valOut Float64: node terms / values (NODE_TERM, VALUE_TERMS per node).
import { DemoError, ErrorCode } from "../shared/errors.js";
import { sharedMemory } from "../shared/platform.js";
import { windGridShape, windGridRows, gridResult } from "./wind_grid.js";
import { latticeNodeWinds } from "./lattice.js";
import { TASK, CTRL, NODE_STATE, NODE_TERM, VALUE_TERMS, shareOf } from "./pool_protocol.js";

const ACK_LIMIT_MS = 10000;          // a helper must take a plan's data within this
const TASK_LIMIT_MS = 30000;         // and finish a task within this
// the waits for them are sliced, so each limit above is checked while waiting
const WAIT_SLICE_MS = 200, TASK_WAIT_SLICE_MS = 1000;
const MAX_NODES = 200;               // the largest route (optimizer nodes) the shared matrices hold

/** `count` doubles in shared memory. */
const sharedFloat64 = (count) => new Float64Array(new SharedArrayBuffer(8 * count));
/** `buffer` if it holds `bytes`, else a new shared buffer of that size. */
const atLeast = (buffer, bytes) => (buffer && buffer.byteLength >= bytes ? buffer : new SharedArrayBuffer(bytes));

/** A pool of `helpers` threads for routes of up to MAX_NODES optimizer nodes, or null (no helpers or shared memory). */
export function createPool(helpers) {
  if (!(helpers > 0) || !sharedMemory()) return null;
  return new HelperPool(helpers);
}

/** The helper threads, their shared buffers, and the tasks the planner thread hands them (see the top of this file). */
class HelperPool {
  constructor(helpers) {
    const maxN = MAX_NODES;
    this.helpers = helpers;
    this.maxN = maxN;
    this.ctrl = new Int32Array(new SharedArrayBuffer(4 * (CTRL.READY + helpers)));
    this.ack = new Int32Array(new SharedArrayBuffer(4 * helpers));
    // node terms of the optimizer (trajopt.js nodeTermEnv / nodeValueTerms): node states in, results out
    this.ybuf = sharedFloat64(NODE_STATE * maxN);
    this.prm = sharedFloat64(3);
    this.nodeOut = sharedFloat64(NODE_TERM * maxN);
    this.valOut = sharedFloat64(VALUE_TERMS * maxN);
    // reused from plan to plan
    this.gridBuffer = null; this.nodeBuffer = null; this.coefBuffer = null; this.hsBuffer = null;
    // the spline coefficients and the safe-height surface the shared copies hold, compared by identity: a plan's
    // coefficient and surface arrays are never changed after they are made
    this.coefSource = null;
    this.hsSource = null;
    this.seq = 0;
    this.broken = false;               // a helper did not answer in time: the helpers were ended
    this.workers = Array.from({ length: helpers }, (_, index) => {
      const w = new Worker(new URL("./helper_worker.js", import.meta.url), { type: "module" });
      w.postMessage({ init: true, ctrl: this.ctrl.buffer, ack: this.ack.buffer, index, helpers, ybuf: this.ybuf.buffer,
                      prm: this.prm.buffer, nodeOut: this.nodeOut.buffer, valOut: this.valOut.buffer });
      return w;
    });
  }

  /** Every helper has started (and none has failed to answer). */
  ready() {
    if (this.broken) return false;
    for (let i = 0; i < this.helpers; i++) if (Atomics.load(this.ctrl, CTRL.READY + i) !== 1) return false;
    return true;
  }

  /** Send this plan's data to every helper and wait until each has taken it. */
  send(data) {
    const { ack, helpers } = this;
    this.seq += 1;
    for (const w of this.workers) w.postMessage({ seq: this.seq, data });
    const t0 = performance.now();
    for (let i = 0; i < helpers; i++) {
      while (Atomics.load(ack, i) !== this.seq) {
        if (Atomics.load(ack, i) === -this.seq) {
          throw new DemoError(ErrorCode.PLANNER, "a helper thread could not take the plan data");
        }
        if (performance.now() - t0 > ACK_LIMIT_MS) {
          this.shutDown(`a helper thread did not take the plan data in ${ACK_LIMIT_MS / 1000} s`);
        }
        Atomics.wait(ack, i, Atomics.load(ack, i), WAIT_SLICE_MS);
      }
    }
  }

  /** Run task `task` on the helpers (shares 1..P-1) while this thread does share 0 (mine(P)). */
  runTask(task, mine) {
    const { ctrl, helpers } = this;
    const P = helpers + 1;
    Atomics.store(ctrl, CTRL.PARTICIPANTS, P);
    Atomics.store(ctrl, CTRL.DONE, 0); Atomics.store(ctrl, CTRL.ERROR, 0); Atomics.store(ctrl, CTRL.TASK, task);
    Atomics.add(ctrl, CTRL.GENERATION, 1);
    Atomics.notify(ctrl, CTRL.GENERATION);
    mine(P);
    const t0 = performance.now();
    for (;;) {
      const done = Atomics.load(ctrl, CTRL.DONE);
      if (done >= helpers) break;
      if (performance.now() - t0 > TASK_LIMIT_MS) {
        this.shutDown(`helper threads did not finish in ${TASK_LIMIT_MS / 1000} s`);
      }
      Atomics.wait(ctrl, CTRL.DONE, done, TASK_WAIT_SLICE_MS);
    }
    if (Atomics.load(ctrl, CTRL.ERROR)) throw new DemoError(ErrorCode.PLANNER, "a helper thread failed");
  }

  /** A helper did not answer: end every helper (ready() is false from now on) and throw `message`. */
  shutDown(message) {
    for (const w of this.workers) w.terminate();
    this.broken = true;
    throw new DemoError(ErrorCode.PLANNER, message);
  }

  /** The planner's wind grid (wind_grid.js), rows shared with the helpers; the same values as windGrid. */
  windGrid(geom, wind) {
    const shape = windGridShape(geom, wind), [n0, n1, n2] = shape.dims;
    // one buffer per planner, reused (the jobs run one after another; every row is written below)
    this.gridBuffer = atLeast(this.gridBuffer, 4 * n0 * n1 * n2 * 3);
    const data = new Float32Array(this.gridBuffer, 0, n0 * n1 * n2 * 3);
    this.send({ kind: "grid", geom, wind, data: data.buffer, length: data.length });
    this.runTask(TASK.GRID, (P) => windGridRows(shape, data, ...shareOf(0, P, n0)));
    return gridResult(shape, data);
  }

  /**
   * A float32 array of `length` in the shared coefficient buffer: a spline built in it is not copied again below.
   * Zeroed, as a new array is (the spline builder leaves the cells outside its region unwritten, and an earlier
   * plan's values there would otherwise reach this plan).
   */
  sharedCoef(length) {
    this.coefBuffer = atLeast(this.coefBuffer, 4 * length);
    this.coefSource = new Float32Array(this.coefBuffer, 0, length).fill(0);
    return this.coefSource;
  }

  /** The spline coefficients in the shared buffer (copied there unless they were built there). */
  shareCoefficients(coef) {
    if (coef.buffer === this.coefBuffer) return coef;
    this.coefBuffer = atLeast(this.coefBuffer, 4 * coef.length);
    const copy = new Float32Array(this.coefBuffer, 0, coef.length);
    copy.set(coef);
    return copy;
  }

  /** Every lattice node's wind (lattice.js latticeNodeWinds), ids shared with the helpers; wind: this thread's. */
  latticeWinds(nodeGrid, spline, wind) {
    const N = nodeGrid.I * nodeGrid.J * nodeGrid.K;
    this.nodeBuffer = atLeast(this.nodeBuffer, 4 * N * 3);
    const nodeWind = new Float32Array(this.nodeBuffer, 0, N * 3);      // every node is written below
    const coef = this.shareCoefficients(spline.coef);
    this.coefSource = spline.coef;     // optStart reuses this copy for the same spline
    this.send({ kind: "lattice", nodeGrid, spline: { ...spline, coef: coef.buffer, coefLength: coef.length },
                nodeWind: nodeWind.buffer, nodeLength: nodeWind.length });
    this.runTask(TASK.LATTICE, (P) => latticeNodeWinds(nodeGrid, wind, nodeWind, ...shareOf(0, P, N)));
    return nodeWind;
  }

  /** The optimizer's fields for the node tasks (plan_fields.js): the spline, the safe-height surface, the edges. */
  optStart({ spline, hs, planes, band, domain, P }) {
    if (this.coefSource !== spline.coef || !this.coefBuffer) {
      this.shareCoefficients(spline.coef);
      this.coefSource = spline.coef;
    }
    if (this.hsSource !== hs.coef) {
      this.hsBuffer = atLeast(this.hsBuffer, 8 * hs.coef.length);
      new Float64Array(this.hsBuffer, 0, hs.coef.length).set(hs.coef);
      this.hsSource = hs.coef;
    }
    this.send({ kind: "opt", spline: { ...spline, coef: this.coefBuffer, coefLength: spline.coef.length },
                hs: { ...hs, coef: this.hsBuffer, coefLength: hs.coef.length }, planes, band, domain,
                P: Float64Array.from(P) });
  }

  /** Task NODE_TERMS or NODE_VALUES for n nodes at (mu, wt, dt): the helpers shares 1..P-1, mine(P) share 0. */
  optRun(task, n, mu, wt, dtv, mine) {
    const { prm, ctrl } = this;
    prm[0] = mu; prm[1] = wt; prm[2] = dtv;
    Atomics.store(ctrl, CTRL.N, n);
    this.runTask(task, mine);
  }
}
