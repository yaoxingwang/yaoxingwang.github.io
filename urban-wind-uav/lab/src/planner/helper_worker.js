// A helper thread of a planner worker (helper_pool.js): it takes each plan's data, then waits for a generation and
// computes its share of the task (pool_protocol.js TASK) and reports done.  The same code as the planner's own share,
// so the results are the same as one thread doing all.
import { makeNodeEnv, nodeTermEnv, nodeValueTerms } from "./trajopt.js";
import { makePlanFields, makeWindField } from "./plan_fields.js";
import { windGridShape, windGridRows } from "./wind_grid.js";
import { latticeNodeWinds } from "./lattice.js";
import { TASK, CTRL, NODE_TERM, VALUE_TERMS, TERM_HESSIAN, TERM_DT_CROSS, TERM_DT, shareOf,
         nodeStateOf } from "./pool_protocol.js";

let plan = null;                      // this plan's data (wind grid, lattice or optimizer fields), sent before its task
let ctrl, ack, index, ybuf, prm, nodeOut, valOut;

self.onmessage = ({ data }) => {
  if (data.seq) takePlanData(data);
  else if (data.init) start(data);
};

/** Take one plan's data; the ack (helper_pool.js) is the plan's sequence number, or minus it when this failed. */
function takePlanData({ seq, data }) {
  try {
    if (data.kind === "opt") {
      // the optimizer's fields, built as on the planner thread (plan_fields.js)
      const { spline, hs, planes, band, domain, P } = data;
      const fields = makePlanFields({ ...spline, coef: new Float32Array(spline.coef, 0, spline.coefLength) },
                                    { ...hs, coef: new Float64Array(hs.coef, 0, hs.coefLength) }, planes, band, domain);
      plan = { kind: "opt", env: makeNodeEnv(fields, P.length / 3, P) };
    } else if (data.kind === "grid") {
      const shape = windGridShape(data.geom, data.wind);
      plan = { kind: "grid", shape, data: new Float32Array(data.data, 0, data.length) };
    } else {
      // the lattice: the wind spline's field, as on the planner thread (plan_core.js)
      const { nodeGrid, spline } = data;
      const field = makeWindField({ ...spline, coef: new Float32Array(spline.coef, 0, spline.coefLength) });
      const wind = (p, out) => field(p, out, null);
      plan = { kind: "lattice", nodeGrid, wind, nodeWind: new Float32Array(data.nodeWind, 0, data.nodeLength) };
    }
  } catch (error) {
    console.error(error);
    plan = null;
    Atomics.store(ack, index, -seq);   // the planner thread fails this plan at once (the helpers stay)
    Atomics.notify(ack, index);
    return;
  }
  Atomics.store(ack, index, seq);
  Atomics.notify(ack, index);
}

/** This helper's share p of the current task (P participants). */
function runShare(task, p, P) {
  if (task === TASK.GRID) {
    const n0 = plan.shape.dims[0];
    windGridRows(plan.shape, plan.data, ...shareOf(p, P, n0));
  } else if (task === TASK.LATTICE) {
    const N = plan.nodeGrid.I * plan.nodeGrid.J * plan.nodeGrid.K;
    latticeNodeWinds(plan.nodeGrid, plan.wind, plan.nodeWind, ...shareOf(p, P, N));
  } else if (task === TASK.NODE_TERMS || task === TASK.NODE_VALUES) {
    const n = ctrl[CTRL.N], [lo, hi] = shareOf(p, P, n), env = plan.env;
    const mu = prm[0], wt = prm[1], dtv = prm[2];
    for (let k = lo; k < hi; k++) {
      const y = nodeStateOf(ybuf, k);
      if (task === TASK.NODE_TERMS) {
        const o = NODE_TERM * k;
        const gy = nodeOut.subarray(o, o + TERM_HESSIAN), Hy = nodeOut.subarray(o + TERM_HESSIAN, o + TERM_DT_CROSS);
        const Hdty = nodeOut.subarray(o + TERM_DT_CROSS, o + TERM_DT);
        nodeOut[o + TERM_DT] = nodeTermEnv(env, k, y, mu, wt, dtv, gy, Hy, Hdty);
      } else {
        nodeValueTerms(env, k, y, mu, wt, dtv, valOut, VALUE_TERMS * k);
      }
    }
  }
}

/** Take the shared buffers (init), report ready, then compute a share of every new generation's task. */
function start(init) {
  ({ index } = init);
  ctrl = new Int32Array(init.ctrl); ack = new Int32Array(init.ack);
  ybuf = new Float64Array(init.ybuf); prm = new Float64Array(init.prm);
  nodeOut = new Float64Array(init.nodeOut); valOut = new Float64Array(init.valOut);
  let last = Atomics.load(ctrl, CTRL.GENERATION);
  Atomics.store(ctrl, CTRL.READY + index, 1);
  // waits asynchronously, so the plan data messages are received between tasks
  (async () => {
    for (;;) {
      const r = Atomics.waitAsync(ctrl, CTRL.GENERATION, last);
      if (r.async) await r.value;
      last = Atomics.load(ctrl, CTRL.GENERATION);
      try {
        runShare(ctrl[CTRL.TASK], index + 1, ctrl[CTRL.PARTICIPANTS]);
      } catch (error) {
        Atomics.store(ctrl, CTRL.ERROR, 1);
        console.error(error);
      }
      Atomics.add(ctrl, CTRL.DONE, 1);
      Atomics.notify(ctrl, CTRL.DONE);
    }
  })();
}
