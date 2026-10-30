// A route planner worker: one per planning method (app/planning.js).  The planning is in plan_core.js; this shell
// handles the messages.
//   { cancelFlag }        SharedArrayBuffer (Int32 [0]): jobs with an id below its value are cancelled
//   { helperThreads }     the number of helper threads for this planner (0: none)
//   { buildTables }       move indices whose energy tables to build; answered { id: "tables-part", moves, na, nc, nz }
//   { setTables }         the joined tables of every move; answered { id: "tables-set" }
//   { id, geom, prepareOnly }  mission geometry, prepared before the wind is known (no answer)
//   { id, geom, wind }    geometry and wind descriptor (wind/samplers.js); prepares first unless the same id was
//                         prepared; answered { id, result }, { id, error } or { id, cancelled }
// Job ids are numbers; the start-up replies are named by the strings "tables-part" and "tables-set" (one reply
// matcher, app/planning.js replyOf, takes both).
import { ErrorCode, hasCode, errorText } from "../shared/errors.js";
import { epmTables } from "./lattice.js";
import { LATTICE_DS, LATTICE_DZ, prepare, plan, setTables, setPrepared, setCancelCheck } from "./plan_core.js";
import { createPool } from "./helper_pool.js";
import { windGrid } from "./wind_grid.js";

let cancelFlag = null;
let helpers = null;              // this planner's helper threads (helper_pool.js), or null: this thread computes all

self.onmessage = ({ data }) => {
  const { id, geom, wind, prepareOnly, buildTables, setTables: joined, helperThreads } = data;
  if (data.cancelFlag) {
    cancelFlag = new Int32Array(data.cancelFlag);
    setCancelCheck((job) => job < Atomics.load(cancelFlag, 0));
    return;
  }
  // jobs of superseded computations are skipped when they come up (and stopped between stages when they run)
  if (id !== undefined && cancelFlag && id < Atomics.load(cancelFlag, 0)) {
    if (!prepareOnly) self.postMessage({ id, cancelled: true });
    return;
  }
  if (helperThreads !== undefined) { helpers = createPool(helperThreads); return; }
  if (buildTables) {
    const part = epmTables(LATTICE_DS, LATTICE_DZ, buildTables);
    const moves = buildTables.map((m) => ({ m, t: part.tables[m].t, len: part.tables[m].len }));
    self.postMessage({ id: "tables-part", moves, na: part.na, nc: part.nc, nz: part.nz }, moves.map((x) => x.t.buffer));
    return;
  }
  if (joined) {
    setTables(joined);
    self.postMessage({ id: "tables-set" });
    return;
  }
  if (prepareOnly) {
    try {
      setPrepared(prepare(id, geom));
    } catch (error) {
      setPrepared(null);
      if (!hasCode(error, ErrorCode.CANCELLED)) console.error(error);
    }
    return;
  }
  try {
    // the helper threads, if started, for the whole plan (decided once: every step takes the same path); the wind
    // grid is sampled here, from the wind sent as plain data (started helpers share its rows: the same values)
    const pool = helpers?.ready() ? helpers : null;
    const grid = pool ? pool.windGrid(geom, wind) : windGrid(geom, wind);
    self.postMessage({ id, result: plan(id, { ...geom, grid }, pool) });
  } catch (error) {
    if (hasCode(error, ErrorCode.CANCELLED)) self.postMessage({ id, cancelled: true });
    else self.postMessage({ id, error: errorText(error) });
  }
};
