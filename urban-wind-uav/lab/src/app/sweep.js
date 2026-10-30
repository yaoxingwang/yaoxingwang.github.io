// The prediction sweep: while the wind is predicted, every SWEEP_EVERY-th predicted layer is shown as a horizontal
// slice (the Slice look) as soon as it is predicted, so a slice sweeps up through the corridor from the lowest layer
// to the top of the flight band; it is replaced by the chosen wind display when the prediction is done.  Its colour
// range is fixed before the prediction (it does not change while sweeping): 0 to SWEEP_RANGE_RATIO x the background
// speed.  The slices are prepared in a worker (view/sweep_worker.js).
import { layerCentreM } from "../shared/units.js";
import { state, windScale } from "./state.js";
import { viewer, shownCorridor } from "./scene.js";
import { CityWorker } from "./city_worker.js";

// the worker; a failed one is let go and the next slice starts a new one, at most MAX_RESTARTS times in a row (then
// the sweep stays blank, the results are not affected)
const sweeper = new CityWorker(new URL("../view/sweep_worker.js", import.meta.url), wire);
let roofsKey = 0, roofsLayout = null, open = false;
const pending = new Map();       // slice id -> { layout, stale }
let nextId = 0;

/** While the sweep is open its slices are drawn; a slice still in the worker when it closes is not drawn. */
export function setSweepOpen(on) { open = on; }

/** Close the sweep and take its slice away (also its colour range). */
export function endSweep() {
  open = false;
  state.sweepRange = null;
  viewer.clearSweep();
}

/** The sweep worker's replies: each slice is drawn while its prediction is still the current one. */
function wire(worker) {
  worker.onmessage = ({ data }) => {
    sweeper.answered();
    const request = pending.get(data.id);
    if (!request) return;
    pending.delete(data.id);
    if (data.error) console.warn("sweep slice failed:", data.error);        // only this slice is lost
    if (data.error || data.stale || request.stale() || !open || !state.sweepRange) return;
    viewer.setSweepSlice(data.texels, request.layout, layerCentreM(data.layer), state.sweepRange);
  };
  // a worker that did not start is replaced by the next slice (the slices waiting for it are not drawn); an error
  // outside a slice of a running worker (a slice's own error is its reply) loses nothing more; an error of a worker
  // already replaced changes nothing
  worker.onerror = (e) => {
    if (worker !== sweeper.worker) return;
    console.warn("sweep worker failed:", e.message || "it did not load");
    if (sweeper.hasAnswered) return;
    sweeper.endAfterFailure();
    roofsLayout = null;
    pending.clear();
  };
}

/** One predicted layer's weighted sums (gpu/predictor.js onLayer) as a sweep slice of `mission`'s corridor. */
export function drawSweep(layer, sums, layout, target, mission, stale) {
  const { gx, gy } = layout;
  if (!sweeper.worker && !sweeper.mayRestart) return;       // failed too often in a row: no sweep (results unaffected)
  const worker = sweeper.ensure(state.city);
  if (roofsLayout !== layout) {
    // once per prediction: the corridor in the wind frame and the grid's placement, for the mask and the roofs
    roofsLayout = layout;
    roofsKey += 1;
    worker.postMessage({ roofs: { key: roofsKey, windPoly: shownCorridor(mission).map(layout.toWind), gx, gy,
      gridX0: layout.gridX0, gridY0: layout.gridY0, theta: layout.theta } });
  }
  const id = ++nextId;
  pending.set(id, { layout, stale });
  const scale = windScale();
  worker.postMessage({ id, key: roofsKey, layer, sums, gx, gy, target, scale }, [sums.buffer]);
}
