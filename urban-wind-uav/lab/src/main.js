// GeoWind2Plan demo: mission-time 3D urban wind prediction (WebGPU) and energy-efficient UAV route planning, all in
// the browser.  This entry module starts the page and wires its parts (src/app/).
import { hasCode, ErrorCode } from "./shared/errors.js";
import { createViewer } from "./app/scene.js";
import { initFolds, showPanel, endResultsReserve, fitPanel } from "./app/folds.js";
import { trackLayout } from "./app/layout.js";
import { drawLegend } from "./app/legend.js";
import { status, fatal, PREPARING } from "./app/status.js";
import { showSkeleton, resultsSummary } from "./app/results.js";
import { drawChart } from "./app/charts.js";
import { startPlanners, configurePlanners, stopPlanners } from "./app/planning.js";
import { prepareModel } from "./app/gpu.js";
import { useCity } from "./app/cities.js";
import { wireViewer, wireControls } from "./app/controls.js";
import { directionDegrees, redrawDial } from "./app/wind_controls.js";
import { setupIntroduction } from "./app/intro.js";
import { state } from "./app/state.js";
import { DEVICE_LIMIT_MS } from "./app/config.js";

trackLayout();                        // first: the panel's place (beside or over the scene), read by the modules below
setupIntroduction(() => fitPanel());        // the panel's content back: fitted to the window as it is now

/** Start the page: the planners, the scene and the panel, the model and the city list, then the first city. */
async function start() {
  // the planner workers first, so their energy tables are built while the GPU, the model and the city load
  startPlanners();
  showSkeleton();
  status(PREPARING, "loading");
  createViewer();
  wireViewer();
  initFolds({
    redraw: { dial: redrawDial, "legend-bar": () => drawLegend(state.range), chart: () => drawChart() },
    directionDegrees,
    resultsSummary,
  });
  const manifest = await prepareModel();
  if (!manifest) {                    // no WebGPU: the page says why
    cannotRun();
    return;
  }
  configurePlanners();
  wireControls();
  showPanel();                        // in its final layout: wired, its sections folded for the screen
  // the first mission: the first urban block's own (config.js FIRST_MISSIONS)
  await useCity(manifest.cities[0]);
}

/** The page cannot run: its planner threads end, and the panel shows as it is (no results will come). */
function cannotRun() {
  stopPlanners();
  endResultsReserve();
  showPanel();
}

start().catch((error) => {
  console.error(error);
  cannotRun();
  if (hasCode(error, ErrorCode.MODEL)) {    // the model files do not fit this page: a reload would not help
    fatal(error.message);
    return;
  }
  // a file that did not load may load on a reload; the graphics card not answering (the only time limit before the
  // first city: app/gpu.js) is said in words; other errors are reported as they are
  const hint = hasCode(error, ErrorCode.DOWNLOAD) ? "Please check the connection and reload the page."
    : "Reload the page to try again.";
  const text = hasCode(error, ErrorCode.TIMEOUT) ? `The graphics card did not answer in ${DEVICE_LIMIT_MS / 1000} s.`
    : error.message;
  fatal(`${text}\n\n${hint}`, true);
});
