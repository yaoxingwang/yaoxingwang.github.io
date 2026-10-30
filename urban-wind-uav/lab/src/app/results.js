// The Results section: the skeleton while computing, the failure card, the paper's two numbers, the chart tabs, and
// the routes in the scene with the route selection.
import { CLEARANCE_M } from "../planner/clearance.js";
import { CHART_TABS } from "./config.js";
import { state, routedMethods } from "./state.js";
import { $, el, announce, unannounce } from "./dom.js";
import { viewer } from "./scene.js";
import { status } from "./status.js";
import { applyDisplay } from "./wind_display.js";
import { fitPanel, endResultsReserve } from "./folds.js";
import { showEnergyChart, routeProfile, drawChartLegend, drawChart, markShownRoutes } from "./charts.js";

// The failure cards: [title, text, button] by kind.  "route" = no safe predicted-wind route; "wind", "planning",
// "cfd" = a computation stopped at that stage; "gpu" = the graphics device could not be made anew; "loading",
// "gpuhang", "gpulost", "toolarge", "open" = a city could not be opened (app/cities.js); "mission" = no first mission
// in the open city; "startup" = the planner threads did not start (a city load or a computation found it).
const FAILURES = {
  route: ["No safe route for this mission",
    `The planner found no route in the corridor that stays ${CLEARANCE_M} m from every building.`,
    "Try another mission"],
  wind: ["The wind could not be predicted", "The graphics card stopped during the wind prediction.", "Try again"],
  planning: ["The routes could not be planned", "The route planner stopped with an error.", "Try another mission"],
  loading: ["The data could not be loaded", "A file for this city or its CFD wind field did not load.", "Try again"],
  gpu: ["The graphics card stopped",
    "The browser lost the graphics card and could not get it back. Reload the page to continue.", "Reload the page"],
  cfd: ["The ground-truth wind did not load",
    "The CFD wind field of this setting could not be downloaded. Try again, or switch Ground truth Off to plan "
    + "without it.",
    "Try again"],
  gpuhang: ["The graphics card did not answer",
    "The city's geometry was not computed in time. Try again; if it happens again, reload the page.", "Try again"],
  startup: ["The route planner did not start", "Its background threads did not answer. Reload the page to try again.",
    "Reload the page"],
  gpulost: ["The graphics card stopped",
    "The graphics card stopped while this urban block was being prepared. Try again; if it happens again, reload the "
    + "page.",
    "Try again"],
  toolarge: ["This urban block is too large",
    "It needs more graphics memory than the demo uses. Choose another urban block.",
    "Choose another urban block"],
  open: ["The urban block could not be opened",
    "Something went wrong while it was being prepared. Try again; if it happens again, reload the page.",
    "Try again"],
  mission: ["No mission could be drawn",
    "No start and goal clear of buildings were found at this height in this urban block. Change the start and goal "
    + "height if you like, then try again.",
    "Try again"],
};

// %: half the shown step (one decimal): an increase smaller than this shows as 0.0 % less, not more
const HALF_SHOWN_STEP = 0.05;
const NO_SAVING_WH = 1e-9;    // Wh/km: a Ground truth saving below this is none (no recovery share)
let retryAction = null;
let shadersWarmed = false;           // viewer.warmShaders has been asked for (after the first results)

/** What "Try again" on the failure card does (null: a new random mission). */
export function setRetry(action) { retryAction = action; }
/** The retry action, taken (it belongs to the failure it was made for). */
export function takeRetry() {
  const action = retryAction;
  retryAction = null;
  return action;
}

/** From the moment a mission starts until its routes are known: the progress line only. */
export function showSkeleton() {
  endOpening({ newTask: true });       // a new task meanwhile (a pin dragged, a new mission or city): see endOpening
  state.failureShown = false;
  leaveFlight();
  $("fly").hidden = true;
  $("headline").hidden = false;
  $("headline-main").hidden = true;
  hideFailure();
  unannounce("results");              // the last results are no longer shown
  $("routes-block").hidden = true;
  $("energy-chart").hidden = true;
  $("profile-block").hidden = true;
  $("timing").textContent = "";
}

/** The failure card goes; the keyboard focus on its button moves to the Results title (not to the page's start). */
function hideFailure() {
  const card = $("failure");
  if (card.contains(document.activeElement)) document.querySelector("#results .fold-head").focus();
  card.hidden = true;
  unannounce("failure");
}

/** The mission failed (FAILURES kinds): the failure card instead of the results; the routes are cleared. */
export function showFailure(kind) {
  state.failureShown = true;
  leaveFlight();
  $("fly").hidden = true;
  viewer.clearRoutes();
  // the wind itself failed: its particles and maps are not trustworthy either
  if (kind === "wind") viewer.clearWind();
  status(null);
  $("headline").hidden = true;
  $("routes-block").hidden = true;
  $("failure").hidden = false;
  const [title, body, button] = FAILURES[kind];
  $("failure-title").textContent = title;
  $("failure-body").textContent = body;
  $("failure-retry").textContent = button;
  announce(`${title}. ${body}`, "failure");       // for a screen reader: the card replaced the results
  endResultsReserve();
  applyDisplay();                                  // the rows a failure hides go, then the panel's fit (applyDisplay)
}

/** The wind the chart quantities are taken in: the CFD wind with Ground truth on (energies are scored there). */
const profileWindAt = () => (state.results?.scoredIn === "cfd" ? state.winds.cfd : state.winds.ours);

/** The line charts' quantities along every route (charts.js routeProfile) in the wind the routes are scored in. */
function computeProfiles() {
  const { plans } = state.results;
  const wind = profileWindAt();
  state.profiles = {};
  for (const m of routedMethods(plans)) state.profiles[m.id] = routeProfile(plans[m.id], wind);
}

/** The chart tabs: the energy bars or one line chart along the route. */
export function applyTab() {
  const evaluation = state.results?.scoredIn === "cfd";
  const tabs = [...$("chart-tabs").children];
  // Ground truth off (no true wind): nothing that needs a wind to compute - no energy, no airspeed
  const needsTrueWind = (tab) => tab === "energy" || CHART_TABS[tab]?.wind;
  for (const b of tabs) b.hidden = !evaluation && needsTrueWind(b.dataset.tab);
  // one column per shown tab, so two tabs fill the bar as four do: all four sized by their text, fewer in equal parts
  $("chart-tabs").classList.toggle("fit", tabs.every((b) => !b.hidden));
  // a tab that needs the ground truth is shown again when Ground truth is switched back on
  if (!evaluation && needsTrueWind(state.tab)) {
    state.evaluationTab = state.tab;
    state.tab = "height";
  } else if (evaluation && state.evaluationTab) {
    state.tab = state.evaluationTab;
    state.evaluationTab = null;
  }
  const tab = state.tab;
  for (const b of tabs) b.setAttribute("aria-pressed", String(b.dataset.tab === tab));
  if (!state.results) return;
  const energyView = tab === "energy";
  // Ground truth on: the energy bars and the line chart keep their room in one place, the other tab's invisible
  // (style.css .inactive), so a tab change never changes Results' height
  $("energy-chart").hidden = !evaluation;
  $("energy-chart").classList.toggle("inactive", !energyView);
  $("profile-block").hidden = false;
  $("profile-block").classList.toggle("inactive", energyView);
  drawChartLegend(state.results.plans, selectRoute, toggleRoute);
  if (!energyView) drawChart();
  fitPanel();
}

/**
 * The computation time in the section title: wind prediction and planning, or planning only (a speed change); in a
 * very narrow window the time alone (style.css .timing-what).
 */
function fillTiming() {
  const t = state.results.timing;
  const seconds = (ms, digits) => (ms / 1000).toFixed(digits);
  const what = t.predictMs > 0 ? "Predicted and planned" : "Planned";
  $("timing").replaceChildren(el("span", { className: "timing-what", textContent: `${what} in ` }),
                              `${seconds(t.wallMs, 1)} s`);
  $("timing").title = `from the click to the last route; of which wind prediction ${seconds(t.predictMs, 2)} s`;
}

/** Route `id` shown on top (a click on its energy bar or its chart legend item). */
function selectRoute(id) {
  if (state.selected === id) return;
  endOpening();
  state.selected = id;
  showRoutes({ shown: true });
}

/** The routes of `plans` as the scene draws them. */
const sceneRoutes = (plans) => routedMethods(plans).map((m) => ({
  id: m.id, colour: m.colour, points: plans[m.id].points, speeds: plans[m.id].speeds }));

/**
 * Route `id` drawn or not (its check box in the energy bars or the chart legend); the selection stays.  Hiding the
 * selected route stops the drone and the Fly button waits until it is drawn again.
 */
function toggleRoute(id) {
  endOpening({ allRoutes: false });              // the visitor's own choice of routes: kept as it is
  if (!state.hiddenRoutes.delete(id)) state.hiddenRoutes.add(id);
  markShownRoutes();
  viewer.showRoutes(sceneRoutes(state.results.plans), state.selected, state.hiddenRoutes);
  if (id === state.selected) applyFlight(false);
  if (state.tab !== "energy") drawChart();
}

let headline = null;          // the headline as shown: { more, value, recovery } texts (resultsSummary)
let openingFlight = false;    // the page's first flight, followed by the view: it ends the opening at the goal
let routesOwed = false;       // the opening ended by a new task: its results show every route (showRoutes)

/**
 * The results of the last computation: the two numbers, the chart and the routes in the scene; shown: they are on
 * screen already (another route chosen among them), else they are new.
 */
export function showRoutes({ shown = false } = {}) {
  if (state.failureShown) {
    state.failureShown = false;
    applyDisplay();
  }
  const { plans, scoredIn } = state.results;
  const evaluation = scoredIn === "cfd";
  // the selected route is kept; where it is not in these results (Ground truth with Ground truth off, a route not
  // found), Ours, or else the first route there is
  if (!plans[state.selected]?.points) {
    state.selected = plans.ours?.points ? "ours" : routedMethods(plans)[0]?.id ?? null;
  }
  // energies in Wh/km of the wind-agnostic, predicted-wind and ground-truth routes
  const A = plans.agnostic?.whPerKm, O = plans.ours?.whPerKm, G = plans.cfd?.whPerKm;
  const saving = (plan) => (Number.isFinite(plan?.whPerKm) && Number.isFinite(A) ? (1 - plan.whPerKm / A) * 100 : null);
  status(null);
  hideFailure();
  $("routes-block").hidden = false;
  // Ground truth off: no headline (no true wind, so no energy)
  $("headline").hidden = !evaluation;
  $("headline-main").hidden = false;
  // the headline is always about the predicted-wind route, whichever route is selected
  if (evaluation) {
    const value = saving(plans.ours);
    const more = value !== null && value < -HALF_SHOWN_STEP;
    // the paper's two numbers (Section 3.2), for this mission: the reduction relative to wind-agnostic planning
    // (A - O) / A, and the share of the Ground truth reduction recovered (A - O) / (A - G), energies in Wh/km
    $("headline-value").textContent = value === null ? "–" : `${Math.abs(value).toFixed(1)} %`;
    $("headline-label").textContent = more ? "more energy than\nWind‑agnostic" : "less energy than\nWind‑agnostic";
    const gtReduction = A - G;
    const recovery = [A, O, G].every(Number.isFinite) && gtReduction > NO_SAVING_WH
      ? (A - O) / gtReduction * 100 : null;
    $("recovery-stat").hidden = !plans.cfd?.points;
    // one decimal, as the energy reduction and as the paper ("6.9% ... recovering 87.9%")
    $("recovery-value").textContent = recovery === null ? "–" : `${recovery.toFixed(1)} %`;
    $("recovery-label").textContent = recovery === null ? "Ground truth\nsaves nothing"
      : "of Ground truth's\nsaving recovered";
    // shown in grey, not emphasized, when Ours saves nothing against Wind-agnostic or recovers a negative share
    $("headline-value").classList.toggle("no-saving", more);
    $("recovery-value").classList.toggle("no-saving", recovery !== null && recovery < 0);
    headline = { more, value: $("headline-value").textContent, recovery: $("recovery-value").textContent };
    showEnergyChart(plans, selectRoute, toggleRoute);
  } else {
    $("energy-chart").hidden = true;
  }
  fillTiming();
  if (routesOwed) {             // the opening ended by a new task: its results show every route
    routesOwed = false;
    state.hiddenRoutes.clear();
    markShownRoutes();
  }
  viewer.showRoutes(sceneRoutes(plans), state.selected, state.hiddenRoutes);
  // the shader programs a first slice or flight would compile in its frame: compiled once the page is idle
  if (!shadersWarmed) {
    shadersWarmed = true;
    (window.requestIdleCallback ?? setTimeout)(() => viewer.warmShaders());
  }
  if (!state.profiles || state.profilesFor !== state.results) {
    computeProfiles();
    state.profilesFor = state.results;
  }
  placeFlyButton(evaluation);
  applyFlight(shown);  // flying: again from the start, along the route now selected
  // the page's first results: the view follows the drone's flight, which ends the opening at the goal (endOpening);
  // anything else shown here (another route chosen, new results) is the visitor's: the opening is over
  openingFlight = endResultsReserve() && state.flying;
  if (openingFlight) {
    viewer.onLap = () => endOpening();
    viewer.onUnfollow = showEveryRoute;   // following stopped during the opening flight: every route shown at once
    viewer.announceDrone();      // a few seconds into the flight: what a click on the drone does
  } else {
    viewer.showDragHint();
  }
  applyTab();          // the chart tab decides what is shown below; the panel fitted to it before it is drawn
}

/** The drone button: right of the strip of the two numbers (Ground truth on) or of the chart tabs and legend. */
function placeFlyButton(evaluation) {
  const slot = evaluation ? $("headline-row") : $("routes-block");
  if ($("fly").parentElement !== slot) slot.append($("fly"));
  $("fly").hidden = false;
}

/**
 * The drone flies along the selected route (from its start; the route shows behind it only), in the route's colour,
 * tilted by its thrust in the wind the routes are scored in (the line charts' quantities), or it is not in the scene;
 * the button says which.  shown: the route was on screen whole (the Fly button, another route chosen), so it fades
 * out first; a new route (new results) is never shown whole.
 */
function applyFlight(shown) {
  // the selected route hidden: no drone (the Fly button cannot start one)
  const hidden = state.hiddenRoutes.has(state.selected);
  if (hidden) state.flying = false;
  $("fly").disabled = hidden;
  $("fly").title = hidden ? "The selected route is hidden" : "";
  const plan = state.results?.plans[state.selected], profile = state.profiles?.[state.selected];
  if (state.flying && plan?.points && profile) {
    const { colour } = routedMethods(state.results.plans).find((m) => m.id === state.selected);
    const flight = { points: plan.points, speeds: plan.speeds, thrusts: profile.thrust, colour };
    viewer.flyDrone(state.selected, flight, shown);
  } else {
    viewer.stopDrone();
  }
  $("fly").setAttribute("aria-pressed", String(state.flying));
  $("fly").setAttribute("aria-label", state.flying ? "Stop the drone" : "Fly a drone along the selected route");
  $("fly-word").textContent = state.flying ? "Stop" : "Fly";
}

/** The opening's flight goes on, the view no longer on the drone: every route shown (the first flight shows all). */
function showEveryRoute() {
  if (!openingFlight) return;
  state.hiddenRoutes.clear();
  markShownRoutes();
  viewer.showRoutes(sceneRoutes(state.results.plans), state.selected, state.hiddenRoutes);
  if (state.tab !== "energy") drawChart();
}

/**
 * The opening ends - its flight reached the goal, or the visitor did anything but watch (Stop, a route chosen or its
 * check box): the page as it used to open - no flight, every route shown (allRoutes: not when the visitor has just set
 * a route's check box), the city framed; a new task meanwhile (newTask: a pin dragged, a new mission or city) only
 * ends it - the flight and the view go on as at any time, its results show every route; the drone's own hint is done,
 * the start pin's "Drag to move" comes.  Nothing outside the opening.
 */
function endOpening({ allRoutes = true, newTask = false } = {}) {
  if (!openingFlight) return;
  openingFlight = false;
  viewer.onLap = null;
  viewer.onUnfollow = null;
  viewer.endDroneAnnounce();
  viewer.showDragHint();
  if (newTask) {                 // a new task: the flight and the view go on as at any time; its results show all
    routesOwed = true;
    return;
  }
  state.flying = false;
  if (allRoutes) state.hiddenRoutes.clear();
  markShownRoutes();
  if (state.results) viewer.showRoutes(sceneRoutes(state.results.plans), state.selected, state.hiddenRoutes);
  applyFlight(false);
  viewer.resetView();
  if (state.tab !== "energy") drawChart();
}

/** The drone button: the drone starts flying (again and again; also through new computations) or stops. */
export function toggleFlight() {
  if (openingFlight) {           // Stop during the opening: the opening ends (as at its goal)
    endOpening();
    return;
  }
  state.flying = !state.flying;
  applyFlight(true);
}

/**
 * The drone leaves the scene while a computation runs or a failure shows (its route is gone); it keeps flying, on the
 * next results' selected route.  Nothing to do when it is not flying (also before the scene exists).
 */
function leaveFlight() {
  if (state.flying && viewer) viewer.hideDrone();     // no scene yet at start-up (Fly is on from the start)
}

/** The folded Results line: "Saves x % energy · y % recovered" while the headline shows, else "". */
export function resultsSummary() {
  if ($("headline-main").hidden || $("headline").hidden || !headline) return "";
  const effect = headline.more ? `Uses ${headline.value} more energy` : `Saves ${headline.value} energy`;
  return `${effect} · ${headline.recovery} recovered`;
}
