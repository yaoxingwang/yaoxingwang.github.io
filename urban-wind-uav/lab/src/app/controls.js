// The panel's remaining controls and the map interaction: the city button, random missions, the start and goal height,
// dragging the start and goal, the wind display choices and its plane slider, the failure card's button, and the
// charts.
import { BAND_LOW_M, BAND_HIGH_M } from "../shared/units.js";
import { verticalPlaneCorners } from "../shared/geometry.js";
import { ErrorCode, hasCode } from "../shared/errors.js";
import { snapToValid } from "../planner/mission.js";
import { CHART_TABS } from "./config.js";
import { state, displayKind, atLeastBand, heightLabel } from "./state.js";
import { $, cssPx, showSliderValue } from "./dom.js";
import { viewer, shownCorridor, shownCorridorMetres } from "./scene.js";
import { setView, setDensity, applyDisplay, drawWind, updatePlaneControls, drawParticles, drawSlice, showWindDirection,
         verticalPlane } from "./wind_display.js";
import { applyTab, takeRetry, toggleFlight } from "./results.js";
import { drawChart, chartDistanceAt } from "./charts.js";
import { setMission, newRandomMission, computedMission } from "./pipeline.js";
import { useCity, cityLoads } from "./cities.js";
import { openPicker, setPickHandler } from "./picker.js";
import { introOpen, showIntroduction } from "./intro.js";
import { togglePanel, panelCollapsed, followLayout } from "./folds.js";
import { panelOverScene, freeArea, onLayoutChange } from "./layout.js";
import { wireWindControls, showHint, hideHint } from "./wind_controls.js";
import { drawLegend } from "./legend.js";

/**
 * The scene's free area (layout.js), where the camera centres the city: right of the panel, or the whole window while
 * the panel is collapsed or lies over the scene (its centre then the screen's from left to right; the bottom margin is
 * larger than the top one; style.css #stage the same for the cards over the scene).  eased: the view slides there.
 */
function sceneInsets(eased = false) {
  const area = freeArea();
  viewer.setInsets(area.left, area.right, area.top, area.bottom, eased);
  if (state.city) showWindDirection();
}

/** The panel's button (or Escape): the panel collapses or expands, its note goes, the view slides along. */
function usePanelButton() {
  togglePanel();
  hideHint();
  sceneInsets(true);
}

/**
 * The scene: the dial follows the camera; its free area follows the window, the layout and the panel (collapsed by
 * the visitor or with the layout: the panel's size); the panel's collapse button and Escape, wired from the start.
 */
export function wireViewer() {
  viewer.on("camera", () => { if (state.city) showWindDirection(); });
  const observer = new ResizeObserver(() => sceneInsets());
  observer.observe($("stage"));
  observer.observe($("panel"));
  // a layout change (also while the page loads): the panel follows it, then the view takes the new free area at once
  // (the window itself has just changed; a slide from the old free area would first move the city the wrong way)
  onLayoutChange(() => {
    followLayout();
    hideHint();                       // what it pointed at may have moved or gone
    sceneInsets();
  });
  sceneInsets();
  $("panel-toggle").addEventListener("click", usePanelButton);
  $("intro-mark").addEventListener("click", hideHint);
  // Escape closes the picker, else the introduction, else collapses a panel lying over the scene (the scene in view
  // again); not one a number field has used to take back what was typed (wind_controls.js)
  window.addEventListener("keydown", (event) => {
    if (event.key !== "Escape" || event.defaultPrevented) return;
    if (!$("city-picker").hidden) {
      openPicker(false);
    } else if (introOpen()) {
      showIntroduction(false);
      $("intro-mark").focus();
    } else if (panelOverScene() && !panelCollapsed()) {
      usePanelButton();
      $("panel-toggle").focus();
    }
  });
}

/** Every control of the panel and the start and goal drag (once the graphics card and the model are ready). */
export function wireControls() {
  // the city switch: the pill names the current city; it opens a picker with height-map previews
  setPickHandler((entry) => { beforePanelAction(); useCity(entry); });
  $("city-button").addEventListener("click", (event) => {
    event.stopPropagation();
    openPicker($("city-picker").hidden);
  });
  document.addEventListener("click", (event) => {
    if (!$("city-picker").contains(event.target) && !$("city-button").contains(event.target)) openPicker(false);
  });
  $("picker-close").addEventListener("click", () => openPicker(false));
  // the keyboard leaves the picker (Tab past its last tile): it closes (a click elsewhere: the listener above)
  $("city-picker").addEventListener("focusout", (event) => {
    const to = event.relatedTarget;
    if (to && !$("city-picker").contains(to) && to !== $("city-button")) openPicker(false);
  });
  wireWindControls();
  wireMission();
  wireDisplay();
  $("failure-retry").addEventListener("click", () => {
    beforePanelAction();
    const retry = takeRetry();
    if (retry) retry();
    else if (state.city) drawRandomMission();
    else if (state.manifest) useCity(state.cityEntry ?? state.manifest.cities[0]);
  });
  wireCharts();
  applyDisplay();                     // every row as the state says before the panel shows (main.js showPanel)
}

/** Before a panel action starts a new mission or city: the wind controls brought up to date, the map cursor reset. */
function beforePanelAction() {
  applyDisplay();
  viewer.setCursor("grab");
}

/** A new random mission of relative wind angle class `kind` (or any); when none is found the routes stay (a note). */
function drawRandomMission(kind = null) {
  try {
    newRandomMission({ kind });
  } catch (error) {
    console.error(error);
    // none found (MISSION): the routes shown stay, a note says so next to the buttons
    if (hasCode(error, ErrorCode.MISSION)) {
      showHint("No such mission was found at this height.\nTry again, or change the height.", $("mission-kind"));
    }
  }
}

/** Random missions, the start and goal height, and the start and goal drag. */
function wireMission() {
  // three buttons, each draws a new random mission of that relative wind angle class
  for (const b of $("mission-kind").children) {
    b.addEventListener("click", () => { beforePanelAction(); drawRandomMission(b.dataset.kind); });
  }
  // one altitude for start and goal: moving the slider moves both pins; on release both are snapped to usable places
  // at that height (CLEARANCE_M from buildings) and the routes are planned again
  const showAltitude = (z) => showSliderValue($("altitude-range"), $("altitude-value"), `${heightLabel(z)} m`);
  $("altitude-range").value = String(state.altitude);
  showAltitude(state.altitude);
  $("altitude-range").addEventListener("input", (event) => {
    const z = parseFloat(event.target.value);
    showAltitude(z);
    if (state.mission) {
      const at = (p) => [p[0], p[1], atLeastBand(z)];
      viewer.setPins(at(state.mission.start), at(state.mission.goal));
    }
  });
  $("altitude-range").addEventListener("change", (event) => {
    state.altitude = atLeastBand(parseFloat(event.target.value));
    if (!state.mission) return;
    const z = state.altitude, m = state.mission;
    const s = snapToValid(state.city, m.start[0], m.start[1], z) || m.start;
    const g = snapToValid(state.city, m.goal[0], m.goal[1], z, s) || m.goal;
    const moved = s[0] !== m.start[0] || s[1] !== m.start[1] || g[0] !== m.goal[0] || g[1] !== m.goal[1];
    setMission({ start: [s[0], s[1], z], goal: [g[0], g[1], z] }, { predict: moved });
  });
  // start and goal are dragged on the map: the pin always sits on the usable place nearest to the pointer (no error,
  // no jump back); there is no distance limit between start and goal
  // a drag belongs to the city load it started in: once a city is opened (its mission cleared, then new) it does
  // nothing more
  let dragOrigin = null, dragLoad = 0;
  const dragGone = () => !state.mission || cityLoads() !== dragLoad;
  viewer.on("pinDragStart", (which) => {
    if (!state.mission) return;
    dragOrigin = state.mission[which].slice();
    dragLoad = cityLoads();
    state.dragMission = state.mission;
  });
  viewer.on("pinDrag", (which, [x, y]) => {
    if (dragGone()) return;
    const other = state.mission[which === "start" ? "goal" : "start"];
    const p = snapToValid(state.city, x, y, state.altitude, other);
    if (!p) return;
    // a new object: a computation still running keeps the mission it started with
    state.mission = { ...state.mission, [which]: [p[0], p[1], state.altitude] };
    viewer.movePin(which, p[0], p[1]);
    viewer.setCorridor(shownCorridor(state.mission));
  });
  viewer.on("pinDragEnd", (which) => {
    if (dragGone()) {
      state.dragMission = null;
      return;
    }
    const m = state.mission[which];
    // back where it started, and no computation took a mission of the drag: the mission the results belong to is
    // the current one again (else the pins' mission is computed)
    const unchanged = state.dragMission && m[0] === dragOrigin[0] && m[1] === dragOrigin[1];
    if (unchanged && computedMission() === state.dragMission) {
      state.mission = state.dragMission;
      state.dragMission = null;
      return;
    }
    state.dragMission = null;
    setMission({ ...state.mission });
  });
}

/** The wind display: Particles (Full field / Horizontal / Vertical) | Slice (Horizontal / Vertical) | Off. */
function wireDisplay() {
  for (const button of $("view-mode").children) button.addEventListener("click", () => setView(button.dataset.view));
  for (const button of $("wind-source").children) {
    button.addEventListener("click", () => {
      if (state.sourceChoice === button.dataset.source) return;
      state.sourceChoice = button.dataset.source;
      applyDisplay();          // the buttons follow the click at once, also while the wind is computed
      drawWind();
    });
  }
  for (const button of $("plane-kind").children) {
    button.addEventListener("click", () => setView(state.view, button.dataset.plane));
  }
  // the particle density: its value follows a drag, the particles are made anew once it is let go (change)
  const density = $("density-range");
  const showDensity = () => showSliderValue(density, $("density-value"), `${+parseFloat(density.value).toFixed(2)}×`);
  density.addEventListener("input", showDensity);
  density.addEventListener("change", () => setDensity(parseFloat(density.value)));
  // while the particles' plane is moved, a transparent green plane shows where it is (the horizontal plane: the
  // corridor at that height; the vertical one: its flight band); gone as soon as the slider is let go
  // (the pointer released, or the arrow key)
  let sliderHeld = false;
  let framed = false;           // the frame asked for by the slider and not yet taken away by letting it go
  const showPlaneFrame = () => {
    if (displayKind() !== "particles2d" || !state.mission) return;
    let points;
    if (state.plane === "h") {
      points = shownCorridorMetres(state.mission).map(([x, y]) => [x, y, state.sliceHeight]);
    } else {
      const pl = verticalPlane();
      if (!(pl.s1 > pl.s0)) return;
      points = verticalPlaneCorners(pl.origin, pl.u, pl.s0, pl.s1, BAND_LOW_M, BAND_HIGH_M);
    }
    viewer.setPlaneFrame(points);
  };
  const hideFrame = () => {
    framed = false;
    viewer.setPlaneFrame(null);
  };
  const releaseSlider = () => {
    if (!sliderHeld) return;
    sliderHeld = false;
    hideFrame();
  };
  $("slice-range").addEventListener("pointerdown", () => { sliderHeld = true; showPlaneFrame(); });
  window.addEventListener("pointerup", releaseSlider);
  window.addEventListener("pointercancel", releaseSlider);
  $("slice-range").addEventListener("keyup", () => { if (!sliderHeld) hideFrame(); });
  // the plane's particles or slice, the legend and the frame at the slider's latest place, once per drawn frame
  // (viewer.beforeFrame: a drag gives more input events than frames); the slice made in the particle worker, so a drag
  // does not hold up the page
  const redrawPlane = () => {
    const kind = displayKind();
    if (kind === "particles2d") drawParticles();
    else if (kind === "slice") drawSlice({ inWorker: true });
    drawLegend(state.range);
    if (framed) showPlaneFrame();
  };
  $("slice-range").addEventListener("input", (event) => {
    if (state.plane === "h") state.sliceHeight = atLeastBand(parseFloat(event.target.value));
    else state.vOffset = parseFloat(event.target.value);
    updatePlaneControls();
    framed = true;
    viewer.beforeFrame("plane", redrawPlane);
  });
}

/**
 * The chart tabs (where they do not fit, in a window narrower than the panel, they scroll sideways, their right edge
 * fades while more tabs follow, and the chosen tab is brought into view), hovering the line chart: a vertical line and
 * every route's value at that distance, and the drone button.
 */
function wireCharts() {
  $("fly").addEventListener("click", toggleFlight);
  const tabs = $("chart-tabs");
  const fadeTabs = () => tabs.classList.toggle("more-right", tabs.scrollLeft + tabs.clientWidth < tabs.scrollWidth - 1);
  // the chosen tab scrolled into view, clear of the faded right edge (style.css --tab-fade)
  const showChosen = () => {
    const chosen = tabs.querySelector('[aria-pressed="true"]');
    if (chosen && tabs.scrollWidth > tabs.clientWidth) {
      const bar = tabs.getBoundingClientRect(), item = chosen.getBoundingClientRect();
      const right = bar.right - (tabs.classList.contains("more-right") ? cssPx("--tab-fade", tabs) : 0);
      if (item.left < bar.left) tabs.scrollLeft -= bar.left - item.left;
      else if (item.right > right) tabs.scrollLeft += item.right - right;
    }
    fadeTabs();
  };
  for (const b of tabs.children) {
    b.addEventListener("click", () => { state.tab = b.dataset.tab; applyTab(); showChosen(); });
  }
  tabs.addEventListener("scroll", fadeTabs);
  const resized = new ResizeObserver(showChosen);  // the bar's width, and tabs shown or hidden with the mode
  for (const element of [tabs, ...tabs.children]) resized.observe(element);
  // the read-out at the pointer: a mouse hovering, or a finger on the chart (until it lifts)
  const canvas = $("chart");
  const readOut = (e) => {
    if (state.profiles && CHART_TABS[state.tab]) drawChart(chartDistanceAt(e.clientX));
  };
  canvas.addEventListener("pointerdown", (e) => { if (e.pointerType === "touch") readOut(e); });
  canvas.addEventListener("pointermove", readOut);
  for (const type of ["pointerleave", "pointercancel"]) canvas.addEventListener(type, () => drawChart());
  canvas.addEventListener("pointerup", (e) => { if (e.pointerType === "touch") drawChart(); });
}
