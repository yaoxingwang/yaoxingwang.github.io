// Foldable panel sections (Mission, Background wind, Wind visualization, Results).  A section's title folds it; a
// folded section shows its setting in one line.  When the panel is first shown, the open sections are computed from
// their heights: by OPENING_ORDER, each one that still fits opens (chooseSections).  From then on a section opens only
// when the visitor presses its title.  When the panel grows taller than the screen, sections fold in FOLD_ORDER until
// it fits (foldToFit): never the section the visitor opened or used last, and Results only when the visitor has just
// opened another section by its title; otherwise the panel scrolls.  The whole panel also collapses to its title
// (setPanelCollapsed); where it lies over the scene, or on a phone held sideways (layout.js opensCollapsed), the page
// opens with it collapsed (the open sections are then computed when it is first expanded).
import { state, heightLabel, cityShort } from "./state.js";
import { $, cssPx } from "./dom.js";
import { openPicker } from "./picker.js";
import { introOpen, showIntroduction } from "./intro.js";
import { opensCollapsed } from "./layout.js";

const FOLDS = ["mission", "wind", "vis", "results"];   // the sections, top to bottom
// the owner's priority: the panel's first fit opens each of these, in this order, if it still fits; later fits fold
// them the other way round (Wind visualization first, Results only as the last resort)
const OPENING_ORDER = ["results", "mission", "wind", "vis"];
const FOLD_ORDER = [...OPENING_ORDER].reverse();
const SUMMARY_INTERVAL_MS = 250;    // the folded lines follow every setting by checking it this often (a few string
                                    // compares) rather than every control reporting its changes
const COMPUTING = "Computing…";    // a folded line while results are on their way
const CONTENT_SETTLE_MS = 300;      // the panel is refitted once Results has stopped growing for this long
const RESIZE_SETTLE_MS = 150;       // a window resize folds only once the size has settled for this long
let lastUsed = null;                // the section the visitor opened or used last: never folded to make room
let sectionsChosen = false;         // the first fit has computed the open sections (chooseSections)
let resultsReserved = true;         // until the first results (endResultsReserve): fits keep the room they will take
let panelChosen = false;            // the visitor has collapsed or expanded the panel (togglePanel)
let pressing = false;               // a pointer is held in the panel (a dial or slider drag): fits wait for its release
let fitWaiting = false;             // a fit was asked for while pressing
let showing = false;                // showPanel has been called (it shows the panel once the fonts are in)
/** The section element of fold `name`. */
const foldOf = (name) => document.querySelector(`#panel .fold[data-fold="${name}"]`);

/** Open or fold section `name` (its head's aria-expanded follows). */
function setFold(name, open) {
  const fold = foldOf(name);
  fold.classList.toggle("collapsed", !open);
  fold.querySelector(".fold-head").setAttribute("aria-expanded", String(open));
}

/**
 * The panel appears, wired and fitted to the screen (collapsed where it lies over the scene or on a phone held
 * sideways: layout.js opensCollapsed; the opening view is the scene alone): at start-up (main.js; index.html starts
 * it hidden), once the fonts are in (the sections' heights are then final); a later call (a start-up error after it)
 * does nothing.  Shown and fitted in one task: no frame of it unfitted.
 */
export function showPanel() {
  if (!booting() || showing) return;
  showing = true;
  document.fonts.ready.then(() => {
    if (opensCollapsed()) setPanelCollapsed(true);
    $("panel").classList.remove("booting");
    fitPanel();
  });
}

/**
 * The first results (routes or a failure card) are shown, or will never come (the page cannot run): from now on the
 * room kept for Results while it waits is the height it last had (results.js, main.js).  Returns whether this ended
 * the first wait.
 */
export function endResultsReserve() {
  const ended = resultsReserved;
  resultsReserved = false;
  return ended;
}

/** Whether the panel is still hidden while the page wires it (index.html, showPanel). */
const booting = () => $("panel").classList.contains("booting");

/**
 * The layout has changed (controls.js wireViewer, on layout.js's change): until the visitor has chosen the panel's
 * state, it follows the opening rule.
 */
export function followLayout() {
  const collapse = opensCollapsed();
  if (panelChosen || booting() || collapse === panelCollapsed()) return;
  setPanelCollapsed(collapse);
}

/** The visitor's panel button (or Escape): collapse or expand the panel (its state is theirs from now on). */
export function togglePanel() {
  panelChosen = true;
  setPanelCollapsed(!panelCollapsed());
}

/** Whether the panel is collapsed to its title (setPanelCollapsed). */
export const panelCollapsed = () => $("panel").classList.contains("collapsed");

/**
 * Collapse the whole panel to its title and the expand button, or expand it again: its sections keep their folds,
 * and the expanded panel is fitted to the screen again.  Collapsing closes the picker and the introduction, and the
 * keyboard focus inside the panel or the picker moves to the panel's button (the one control left).
 */
function setPanelCollapsed(collapsed) {
  if (collapsed) {
    const focused = document.activeElement;
    const focusInside = $("panel").contains(focused) || $("city-picker").contains(focused);
    openPicker(false);
    showIntroduction(false);
    if (focusInside) $("panel-toggle").focus();
  }
  $("panel").classList.toggle("collapsed", collapsed);
  const toggle = $("panel-toggle"), label = collapsed ? "Expand the panel" : "Collapse the panel";
  toggle.setAttribute("aria-expanded", String(!collapsed));
  toggle.setAttribute("aria-label", label);
  toggle.title = label;
  if (!collapsed) fitPanel();
}

/**
 * Fit the panel to the screen.  The first fit with its content shown computes the open sections (chooseSections);
 * every later one only folds (foldToFit), never opens a section.  While a pointer is held in the panel it waits for
 * the release.
 */
export function fitPanel() {
  // not wired yet; the introduction replaces the content; a collapsed panel shows none (fitted when it is expanded)
  if (booting() || introOpen() || panelCollapsed()) return;
  if (pressing) {
    fitWaiting = true;
    return;
  }
  if (sectionsChosen) foldToFit(false);
  else chooseSections();
}

/**
 * Run `measure` with Results given the room it will take while it waits for results (#panel.results-reserved: before
 * the first ones --results-h, later the height it last had with routes, --results-room); with routes shown, that
 * height is noted for the next wait.
 */
function withResultsRoom(measure) {
  const panel = $("panel");
  // waiting: neither routes nor a failure card in Results (results.js showSkeleton), or the first results not shown
  const waiting = resultsReserved || ($("routes-block").hidden && $("failure").hidden);
  if (!waiting && !$("routes-block").hidden && !foldOf("results").classList.contains("collapsed")) {
    panel.style.setProperty("--results-room", `${foldOf("results").getBoundingClientRect().height}px`);
  }
  if (waiting) panel.classList.add("results-reserved");
  measure();
  panel.classList.remove("results-reserved");
}

/** The height the panel's content may take: the stage less its margins, less the panel's own border. */
function panelRoom() {
  const panel = $("panel");
  return $("stage").clientHeight - 2 * cssPx("--gutter") - (panel.offsetHeight - panel.clientHeight);
}

/** Whether the panel's content fits it (a 1 px tolerance for rounding). */
const fits = () => $("panel").scrollHeight <= $("panel").clientHeight + 1;

/**
 * The open sections, computed once from their heights (all sections are open until then): with every section folded
 * to its title row, each one in OPENING_ORDER opens if the panel then still fits (one that does not is skipped and the
 * next tried), Results always.  The folds are set once; a rounding miss is folded away (foldToFit).
 */
function chooseSections() {
  sectionsChosen = true;
  const height = (e) => e.getBoundingClientRect().height;
  withResultsRoom(() => {
    const extra = {};               // each section's open height less its title row
    for (const name of FOLDS) extra[name] = height(foldOf(name)) - height(foldOf(name).querySelector(".fold-heading"));
    const room = panelRoom();
    let total = $("panel").scrollHeight - FOLDS.reduce((sum, name) => sum + extra[name], 0);
    for (const name of OPENING_ORDER) {
      const open = name === "results" || total + extra[name] <= room;
      if (open) total += extra[name];
      setFold(name, open);
    }
  });
  foldToFit(false);
}

/**
 * Fold sections in FOLD_ORDER until the panel fits, never the one the visitor opened or used last; Results only when
 * the visitor has just opened another section by its title (byTitle).  What still does not fit scrolls.
 */
function foldToFit(byTitle) {
  withResultsRoom(() => {
    for (const name of FOLD_ORDER) {
      if (fits()) break;
      if (name === lastUsed || foldOf(name).classList.contains("collapsed")) continue;
      if (name === "results" && !byTitle) continue;
      setFold(name, false);
    }
  });
  markScroll();
}

/** The panel scrolls while its content does not fit (#panel.scrolls), its bottom edge faded while more is below. */
function markScroll() {
  $("panel").classList.toggle("scrolls", !fits());
  markMoreBelow();
}

/** Whether more of a scrolling panel is below what is shown (style.css #panel.more-below: its bottom edge fades). */
function markMoreBelow() {
  const panel = $("panel");
  panel.classList.toggle("more-below", panel.scrollTop + panel.clientHeight < panel.scrollHeight - 1);
}

/**
 * The one-line settings of the folded sections, and the collapsed panel's results line (Ground truth on; a failure
 * card's title in either mode, in the warning colour).
 */
function updateSummaries(directionDegrees, resultsSummary) {
  const failed = state.failureShown && !state.busy;
  const horizontal = `Horizontal ${heightLabel(state.sliceHeight)} m`;
  const particleType = { "3d": "Full field", h: horizontal, v: "Vertical" }[state.particlePlane];
  const text = {
    "sum-mission": `${cityShort()} · ${heightLabel(state.altitude)} m`,
    "sum-wind": `${directionDegrees()}° · ${state.speed} m/s`,
    "sum-vis": state.view === "none" ? "Off"
      : state.view === "particles" ? `Particles · ${particleType}`
      : `Slice · ${state.slicePlane === "h" ? horizontal : "Vertical"}`,
    "sum-results": state.busy ? COMPUTING : failed ? $("failure-title").textContent : resultsSummary(),
  };
  // the collapsed panel's line: with Ground truth on, while results are on their way the loading card's words when it
  // shows (one status on screen), else "Computing…" (the first ones, a city loading; a computation's own line says so
  // already); nothing when none will come (endResultsReserve)
  const coming = resultsReserved || state.cityLoading;
  const card = $("loading-card").classList.contains("show") ? $("loading-text").textContent : "";
  const evaluationLine = text["sum-results"] || (coming ? card || COMPUTING : "");
  text["sum-panel"] = failed ? text["sum-results"] : state.mode === "evaluation" ? evaluationLine : "";
  for (const [id, t] of Object.entries(text)) if ($(id).textContent !== t) $(id).textContent = t;
  for (const id of ["sum-results", "sum-panel"]) $(id).classList.toggle("failed", failed);
}

/** Whether a text field of the panel has the focus on a touch screen (its keyboard may shorten the window). */
const typingOnTouch = () => matchMedia("(pointer: coarse)").matches && $("panel").contains(document.activeElement)
  && document.activeElement.matches("input:not([type=range]), textarea");

/**
 * Wire the section titles and the folding.  redraw: canvas id -> function that redraws it (a canvas in a folded
 * section has no size and loses what was drawn meanwhile); directionDegrees(): the direction shown (whole degrees,
 * -180..180); resultsSummary(): the Results line ("" when there is none).
 */
export function initFolds({ redraw, directionDegrees, resultsSummary }) {
  const panel = $("panel");
  panel.addEventListener("scroll", markMoreBelow, { passive: true });
  for (const name of FOLDS) {
    foldOf(name).querySelector(".fold-head").addEventListener("click", () => {
      const open = foldOf(name).classList.contains("collapsed");
      setFold(name, open);
      if (open) {
        lastUsed = name;
        foldToFit(true);
      } else {
        if (lastUsed === name) lastUsed = null;
        markScroll();
      }
    });
  }
  // a press or a key inside a section's content marks it as the one in use (kept open); a held pointer defers fits
  const markUsed = (event) => {
    const fold = event.target.closest?.(".fold");
    if (fold && !event.target.closest(".fold-head")) lastUsed = fold.dataset.fold;
  };
  panel.addEventListener("pointerdown", (event) => {
    markUsed(event);
    pressing = true;
  });
  panel.addEventListener("keydown", markUsed);
  const release = () => {
    pressing = false;
    if (fitWaiting) {
      fitWaiting = false;
      fitPanel();
    }
  };
  window.addEventListener("pointerup", release);
  window.addEventListener("pointercancel", release);
  // later growth of Results (e.g. the route charts) only folds
  let contentTimer = null;
  new ResizeObserver(() => {
    clearTimeout(contentTimer);
    contentTimer = setTimeout(fitPanel, CONTENT_SETTLE_MS);
  }).observe($("results"));
  // The tutorial dock also changes available height without resizing the window.
  // Enable scrolling immediately; fold after settling, except while a touch keyboard is in use.
  let resizeTimer = null;
  const resize = () => {
    if (!booting() && !panelCollapsed()) markScroll();
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (!typingOnTouch()) fitPanel(); }, RESIZE_SETTLE_MS);
  };
  window.addEventListener("resize", resize);
  new ResizeObserver(resize).observe($("stage"));
  for (const [id, draw] of Object.entries(redraw)) {
    let last = "";
    new ResizeObserver(() => {
      const c = $(id), key = `${c.clientWidth}x${c.clientHeight}`;
      if (key === last) return;
      last = key;
      if (c.clientWidth > 0 && c.clientHeight > 0) draw();
    }).observe($(id));
  }
  const update = () => updateSummaries(directionDegrees, resultsSummary);
  setInterval(update, SUMMARY_INTERVAL_MS);
  update();
}
