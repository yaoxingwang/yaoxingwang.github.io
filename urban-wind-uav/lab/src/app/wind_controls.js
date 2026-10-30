// The Background wind controls: the direction dial and number, the wind speed (slider, number, or the simulated
// speeds with Ground truth on), the Ground truth switch, and the hint that says how to change a locked setting.
import { SPEED_MIN, SPEED_MAX, SPEED_STEP } from "./config.js";
import { state, currentResults, sameSpeed, cfdSpeeds } from "./state.js";
import { $, el, cssPx, announce, unannounce } from "./dom.js";
import { Dial } from "./dial.js";
import { showWindDirection, setDirectionDial, showSourceRow } from "./wind_display.js";
import { fitPanel } from "./folds.js";
import { run } from "./pipeline.js";

// the hint shows long enough to be read: HINT_NOTICE_MS to see it, then HINT_WORD_MS a word (about 170 words a
// minute, slower than usual reading), at least HINT_MIN_MS
const HINT_NOTICE_MS = 1500, HINT_WORD_MS = 350, HINT_MIN_MS = 3000;
const HINT_ARROW_INSET_PX = 12;  // the arrow of a hint under or over its element stays this far inside its corners
const ANGLE_TOLERANCE = 1e-9;    // radians: directions closer than this are the same
let dial = null;
let hint = null, hintTimer = null;

/** Draw the dial again (it lost its drawing while its section was folded). */
export const redrawDial = () => dial?.draw();

/** The direction shown (whole degrees, -180..180): the dial's while it is dragged. */
export const directionDegrees = () => Math.round(dial ? dial.degrees : state.theta * 180 / Math.PI);

/** The speed slider at speed v: its track filled up to v, and v with its unit for a screen reader. */
const fillSlider = (slider, v) => {
  slider.style.setProperty("--fill", `${(v - SPEED_MIN) / (SPEED_MAX - SPEED_MIN) * 100}%`);
  slider.setAttribute("aria-valuetext", `${v} m/s`);
};
/** A speed in the controls' range and SPEED_STEP steps. */
const snapSpeed = (v) => Math.min(SPEED_MAX,
  Math.max(SPEED_MIN, +(Math.round(v / SPEED_STEP) * SPEED_STEP).toFixed(2)));

/** Ground truth on: this city's simulated speeds as segment buttons (a single one as plain text). */
export function drawSpeedChoice() {
  const speeds = cfdSpeeds();
  const box = $("speed-choice");
  $("speed-single").hidden = speeds.length !== 1;
  box.hidden = speeds.length === 1;
  $("speed-single").textContent = speeds.length === 1 ? `${speeds[0]}` : "";      // the row label says m/s
  box.replaceChildren(...speeds.map((v) => {
    const b = el("button", { type: "button", textContent: `${v}` });
    b.title = `${v} m/s`;
    b.setAttribute("aria-pressed", String(sameSpeed(v, state.speed)));
    b.addEventListener("click", () => { if (!sameSpeed(v, state.speed)) setSpeed(v, true); });
    return b;
  }));
  fitPanel();                // a row changed (also the mode's rows, applyMode): fold what no longer fits
}

/** Set the wind speed (brought into the controls' range and steps, snapSpeed); commit: compute again. */
function setSpeed(value, commit) {
  const slider = $("speed-range"), number = $("speed-number");
  if (!Number.isFinite(value)) { number.value = String(state.speed); return; }
  const v = snapSpeed(value);
  state.speed = v;
  slider.value = String(v);
  fillSlider(slider, v);
  number.value = String(v);
  drawSpeedChoice();
  // nothing to do when the results shown are already for this speed
  if (commit && !(currentResults() && !state.busy)) run({ predict: false });
}

/**
 * With Ground truth on, a press on the locked dial, the direction or a single fixed speed says how to change them: a
 * note pointing at the Ground truth "Off" button (showHint: right of the panel, or under or over the button).
 */
function lockedHint() {
  if (state.mode !== "evaluation") return;
  const off = $("mode").querySelector('[data-mode="free"]');
  showHint("Turn off Ground truth to allow\nany wind speed and direction.\n"
           + "Ground truth is simulated with CFD\nand stored for only a few winds.", off);
}

/**
 * A note (text: "\n" between its lines, each kept whole), shown as long as it takes to read (HINT_WORD_MS a word),
 * pointing at element `at`: just right of the left panel; where the window leaves no room beside the panel, under the
 * element (over it where the window's
 * bottom is too close), inside the window's margins with its arrow at the element's middle.
 */
export function showHint(text, at) {
  at.scrollIntoView({ block: "nearest" });          // a short window's scrolled panel: the element in view first
  const p = $("panel").getBoundingClientRect();
  const r = at.getBoundingClientRect();
  hint.replaceChildren(...text.split("\n").map((line) => el("p", { textContent: line })));
  hint.style.left = "0px";                          // measured where it has the whole window's width
  hint.style.top = "0px";
  const gap = cssPx("--beside-gap"), gutter = cssPx("--gutter");
  const w = hint.offsetWidth, h = hint.offsetHeight;
  let place = "beside", left = p.right + gap, top = r.top + r.height / 2;
  if (left + w > window.innerWidth - gutter) {
    place = r.bottom + gap + h <= window.innerHeight - gutter ? "below" : "above";
    left = Math.max(gutter, Math.min(r.left, window.innerWidth - gutter - w));
    top = place === "below" ? r.bottom + gap : Math.max(gutter, r.top - gap - h);
    const arrow = Math.min(Math.max(r.left + r.width / 2 - left, HINT_ARROW_INSET_PX), w - HINT_ARROW_INSET_PX);
    hint.style.setProperty("--arrow-x", `${Math.round(arrow)}px`);
  }
  hint.dataset.place = place;
  hint.style.left = `${Math.round(left)}px`;
  hint.style.top = `${Math.round(top)}px`;
  hint.classList.add("show");
  announce(hint.textContent, "hint");
  clearTimeout(hintTimer);
  const words = text.split(/\s+/).filter(Boolean).length;
  hintTimer = setTimeout(hideHint, Math.max(HINT_MIN_MS, HINT_NOTICE_MS + words * HINT_WORD_MS));
}

/**
 * The note goes: once read (showHint), when the window's size changes (the element it points at moves), and when the
 * panel collapses, the introduction opens or the layout changes (controls.js).  Before the controls are wired there is
 * none.
 */
export function hideHint() {
  if (!hint) return;
  clearTimeout(hintTimer);
  hint.classList.remove("show");
  unannounce("hint");
}

/**
 * The controls as the mode says.  Ground truth on: wind toward 0 degrees (the dial locked) at one of this city's
 * simulated speeds.  Off: any wind.
 */
function applyMode() {
  const evaluation = state.mode === "evaluation";
  for (const b of $("mode").children) b.setAttribute("aria-pressed", String(b.dataset.mode === state.mode));
  $("controls").classList.toggle("evaluation", evaluation);
  // the controls of one mode only: the speed number and slider with Ground truth off, the CFD speeds with it on
  for (const e of $("controls").querySelectorAll(".free-only")) e.hidden = evaluation;
  for (const e of $("controls").querySelectorAll(".eval-only")) e.hidden = !evaluation;
  dial.locked = evaluation;
  // Ground truth on: the direction is plain text (no input box, style.css), so it is not a stop of the Tab key either
  const direction = $("direction-number");
  direction.readOnly = evaluation;
  direction.tabIndex = evaluation ? -1 : 0;
  showSourceRow();
  drawSpeedChoice();
  dial.draw();
}

/** A mode change: the wind brought into the mode; returns whether the wind must be predicted again. */
export function enterMode(mode) {
  const was = state.mode;
  state.mode = mode;
  let predict = false;
  if (mode === "evaluation") {
    if (was === "free") state.freeWind = { degrees: dial.degrees, theta: state.theta, speed: state.speed };   // for Off
    if (Math.abs(state.theta) > ANGLE_TOLERANCE) {
      dial.silentSet(0);
      state.theta = 0;
      predict = true;
    }
    const speeds = cfdSpeeds();
    if (speeds.length && !speeds.some((v) => sameSpeed(v, state.speed))) {
      setSpeed(speeds.reduce((a, b) => (Math.abs(b - state.speed) < Math.abs(a - state.speed) ? b : a)), false);
    }
  } else if (state.freeWind) {
    // Ground truth off again: the wind the user had set in this mode
    const { degrees, theta, speed } = state.freeWind;
    if (Math.abs(theta - state.theta) > ANGLE_TOLERANCE) {
      dial.silentSet(degrees);
      state.theta = theta;
      predict = true;
    }
    if (!sameSpeed(speed, state.speed)) setSpeed(speed, false);
  }
  applyMode();
  showWindDirection();
  return predict;
}

/** Ground truth On / Off pressed: the mode with its wind (enterMode), then a new computation. */
function setMode(mode) {
  if (mode === state.mode || !state.city) return;
  const predict = enterMode(mode);
  run({ predict });
}

/** Wire the Background wind controls (once the graphics card and the model are ready). */
export function wireWindControls() {
  // while the dial is dragged only the dial and the inflow arrows follow; the wind direction in use changes on
  // release (a computation running meanwhile would otherwise read the dragged direction half way)
  dial = new Dial($("dial"), $("direction-number"),
    (theta) => { state.dialTheta = theta; showWindDirection(theta); },
    () => {
      if (state.dialTheta == null || state.dialTheta === state.theta) { showWindDirection(); return; }
      state.theta = state.dialTheta;
      showWindDirection();
      run({ predict: true });           // a wind change keeps the current view
    });
  setDirectionDial(dial);
  dial.draw();
  const slider = $("speed-range"), number = $("speed-number");
  // while the slider is dragged only its number follows; the wind speed changes on release (routes planned with the
  // previous speed would otherwise be labelled with the new one)
  slider.addEventListener("input", () => {
    const v = snapSpeed(parseFloat(slider.value));
    fillSlider(slider, v);
    number.value = String(v);
  });
  slider.addEventListener("change", () => setSpeed(parseFloat(slider.value), true));
  number.addEventListener("change", () => setSpeed(parseFloat(number.value), true));
  // Escape in a number field takes back what was typed: it shows the wind in use again (and that Escape does no more)
  const shown = {
    "direction-number": () => dial.showNumber(),
    "speed-number": () => { number.value = String(state.speed); },
  };
  for (const [id, show] of Object.entries(shown)) {
    $(id).addEventListener("keydown", (event) => {
      if (event.key !== "Escape") return;
      const typed = $(id).value;
      show();
      if ($(id).value !== typed) event.preventDefault();
    });
  }
  hint = el("div", { id: "side-hint" });
  document.body.append(hint);
  window.addEventListener("resize", hideHint);
  for (const id of ["dial", "direction-number", "speed-single"]) $(id).addEventListener("pointerdown", lockedHint);
  for (const b of $("mode").children) b.addEventListener("click", () => setMode(b.dataset.mode));
  applyMode();
  fillSlider(slider, state.speed);
}
