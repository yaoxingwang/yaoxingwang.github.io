// What the page is doing: the loading card in the scene (preparation before a mission), the status line and progress
// bar where the results go (a computation), and the message over the scene when the page cannot run.
import { state, cfdEntry } from "./state.js";
import { $, el, announce, unannounce } from "./dom.js";
import { cfdShareDone } from "./cfd.js";

// texts of more than one step
export const PREPARING = "Preparing the demo…";
export const COMPUTING_MDDF = "Computing multi-directional distance features (MDDF)…";

// ---------- progress ----------
// The progress bar shows that the page is working and will finish soon; it does not measure the computation: it
// moves steadily, never stops, never goes back, and ends full.  It moves at a constant speed, so every stage
// (prediction, route planning) takes a share of the bar in proportion to its time: to MAIN_PERCENT over the expected
// time (the last measured time of the same kind), then on to TAIL_END_PERCENT, slower, if this computation takes
// longer.  It is an animation of the bar's transform, run by the browser's compositor, so it keeps moving while the
// main thread is busy (e.g. reading the wind back from the GPU).  When the routes are ready it fills to 100 % in
// FINISH_MS and the results appear.
// per kind of computation: the expected time (ms; until the first measurement, a typical one) and the city size
// (cells) it was measured on (expectedMs scales it by size)
const expected = { full: { ms: 2200, cells: 90000 }, planOnly: { ms: 1400, cells: 90000 } };
// the bar's stages (%): to MAIN_PERCENT in the expected time, then on to TAIL_END_PERCENT at 1 / TAIL_SLOWDOWN of
// that speed
const MAIN_PERCENT = 90, TAIL_END_PERCENT = 97, TAIL_SLOWDOWN = 10;
const progressByCity = {};                       // "kind:city id" -> the last measured time in that city
const FINISH_MS = 160;
const FINISH_SLACK_MS = 10;                       // progressFinish resolves this long after the bar is full
const CFD_TICK_MS = 250;                          // a ground-truth download's share shown this often (showCfdLoading)
const CFD_GLIDE_MS = 300;                         // the bar's glide to each new share of the download

/** The expected time of a computation: measured in this city, else the last measurement scaled by the city's size. */
function expectedMs(kind) {
  const city = state.city;
  if (city && progressByCity[`${kind}:${city.id}`]) return progressByCity[`${kind}:${city.id}`];
  return expected[kind].ms * (city ? city.nx * city.ny / expected[kind].cells : 1);
}

/** The measured time of a computation ("full": prediction and planning, "planOnly"), for the next bar's duration. */
export function recordProgress(kind, ms, city) {
  expected[kind] = { ms, cells: city.nx * city.ny };
  progressByCity[`${kind}:${city.id}`] = ms;
}

/** The progress bar's fill (scaled along x as the computation goes on). */
const bar = () => $("progress").querySelector(".progress-fill");
let progressRun = null;                          // the bar's running animation

/** Start the bar for a computation of `kind`. */
export function progressStart(kind = "full") {
  const element = bar(), ms = Math.round(expectedMs(kind));
  progressRun?.cancel();
  element.style.transition = "none";
  element.style.transform = "scaleX(0)";
  // 0 -> MAIN_PERCENT in ms, then on to TAIL_END_PERCENT at 1 / TAIL_SLOWDOWN of that speed; linear in each part
  const tail = ms * TAIL_SLOWDOWN * (TAIL_END_PERCENT - MAIN_PERCENT) / MAIN_PERCENT;
  const frames = [{ transform: "scaleX(0)" }, { transform: `scaleX(${MAIN_PERCENT / 100})`, offset: ms / (ms + tail) },
                  { transform: `scaleX(${TAIL_END_PERCENT / 100})` }];
  progressRun = element.animate(frames, { duration: ms + tail, easing: "linear", fill: "forwards" });
}

/** Stop the animation where the bar is drawn now (later changes continue from there). */
function holdProgress(element) {
  const now = getComputedStyle(element).transform;
  progressRun?.cancel();
  progressRun = null;
  element.style.transition = "none";
  element.style.transform = now;
  void element.offsetWidth;            // the held transform is applied before any later transition starts
}

/** Fill the bar; resolves when it is full. */
export function progressFinish() {
  const element = bar();
  holdProgress(element);
  element.style.transition = `transform ${FINISH_MS}ms ease-out`;
  element.style.transform = "scaleX(1)";
  return new Promise((resolve) => setTimeout(resolve, FINISH_MS + FINISH_SLACK_MS));
}

/**
 * The status by kind: "progress" (a computation: the text and the progress bar where the results go), "loading"
 * (preparation before a mission: the card in the middle of the scene, and the text alone where the results go),
 * "note" (the text alone where the results go: nothing is computing) or null (hidden).  The loading card shows the
 * paper's title until the first city has opened (in a window lower than 500 px never: style.css), afterwards the
 * status line alone.
 */
export function status(text, kind = null) {
  const box = $("progress"), card = $("loading-card");
  if (kind !== "loading") unannounce("card");
  card.classList.toggle("show", kind === "loading");
  if (kind === "loading") {
    $("loading-intro").hidden = state.firstLoadDone;
    $("loading-text").textContent = text;
    announce(text, "card");
  }
  box.hidden = kind === null;
  if (kind !== null) {
    $("headline-main").hidden = true;
    $("progress-text").textContent = text;
    box.querySelector(".progress-track").hidden = kind !== "progress";    // a bar only for a computation
  }
}

/** The page cannot run: a message card over the scene, with a reload button under the text when asked. */
export function fatal(text, reload = false) {
  const box = $("error");
  box.hidden = false;
  const card = el("div", { textContent: text });
  if (reload) {
    card.append(el("button", { type: "button", className: "primary", textContent: "Reload the page",
                               onclick: () => location.reload() }));
  }
  box.replaceChildren(card);           // role=alert (index.html): read out at once
  status("Unavailable", "note");
}

/**
 * Before a computation, while the ground truth it needs is downloaded (pipeline.js run): the download's share in the
 * status text and on the bar.  Returns the function that stops it.
 */
export function showCfdLoading(stale) {
  const file = cfdEntry()?.file, element = bar();
  progressRun?.cancel();
  progressRun = null;
  element.style.transition = "none";
  element.style.transform = "scaleX(0)";
  const tick = () => {
    if (stale()) return;
    const share = cfdShareDone(file);
    status(`Loading the ground-truth wind… ${Math.round(100 * share)} %`, "progress");
    element.style.transition = `transform ${CFD_GLIDE_MS}ms linear`;
    element.style.transform = `scaleX(${share})`;
  };
  tick();
  const timer = setInterval(tick, CFD_TICK_MS);
  return () => clearInterval(timer);
}
