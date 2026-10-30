// Small helpers of the page: elements, canvases, yielding to the browser, time limits.
import { DemoError, ErrorCode } from "../shared/errors.js";
import { yieldTurn } from "../shared/turn.js";
import { pixelRatio } from "../shared/style.js";

/** The element with this id. */
export const $ = (id) => document.getElementById(id);

/**
 * A length token of style.css (a custom property, of the page's root unless `element` is given), in px.  A token
 * that is a calc() reads as px only when style.css registers it (@property), and a registered one only on the
 * element that sets it and its descendants (elsewhere it reads its initial value); a value that is not
 * "<number>px" is an error.
 */
export function cssPx(name, element = document.documentElement) {
  const text = getComputedStyle(element).getPropertyValue(name).trim();
  if (!/^-?[\d.]+px$/.test(text)) throw new Error(`style.css ${name} is "${text}", not a length in px`);
  return parseFloat(text);
}

/** A new element with properties and children. */
export const el = (tag, props = {}, ...children) => {
  const e = Object.assign(document.createElement(tag), props);
  e.append(...children);
  return e;
};

/** A canvas sized for the device pixel ratio, cleared; returns its 2-D context in CSS pixels and the size. */
export function context2d(canvas) {
  const dpr = pixelRatio();
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
  }
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, w, h);
  return { c, w, h };
}

// the longest wait for a screen refresh: a page that is not shown but not reported hidden either (an iframe scrolled
// out of view) gets no refresh at all
const FRAME_WAIT_MS = 250;

/**
 * A turn for the browser between steps of a computation.  While the page is visible: after the next screen refresh
 * (progress shows), or after FRAME_WAIT_MS without one.  While it is hidden (another tab, a minimized or covered
 * window) the browser stops refreshing, so the computation must not wait for one: a message-channel task instead,
 * and a wait already pending ends when the page is hidden.
 */
export const nextFrame = () => {
  if (document.hidden) return yieldTurn();
  return new Promise((resolve) => {
    let done = false;
    const finish = () => {
      if (done) return;
      done = true;
      clearTimeout(fallback);
      document.removeEventListener("visibilitychange", onHidden);
      resolve();
    };
    const onHidden = () => { if (document.hidden) finish(); };
    document.addEventListener("visibilitychange", onHidden);
    const fallback = setTimeout(finish, FRAME_WAIT_MS);
    requestAnimationFrame(() => setTimeout(finish, 0));
  });
};

/** `promise`, or a rejection DemoError TIMEOUT "<what>: no answer in N s" after `ms`. */
export const withLimit = (promise, ms, what) => {
  let timer;
  const limit = new Promise((_, reject) => {
    timer = setTimeout(() => reject(new DemoError(ErrorCode.TIMEOUT, `${what}: no answer in ${ms / 1000} s`)), ms);
  });
  return Promise.race([promise, limit]).finally(() => clearTimeout(timer));
};

const announced = new Map();          // source -> its message still shown (in the order said)
/**
 * Say `text` to a screen reader for `source` (what shows it: "card", "hint", "failure", "results").  The page's one
 * live region, #announce, is always present and never shown, so a message is read out even when what shows it
 * appears at the same moment.
 */
export function announce(text, source) {
  announced.delete(source);
  announced.set(source, text);
  $("announce").textContent = text;
}

/** `source` no longer shows its message: the region holds the latest message still shown, or nothing. */
export function unannounce(source) {
  if (!announced.delete(source)) return;
  $("announce").textContent = [...announced.values()].pop() ?? "";
}

/** A slider's shown value: its output's text and, the same text, what a screen reader says for the slider. */
export function showSliderValue(range, output, text) {
  output.textContent = text;
  range.setAttribute("aria-valuetext", text);
}
