// The page's layout: whether the opened panel stands beside the scene (the camera centres the city in the free area
// right of it) or lies over it (the free area is the whole window).  It is decided here, from the window's width,
// and marked on the page's root (html[data-layout]) for style.css; controls.js follows its changes (onLayoutChange).
import { $, cssPx } from "./dom.js";

// the panel stands beside the scene only where the city, drawn there, is at least as wide as the panel: the city takes
// at least this share of the free area's width near that point (measured 0.69 to 0.745 for windows 400 to 1100 px
// high: the taller, the larger), so the free area must be the panel's width / CITY_SHARE (about 522 px: windows from
// about 930 px wide; phones, upright tablets and narrow windows get the panel over the scene).  Below about 400 px
// high the height limits the city at any width.  Only the width decides: the window's height changes with a phone's
// toolbars and keyboard (the project's tests check the rule at several heights).
const CITY_SHARE = 0.69;
// the layout is decided again once the window has kept its size this long (a dragged window edge does not flip it
// back and forth)
const LAYOUT_SETTLE_MS = 150;
// a touch screen lower than this is a phone held sideways: the page opens with the panel collapsed there too
const SHORT_TOUCH_SCREEN_PX = 500;
const listeners = [];

/**
 * The scene's free area as insets (px) from the stage's edges, as style.css has it now (#stage --left-inset: beside
 * the panel, or the page's margin where the panel is collapsed or lies over the scene; --scene-inset, --scene-bottom).
 */
export function freeArea() {
  const edge = cssPx("--scene-inset");
  return { left: cssPx("--left-inset", $("stage")), right: edge, top: edge, bottom: cssPx("--scene-bottom") };
}

/** Whether the opened panel lies over the scene (as marked by trackLayout). */
export const panelOverScene = () => document.documentElement.dataset.layout === "over";

/** Whether the page opens with the panel collapsed: where it lies over the scene, or on a phone held sideways. */
export const opensCollapsed = () =>
  panelOverScene() || (matchMedia("(pointer: coarse)").matches && window.innerHeight < SHORT_TOUCH_SCREEN_PX);

/** Call `fn()` after every change of the layout (not at its first mark). */
export function onLayoutChange(fn) { listeners.push(fn); }

/**
 * Mark the layout now and after every resize: at once for a single one (a window maximized, snapped, a tablet
 * turned), and once more when the window has kept its size for LAYOUT_SETTLE_MS (an edge dragged across the
 * threshold flips it only where the drag starts or rests that long, not at every crossing).
 */
export function trackLayout() {
  const mark = () => {
    const free = $("stage").clientWidth - cssPx("--beside-inset") - cssPx("--scene-inset");
    const was = document.documentElement.dataset.layout;
    const now = free * CITY_SHARE < cssPx("--left-w") ? "over" : "beside";
    document.documentElement.dataset.layout = now;
    if (was && was !== now) for (const fn of listeners) fn();
  };
  mark();
  let timer = null;
  window.addEventListener("resize", () => {
    if (timer === null) mark();
    clearTimeout(timer);
    timer = setTimeout(() => {
      timer = null;
      mark();
    }, LAYOUT_SETTLE_MS);
  });
}
