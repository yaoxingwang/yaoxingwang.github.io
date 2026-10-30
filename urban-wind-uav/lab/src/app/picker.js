// The city picker: height-map previews of every urban block at one scale (true relative size), built on its first
// opening; a click on a tile opens that city (onPick).
import { CELL_M, LAYER_M, kmText } from "../shared/units.js";
import { BACKGROUND_RGB, pixelRatio } from "../shared/style.js";
import { loadCity, cityPreviews } from "../city/city.js";
import { PICKER_PX_PER_M, PICKER_RAMP } from "./config.js";
import { state, shortName } from "./state.js";
import { $, el } from "./dom.js";

const LARGE_RATIO = 1.5;             // a city this many times wider than the smallest gets its own column
let pickerReady = null;
let onPick = () => {};               // set by setPickHandler

/** What a click on another city's tile does: handler(manifest entry). */
export function setPickHandler(handler) { onPick = handler; }

/**
 * Open or close the city picker (built on its first opening).  The keyboard focus moves to the open city's tile on
 * opening and back to the city button on closing (when it was in the picker).
 */
export function openPicker(open) {
  const panel = $("city-picker");
  if (open && panel.hidden) {
    panel.hidden = false;
    $("city-button").setAttribute("aria-expanded", "true");
    pickerReady ??= buildPicker();       // its tiles exist at once; their previews follow
    markPicker();
    $("city-tiles").querySelector('[aria-current="true"]')?.focus();
  } else if (!open && !panel.hidden) {
    const hadFocus = panel.contains(document.activeElement);
    panel.hidden = true;
    $("city-button").setAttribute("aria-expanded", "false");
    if (hadFocus) $("city-button").focus();
  }
}

/** The open city's tile is marked. */
export function markPicker() {
  for (const b of $("city-tiles").querySelectorAll("button")) {
    b.setAttribute("aria-current", String(b.dataset.id === state.cityEntry?.id));
  }
}

/** Every city's tile (on the first opening): sizes at one scale, then the height-map previews as they load. */
async function buildPicker() {
  const { cities } = state.manifest;
  const grid = $("city-tiles");
  const sides = cities.map((e) => e.shape[1] * CELL_M);
  const small = Math.min(...sides);
  const tiles = cities.map((entry, k) => {
    const [ny, nx] = entry.shape;
    const w = Math.round(sides[k] * PICKER_PX_PER_M), h = Math.round(ny * CELL_M * PICKER_PX_PER_M);   // CSS pixels
    const large = sides[k] > LARGE_RATIO * small;
    const canvas = el("canvas");
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const button = el("button", { type: "button", className: "tile" },
      canvas, el("span", { className: "tile-name", textContent: shortName(entry) }),
      // the size of each city, width x height
      el("span", { className: "tile-size", textContent: `${kmText(nx * CELL_M)} × ${kmText(ny * CELL_M)} km` }),
      // the grid of cells the wind is predicted and planned on, width x height
      el("span", { className: "tile-size", textContent: `${nx} × ${ny} grid` }),
      // a city without a CFD wind is marked
      ...(entry.cfd?.length ? [] : [el("span", { className: "tile-size", textContent: "no simulation" })]));
    button.dataset.id = entry.id;
    button.addEventListener("click", () => {
      openPicker(false);
      if (state.cityEntry?.id !== entry.id) onPick(entry);
    });
    return { button, canvas, w, h, large };
  });
  // the small cities in a grid (its columns: style.css .tile-grid), the large city after them (beside or under it)
  grid.replaceChildren(el("div", { className: "tile-grid" }, ...tiles.filter((t) => !t.large).map((t) => t.button)),
                       ...tiles.filter((t) => t.large).map((t) => t.button));
  markPicker();
  // a preview that does not load leaves its tile blank
  const preview = (e) => loadCity(e).catch((error) => { console.warn(error.message); return null; });
  const maps = await Promise.all(cities.map(preview));
  // one height scale for every preview: up to the tallest building over all cities (layers)
  const topLayers = Math.max(...cities.map((e) => e.maximum_height_m)) / LAYER_M;
  const dpr = pixelRatio();
  const sizes = tiles.map(({ w, h }) => [Math.round(w * dpr), Math.round(h * dpr)]);
  const pixels = await cityPreviews(maps, sizes, topLayers, PICKER_RAMP, BACKGROUND_RGB)
    .catch((error) => { console.warn(error.message); return []; });
  tiles.forEach(({ canvas }, k) => {
    if (!pixels[k]) return;
    const [W, H] = sizes[k];
    canvas.width = W; canvas.height = H;
    canvas.getContext("2d").putImageData(new ImageData(pixels[k], W, H), 0, 0);
  });
}
