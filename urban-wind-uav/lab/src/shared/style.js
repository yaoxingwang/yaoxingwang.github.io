// The look shared by the scene and the page's own canvases (legends, the dial, the charts, the pins): colours,
// opacities, the canvas font and the pixel ratio.  A white model on the page background; colour only for data.
import { windColour } from "./colormap.js";

/** [r, g, b] bytes of a 0xrrggbb colour. */
const rgb = (hex) => [(hex >> 16) & 255, (hex >> 8) & 255, hex & 255];
/** A 0xrrggbb colour as a CSS string "#rrggbb". */
const css = (hex) => `#${hex.toString(16).padStart(6, "0")}`;

// the text colours of style.css (--ink, --muted, --faint) and its white and slider track (--white, --track)
const INK = 0x242b32;
const MUTED = 0x52606b, FAINT = 0x65717c, TRACK = 0xd8e0e5;
export const WHITE = 0xffffff;               // also the routes' casing
export const INK_CSS = css(INK), MUTED_CSS = css(MUTED), FAINT_CSS = css(FAINT);
export const WHITE_CSS = css(WHITE), TRACK_CSS = css(TRACK);
/** --ink at opacity `alpha` (style.css: rgb(var(--ink-rgb) / alpha)) for canvas drawing. */
export const inkAlpha = (alpha) => `rgba(${rgb(INK).join(",")},${alpha})`;

export const BACKGROUND = 0xf5f7f8;          // = --bg in style.css
export const BASE_PLATE = 0xe3e8eb;          // the city's base plate, outside the corridor
export const CORRIDOR_GROUND = 0xebf0f3;     // the ground inside the predicted corridor, lighter than the plate
export const BASE_EDGE = 0xd8e0e5;           // the plate's 1 px outline (= --outline in style.css)
export const CORRIDOR_LINE = MUTED;          // the corridor's 1 px ground line
export const BUILDING = 0xfbfcfd;            // the white model's buildings
// colours of single canvas details: the legend's colour bar before the wind's range exists, the dial's ticks at 0,
// 90, 180 and 270 degrees, the edge labels' text (a grey between --ink and --muted), the chart's hover box
export const PLACEHOLDER_BAR_CSS = "#e3e8eb";
const TICK_GREY_CSS = "#939fa8";
/** The dial's ticks at 0, 90, 180 and 270 degrees. */
export const MAJOR_TICK_CSS = TICK_GREY_CSS;
/** The pins' stems and ground dots: the dial's tick grey, lighter than --muted. */
export const PIN_STEM_CSS = TICK_GREY_CSS;
export const EDGE_LABEL_CSS = "#47545f";
export const HOVER_BOX_CSS = `rgba(${rgb(WHITE).join(",")},0.94)`;
/** The font of text drawn on canvases, after its weight and size (style.css --font, shortened). */
export const CANVAS_FONT = "Inter, -apple-system, system-ui, sans-serif";
/** Canvases are drawn at the device pixel ratio up to this (sharper costs more than it shows). */
const MAX_PIXEL_RATIO = 2;
/** The pixel ratio canvases are drawn at. */
export const pixelRatio = () => Math.min(MAX_PIXEL_RATIO, window.devicePixelRatio || 1);
export const SLICE_OPACITY = 0.55;           // the speed maps
export const PARTICLE_OPACITY = 0.4;         // a little more transparent than the slices
export const PLANE_OPACITY = 0.3;            // the particles' birth plane while its slider is held

/**
 * [r, g, b] of the page background and of the corridor ground (legends are drawn the way a translucent layer
 * appears over them).
 */
export const BACKGROUND_RGB = rgb(BACKGROUND);
export const CORRIDOR_GROUND_RGB = rgb(CORRIDOR_GROUND);

/** A green from the wind colour map for the particle birth plane. */
export const ARROW_COLOUR = windColour(0.7);
/** Fixed annotation colour, independent of the wind-speed colour map. */
export const INFLOW_ARROW = 0x6acb98;

/** Colour `rgb` (0-255 channels) at opacity `alpha` over the opaque colour `under`: the colour seen, rounded. */
export const seenOver = (rgb, alpha, under) => rgb.map((v, i) => Math.round(v * alpha + under[i] * (1 - alpha)));

/** The inflow arrows' colour as CSS (the wind dial's pointer has it too). */
export const INFLOW_ARROW_CSS = css(INFLOW_ARROW);
