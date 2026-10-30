// The colour legend of the wind display: the scale as the layer appears over the corridor ground (particles: the
// comet heads' opacity, slices and the prediction sweep: the slices'), with ticks at whole steps.
import { windColour } from "../shared/colormap.js";
import { SLICE_OPACITY, PARTICLE_OPACITY, CORRIDOR_GROUND_RGB, PLACEHOLDER_BAR_CSS, CANVAS_FONT,
         seenOver } from "../shared/style.js";
import { state, particlesShown } from "./state.js";
import { $, el, context2d } from "./dom.js";

const GRADIENT_STOPS = 64;              // colour stops of the bar's gradient
const LABEL_GAP_PX = 10;                // a tick label keeps this far from the maximum's label
/** The tick step (m/s) for a scale up to `top`. */
const speedTickStep = (top) => (top <= 6 ? 1 : top <= 12 ? 2 : 5);
const legendProbe = document.createElement("canvas").getContext("2d");    // text measurement of the legend's labels

/** The colour legend as the layer appears: particles = the comet-head colours, slices = the map over the corridor. */
export function drawLegend(range) {
  if ($("legend-bar").clientWidth < 8) return;        // folded: redrawn when it gets its size again
  // while the prediction sweep runs (no wind range yet): its fixed range, in the slice colours
  const sweep = !range && Boolean(state.sweepRange);
  if (sweep) range = state.sweepRange;
  const { c, w, h } = context2d($("legend-bar"));
  if (!range) {
    // a grey placeholder until the wind's range exists
    c.fillStyle = PLACEHOLDER_BAR_CSS;
    c.fillRect(0, 0, w, h);
    $("legend-low").textContent = "–";
    $("legend-high").textContent = "– m/s";
    $("legend-ticks").hidden = true;
    $("legend").querySelector(".range-row").hidden = false;     // the placeholder's low and high texts instead
    return;
  }
  const [low, high] = range;
  const alpha = particlesShown() && !sweep ? PARTICLE_OPACITY : SLICE_OPACITY;   // the legend as the layer appears
  // one continuous gradient (1-px strips left hairline seams between them at fractional device-pixel scales)
  const gradient = c.createLinearGradient(0, 0, w, 0);
  for (let k = 0; k <= GRADIENT_STOPS; k++) {
    const t = k / GRADIENT_STOPS;
    const [r, g, b] = seenOver(windColour(t), alpha, CORRIDOR_GROUND_RGB);
    gradient.addColorStop(t, `rgb(${r},${g},${b})`);
  }
  c.fillStyle = gradient;
  c.fillRect(0, 0, w, h);
  $("legend-low").textContent = "";
  $("legend-high").textContent = "";
  // a tick and a label at every whole step from 0 to the top, kept only while their label stays LABEL_GAP_PX
  // clear of the maximum's label (measured on the bar's real width); then the maximum itself
  const ticks = $("legend-ticks");
  ticks.hidden = false;
  $("legend").querySelector(".range-row").hidden = true;
  const step = speedTickStep(high);
  // the tick labels' own font (style.css; a browser without a computed font shorthand: the same size)
  legendProbe.font = getComputedStyle(ticks).font || `11px ${CANVAS_FONT}`;
  const maxLeft = w - legendProbe.measureText(`${high.toFixed(1)} m/s`).width;
  const labels = [];
  for (let v = low; v < high; v += step) {
    const width = legendProbe.measureText(String(+v.toFixed(1))).width;
    const right = v === low ? width : w * (v - low) / (high - low) + width / 2;
    if (right + LABEL_GAP_PX <= maxLeft) labels.push(v);
  }
  labels.push(high);
  ticks.replaceChildren(...labels.map((v, k) => {
    const s = el("span", { textContent: k === labels.length - 1 ? `${v.toFixed(1)} m/s` : `${+v.toFixed(1)}` });
    s.style.left = `${(100 * (v - low) / (high - low)).toFixed(3)}%`;
    if (k === 0) {
      s.className = "first";
    } else if (k === labels.length - 1) {
      s.className = "last";
    }
    return s;
  }));
}
