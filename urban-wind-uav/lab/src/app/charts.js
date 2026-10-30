// The Results charts: one horizontal energy bar per route (Ground truth on), and line charts along the routes
// (altitude, ground speed, airspeed) with a hover read-out.
import { INK_CSS, FAINT_CSS, CANVAS_FONT, HOVER_BOX_CSS, inkAlpha } from "../shared/style.js";
import { interpolateAlong } from "../shared/geometry.js";
import { pointAccelerations, thrust } from "../planner/energy.js";
import { METHODS, CHART_TABS, methodById } from "./config.js";
import { state, routedMethods } from "./state.js";
import { $, el, context2d } from "./dom.js";

// the line chart's plot margins (CSS pixels): the value labels on the left, the distance labels below
const PLOT_LEFT = 30, PLOT_RIGHT = 6, PLOT_TOP = 18, PLOT_BOTTOM = 16;

/** The longest of the routes' charted lengths (m). */
const chartLength = () => Math.max(...Object.values(state.profiles).map((profile) => profile.s.at(-1)));

/** The distance along the routes (m) under the page x coordinate `clientX` on the line chart. */
export function chartDistanceAt(clientX) {
  const sMax = chartLength(), r = $("chart").getBoundingClientRect();
  const share = (clientX - r.left - PLOT_LEFT) / (r.width - (PLOT_LEFT + PLOT_RIGHT));
  return Math.min(sMax, Math.max(0, share * sMax));
}

/** A "nice" step (1, 2 or 5 x 10^k) close to `raw`. */
function niceStep(raw) {
  const p = 10 ** Math.floor(Math.log10(raw)), f = raw / p;
  return p * (f <= 1 ? 1 : f <= 2 ? 2 : f <= 5 ? 5 : 10);
}

/** Decimals from the step (1, 2, 5 x 10^k): as many as the step needs. */
const formatTick = (t, step) => t.toFixed(Math.max(0, -Math.floor(Math.log10(step) + 1e-9)));

/**
 * The energy axis (Wh/km): lo / hi = smallest / largest route energy, span = max(hi - lo, 0.15 hi); the axis runs from
 * hi - 1.6 span (not below 0, and at most 0.6 hi: the bars exaggerate at most 2.5 x) to hi + 0.1 span, both rounded
 * outward to a nice step that gives about 4 intervals.
 */
function energyAxis(values) {
  const lo = Math.min(...values), hi = Math.max(...values);
  const span = Math.max(hi - lo, 0.15 * hi);
  const rawLo = Math.max(0, Math.min(hi - 1.6 * span, 0.6 * hi)), rawHi = hi + 0.1 * span;
  const step = niceStep((rawHi - rawLo) / 4);
  const a0 = Math.max(0, Math.floor(rawLo / step + 1e-9) * step), a1 = Math.ceil(rawHi / step - 1e-9) * step;
  const ticks = [];
  for (let t = a0; t <= a1 + step * 1e-6; t += step) ticks.push(+t.toFixed(6));
  return { a0, a1, step, ticks };
}

/**
 * One horizontal bar per route: energy per km (the paper's unit-distance energy consumption), scored in the CFD wind.
 * A click on a row calls onSelect(method id), its check box onToggle(method id).
 */
export function showEnergyChart(plans, onSelect, onToggle) {
  const box = $("energy-chart");
  const methods = routedMethods(plans);
  const wh = (m) => plans[m.id].whPerKm;
  const axis = energyAxis(methods.map(wh));
  // bars and ticks are positioned in one track width (the same element structure for both); the value label sits
  // right of the bar end, absolutely positioned, so it never takes width from the bar
  const x = (v) => `${(100 * (v - axis.a0) / (axis.a1 - axis.a0)).toFixed(3)}%`;
  // one line per route (check box, then a button: label left, bar right) so the single panel fits a 720 px screen
  const rows = methods.map((m) => {
    const row = el("button", { type: "button", className: "bar-row" });
    row.style.setProperty("--c", m.colour);
    row.setAttribute("aria-pressed", String(state.selected === m.id));
    const bar = el("span", { className: "bar" });
    bar.style.width = x(wh(m));
    const value = el("span", { className: "bar-value", textContent: wh(m).toFixed(1) });
    value.style.left = x(wh(m));
    row.title = `${m.name}: ${m.sub}`;
    row.setAttribute("aria-label", `${m.name} ${wh(m).toFixed(1)} Wh/km`);
    row.append(el("span", { className: "bar-label", textContent: m.name }),
               el("span", { className: "bar-track" }, bar, value));
    row.addEventListener("click", () => onSelect(m.id));
    return routeLine("bar-line", m, onToggle, row);
  });
  const ticks = el("span", { className: "bar-track axis" }, ...axis.ticks.map((t) => {
    const s = el("span", { className: "tick", textContent: formatTick(t, axis.step) });
    s.style.left = x(t);
    return s;
  }));
  const axisRow = el("div", { className: "bar-row axis-row" }, ticks);
  const axisLine = el("div", { className: "bar-line" }, el("span"), axisRow);
  box.replaceChildren(...rows, axisLine,
                      el("p", { className: "axis-caption", textContent: "Wh/km" }));
  box.hidden = false;
}

/**
 * Per-point quantities of a planned route, as planner/energy.js trajectoryEnergy computes them: s = distance flown (m),
 * the height, the ground speed, the airspeed |velocity - wind(point)| and the thrust vector [tx, ty, tz]
 * (the drone's tilt in the scene).
 */
export function routeProfile(plan, windAt) {
  const pts = plan.points, n = pts.length;
  const vel = plan.velocities;
  const { acc } = pointAccelerations(pts, plan.speeds, vel);
  const s = [0];
  for (let i = 0; i < n - 1; i++) {
    s.push(s[i] + Math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1], pts[i + 1][2] - pts[i][2]));
  }
  const out = { s, height: [], ground: [], air: [], thrust: [] };
  for (let i = 0; i < n; i++) {
    const w = windAt(pts[i][0], pts[i][1], pts[i][2]);
    const ax = vel[i][0] - w[0], ay = vel[i][1] - w[1], az = vel[i][2] - w[2];
    out.height.push(pts[i][2]);
    out.ground.push(Math.hypot(...vel[i]));
    out.air.push(Math.hypot(ax, ay, az));
    out.thrust.push(thrust(ax, ay, az, ...acc[i]));
  }
  return out;
}

/**
 * A route's line in the energy bars or the chart legend (class `kind`): its check box (the route drawn or not; a
 * click calls onToggle(method id)), then `button` (choosing it).
 */
function routeLine(kind, m, onToggle, button) {
  const check = el("input", { type: "checkbox", className: "route-check", checked: !state.hiddenRoutes.has(m.id),
                              title: `Show ${m.name}` });
  check.setAttribute("aria-label", `Show ${m.name}`);
  check.addEventListener("change", () => onToggle(m.id));
  const line = el("div", { className: kind }, el("label", { className: "check-hit" }, check), button);
  line.style.setProperty("--c", m.colour);
  line.dataset.id = m.id;
  line.classList.toggle("off", state.hiddenRoutes.has(m.id));
  return line;
}

/** The energy bars' and the legend's routes marked as drawn or not (state.hiddenRoutes), in place. */
export function markShownRoutes() {
  for (const line of document.querySelectorAll("#energy-chart .bar-line[data-id], #chart-legend .legend-item")) {
    const off = state.hiddenRoutes.has(line.dataset.id);
    line.classList.toggle("off", off);
    line.querySelector(".route-check").checked = !off;
  }
}

/**
 * The legend of the line charts, two by two: per route its check box (a click calls onToggle(method id)) and its name
 * (a click calls onSelect(method id)).
 */
export function drawChartLegend(plans, onSelect, onToggle) {
  $("chart-legend").replaceChildren(...routedMethods(plans).map((m) => {
    const b = el("button", { type: "button", className: "route-key", textContent: m.name });
    b.title = `${m.name}: ${m.sub}`;
    b.setAttribute("aria-pressed", String(state.selected === m.id));
    b.addEventListener("click", () => onSelect(m.id));
    return routeLine("legend-item", m, onToggle, b);
  }));
}

/**
 * The line chart of the current tab: distance along the route on x, one line per drawn route (its check box set), the
 * selected one on top, the value axis fitted to them (to every route when none is drawn); hoverS (m): a vertical line
 * there and every drawn route's value in a box.
 */
export function drawChart(hoverS = null) {
  const q = state.tab;
  if (!state.profiles || !CHART_TABS[q]) return;
  const canvas = $("chart");
  if (canvas.clientWidth < 40 || canvas.clientHeight < 30) return;
  // its name for a screen reader: the tab's quantity (the tab's title where it has one)
  const tab = $("chart-tabs").querySelector(`[data-tab="${q}"]`);
  canvas.setAttribute("aria-label", `${tab.title || tab.textContent} along each route`);
  const routed = METHODS.map((m) => m.id).filter((id) => state.profiles[id]);
  if (!routed.length) return;
  const ids = routed.filter((id) => !state.hiddenRoutes.has(id));
  const sMax = chartLength();
  const { c, w, h } = context2d(canvas);
  const all = (ids.length ? ids : routed).flatMap((id) => state.profiles[id][q]);
  let lo = Math.min(...all), hi = Math.max(...all);
  const pad = Math.max((hi - lo) * 0.08, q === "height" ? 2 : 0.2);
  lo -= pad; hi += pad;
  const L = PLOT_LEFT, R = PLOT_RIGHT, T = PLOT_TOP, B = PLOT_BOTTOM;
  const px = (sv) => L + (w - L - R) * sv / sMax, py = (v) => T + (h - T - B) * (1 - (v - lo) / (hi - lo || 1));
  let step = niceStep((hi - lo || 1) / 3);
  for (let k = 0; k < 40 && (h - T - B) * step / (hi - lo || 1) < 16; k++) step = niceStep(step * 1.6);
  c.font = `400 12px ${CANVAS_FONT}`;
  c.fillStyle = FAINT_CSS; c.strokeStyle = inkAlpha(0.08); c.lineWidth = 1;
  c.textAlign = "right"; c.textBaseline = "middle";
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    const y = Math.round(py(v)) + 0.5;
    c.beginPath(); c.moveTo(L, y); c.lineTo(w - R, y); c.stroke();
    c.fillText(formatTick(v, step), L - 5, y);
  }
  c.textBaseline = "alphabetic"; c.textAlign = "left"; c.fillText("0 km", L, h - 2);
  c.textAlign = "right"; c.fillText(`${(sMax / 1000).toFixed(2)} km`, w - R, h - 2);
  const order = [...ids.filter((id) => id !== state.selected), ...ids.filter((id) => id === state.selected)];
  for (const id of order) {
    const m = methodById(id), p = state.profiles[id], sel = id === state.selected;
    c.strokeStyle = m.colour; c.lineWidth = sel ? 2.25 : 1.5; c.globalAlpha = sel ? 1 : 0.9;
    c.beginPath();
    p.s.forEach((sv, k) => (k ? c.lineTo(px(sv), py(p[q][k])) : c.moveTo(px(sv), py(p[q][k]))));
    c.stroke();
  }
  c.globalAlpha = 1;
  // the unit, once, at the top left
  c.textAlign = "left"; c.textBaseline = "top"; c.fillStyle = FAINT_CSS;
  c.fillText(CHART_TABS[q].unit, 2, 1);
  if (hoverS !== null) drawHover(c, w, h, { ids, sMax, q, px, py }, hoverS);
}

/** The hover read-out of the line chart: a dot on every line and a box with the distance and every route's value. */
function drawHover(c, w, h, { ids, sMax, q, px, py }, hoverS) {
  const sv = Math.min(sMax, Math.max(0, hoverS)), x = Math.round(px(sv)) + 0.5;
  c.strokeStyle = inkAlpha(0.45); c.beginPath(); c.moveTo(x, PLOT_TOP); c.lineTo(x, h - PLOT_BOTTOM); c.stroke();
  const rows = [];
  for (const id of ids) {
    const p = state.profiles[id], m = methodById(id);
    const v = interpolateAlong(p.s, p[q], sv);
    if (sv <= p.s.at(-1)) {
      c.fillStyle = m.colour; c.beginPath(); c.arc(x, py(v), 3, 0, Math.PI * 2); c.fill();
      rows.push([m, v]);
    }
  }
  c.font = `500 11px ${CANVAS_FONT}`;
  const lines = [[null, `${(sv / 1000).toFixed(2)} km`],
                 ...rows.map(([m, v]) => [m, `${m.name} ${v.toFixed(q === "height" ? 0 : 1)}`])];
  const bw = Math.max(...lines.map(([, t]) => c.measureText(t).width)) + 18, lh = 13, bh = lines.length * lh + 6;
  const bx = x + 8 + bw < w - PLOT_RIGHT ? x + 8 : x - 8 - bw, by = PLOT_TOP;
  c.fillStyle = HOVER_BOX_CSS; c.strokeStyle = inkAlpha(0.15);
  c.beginPath(); c.roundRect(bx, by, bw, bh, 6); c.fill(); c.stroke();
  c.textAlign = "left"; c.textBaseline = "middle";
  lines.forEach(([m, t], k) => {
    const y = by + 3 + lh * (k + 0.5);
    if (m) { c.fillStyle = m.colour; c.fillRect(bx + 6, y - 1, 7, 2.5); }
    c.fillStyle = m ? INK_CSS : FAINT_CSS;
    c.fillText(t, bx + (m ? 16 : 6), y);
  });
}
