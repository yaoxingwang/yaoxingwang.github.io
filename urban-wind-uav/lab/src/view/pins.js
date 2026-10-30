// The start and goal pins: a light grey stem from a dot on the ground up to the end point, a bead of the same grey
// at the end point (it closes the routes' open ends there), and above it a map pin - a rounded label whose bottom
// narrows to a point aimed at the end point, stopping just short of the bead.  The pin names the end point and is its
// drag handle.  Stem, dot and bead are parts of the scene (metres, they scale with it); the pin is a label: a constant
// size on screen at its smallest, larger with the scene when the view is closer, redrawn at the resolution it is
// shown at (Viewer sizePinButtons).
import * as THREE from "../../vendor/three.module.js";
import { PIN_STEM_CSS, EDGE_LABEL_CSS, CANVAS_FONT, WHITE_CSS, inkAlpha } from "../shared/style.js";
import { RENDER_ORDER, GROUND_Z } from "./materials.js";

// the pin on screen at its smallest (CSS pixels): its head's height, the point under it, and the gap from the point's
// tip down to the end point; closer than where it is that small, it scales with the scene (its head PIN_HEAD_M high)
// up to PIN_MAX_ZOOM times its smallest
const HEAD_PX = 26, TIP_PX = 16, TIP_GAP_PX = 3;
const PIN_HEAD_M = 26, PIN_MAX_ZOOM = 3;
/** The pin's size on screen as a multiple of its smallest, at `metresPerPixel` (Viewer sizePinButtons). */
export const pinZoom = (metresPerPixel) => Math.min(PIN_MAX_ZOOM, Math.max(1, PIN_HEAD_M / HEAD_PX / metresPerPixel));
/** The pin's tip keeps this far (CSS px) above the bead at least (Viewer sizePinButtons). */
export const BEAD_CLEAR_PX = 2;
// a white shape looks larger than a dark one of the same size: the dark pin is drawn this much larger, so the two
// look the same size
const DARK_GROWTH = 1.07;
const HINT_GAP_PX = 3;            // from the pin's head up to its hint's text box (drawHint: past its glow margin)
// the stem's radius and the ground dot's radius (metres)
const STEM_RADIUS_M = 0.8, DOT_RADIUS_M = 4;
/** The bead's radius (metres): over the widest route tube (routes.js, the chosen one) with its white casing. */
export const BEAD_RADIUS_M = 7;
// the pin heads' widths on screen at the pin's smallest (CSS px)
const headWidths = {};

/**
 * The pin's mark after its name, `size` high, its left edge at x and its middle at y (canvas 2-D context c): the
 * start's a ring (the usual start mark of a route; a triangle read as a play button), the goal's a flag on a short
 * pole.
 */
function drawMark(c, start, x, y, size, colour) {
  c.fillStyle = colour; c.strokeStyle = colour;
  if (start) {
    const width = size * 0.22;           // the ring as tall as the flag, the mark's whole size
    c.beginPath();
    c.arc(x + size / 2, y, (size - width) / 2, 0, Math.PI * 2);
    c.lineWidth = width;
    c.stroke();
  } else {
    const pole = size * 0.13;
    c.fillRect(x, y - size / 2, pole, size);
    c.beginPath();
    c.moveTo(x + pole, y - size / 2);
    c.lineTo(x + size * 0.9, y - size / 2);
    c.lineTo(x + size * 0.72, y - size * 0.22);
    c.lineTo(x + size * 0.9, y + size * 0.06);
    c.lineTo(x + pole, y + size * 0.06);
    c.closePath();
    c.fill();
  }
}

/**
 * The pin drawn at `k` canvas pixels per CSS pixel: a pill-shaped head with the name and, on its right, a ring
 * (start) or a flag (goal), the bottom flowing into a point; in one slate grey (the edge labels' text colour,
 * not black) and white: light (white, a slate border and contents) or dark (slate, a white border and contents), a
 * soft shadow; the point's tip TIP_GAP_PX above the canvas's bottom middle (the end point).  Both pins have the
 * width of the longer name's (the contents centred), the dark one DARK_GROWTH larger.
 */
function drawPin(text, dark, k) {
  const grow = dark ? DARK_GROWTH : 1;
  const h = Math.round(HEAD_PX * grow * k), tip = TIP_PX * k, gap = TIP_GAP_PX * k, sh = 8 * k, r = h / 2;
  const pad = 11 * k, mark = 9 * k, markGap = 6 * k;
  const font = `600 ${13 * k}px ${CANVAS_FONT}`;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const textW = probe.measureText(text).width;
  const widest = Math.max(...["Start", "Goal"].map((name) => probe.measureText(name).width));
  const w = Math.ceil((pad + widest + markGap + mark + pad) * grow);
  headWidths[text] = w / k;
  const canvas = document.createElement("canvas");
  canvas.width = w + 2 * sh; canvas.height = sh + h + tip + gap;
  const c = canvas.getContext("2d");
  const x0 = sh, y0 = sh, cx = canvas.width / 2, bottom = y0 + h;
  // the outline: the pill, its lower edge drawn in towards the middle and down to the tip in two soft curves
  c.beginPath();
  c.moveTo(x0 + r, y0);
  c.lineTo(x0 + w - r, y0);
  c.arc(x0 + w - r, y0 + r, r, -Math.PI / 2, Math.PI / 2);
  c.lineTo(cx + 8 * k, bottom);
  c.quadraticCurveTo(cx + 1.5 * k, bottom + 4 * k, cx, bottom + tip);
  c.quadraticCurveTo(cx - 1.5 * k, bottom + 4 * k, cx - 8 * k, bottom);
  c.lineTo(x0 + r, bottom);
  c.arc(x0 + r, y0 + r, r, Math.PI / 2, -Math.PI / 2);
  c.closePath();
  const fill = dark ? EDGE_LABEL_CSS : WHITE_CSS, ink = dark ? WHITE_CSS : EDGE_LABEL_CSS;
  c.shadowColor = inkAlpha(0.2); c.shadowBlur = 6 * k; c.shadowOffsetY = 1.5 * k;
  c.fillStyle = fill;
  c.fill();
  c.shadowColor = "transparent";
  c.strokeStyle = ink; c.lineWidth = 1.5 * k; c.lineJoin = "round";
  c.stroke();
  c.font = font; c.fillStyle = ink; c.textBaseline = "middle";
  const left = cx - (textW + markGap + mark) / 2;            // the name and its mark, centred
  c.fillText(text, left, y0 + h / 2 + 0.5 * k);
  drawMark(c, !dark, left + textW + markGap, y0 + h / 2, mark, ink);
  return canvas;
}

// the hints over the pins and the drone (drawHint): no frame of their own (a pin already is one) - semi-bold text in
// the pins' slate, a soft white glow behind it so it reads on buildings and particles; CSS px: the text size, a
// line's height, the glow's reach (the canvas margin around the text), the icon's gap after the text, its line width
const HINT_TEXT_PX = 13, HINT_LINE_H = 16, HINT_GLOW_PX = 6, HINT_ICON_GAP = 6, HINT_ICON_LINE_PX = 1.7;
// the halo behind them: a white outline this wide (CSS px), softened by this blur, so the text has a solid white
// edge on any background (buildings, particles)
const HINT_HALO_PX = 5, HINT_HALO_BLUR_PX = 2.5;

/**
 * A minimal video camera as a line icon (a rounded body, a lens on the right), in the hints' slate, `h` CSS px high
 * (its width 1.75 h), its left edge at x and its middle at y, CSS px times k; off: crossed by a slash with a white
 * gap along it (camera off).
 */
function drawCamera(c, x, y, h, k, off, halo = false) {
  const u = (h / 8) * k, line = HINT_ICON_LINE_PX * k;      // the drawing in units of h / 8
  c.lineJoin = "round"; c.lineCap = "round";
  c.lineWidth = halo ? HINT_HALO_PX * k : line; c.strokeStyle = halo ? WHITE_CSS : EDGE_LABEL_CSS;
  c.fillStyle = WHITE_CSS;
  c.beginPath();
  c.roundRect(x + line / 2, y - 4 * u + line / 2, 9 * u, 8 * u - line, 2 * u);
  c.moveTo(x + 10 * u, y - 1.2 * u); c.lineTo(x + 14 * u - line / 2, y - 3.4 * u);
  c.lineTo(x + 14 * u - line / 2, y + 3.4 * u); c.lineTo(x + 10 * u, y + 1.2 * u);
  if (halo) c.fill();                    // the halo pass fills the icon white too: nothing shows through it
  c.stroke();
  if (!off || halo) return;
  // the slash steeper than the lens's lower edge (it passes under the lens, not along it)
  const slash = () => { c.beginPath(); c.moveTo(x + 2 * u, y - 5 * u); c.lineTo(x + 10.5 * u, y + 5 * u); };
  slash(); c.lineWidth = line + 2.4 * k; c.strokeStyle = WHITE_CSS; c.stroke();
  slash(); c.lineWidth = line; c.strokeStyle = EDGE_LABEL_CSS; c.stroke();
}

/**
 * A hint drawn at `k` canvas pixels per CSS pixel: `text` (its lines parted by "\n", left-aligned; and icon "camera"
 * or "camera-off" right of them, as high as the lines, drawCamera) on a white halo (HINT_HALO_PX), HINT_GLOW_PX of
 * canvas around them.
 */
function drawHint(text, k, icon = null) {
  const font = `600 ${HINT_TEXT_PX * k}px ${CANVAS_FONT}`, glow = HINT_GLOW_PX * k, lines = text.split("\n");
  const lineH = HINT_LINE_H;
  const probe = document.createElement("canvas").getContext("2d");
  probe.font = font;
  const textW = Math.max(...lines.map((line) => probe.measureText(line).width));
  // the icon a little taller than a line, centred on the lines (CSS px)
  const blockH = lines.length * lineH, iconH = lines.length > 1 ? 13 : 9;
  const iconW = icon ? (HINT_ICON_GAP + 1.75 * iconH) * k : 0;
  const canvas = document.createElement("canvas");
  canvas.width = Math.ceil(textW + iconW + 2 * glow); canvas.height = Math.ceil(blockH * k + 2 * glow);
  const c = canvas.getContext("2d");
  const ix = glow + textW + HINT_ICON_GAP * k, iy = glow + blockH * k / 2;
  const lineY = (i) => glow + (i + 0.5) * lineH * k + 0.5 * k;
  c.font = font; c.textAlign = "left"; c.textBaseline = "middle"; c.lineJoin = "round";
  // the halo: a wide white outline of the text and the icon, softened by a white blur; then the text and icon
  c.shadowColor = WHITE_CSS; c.shadowBlur = HINT_HALO_BLUR_PX * k;
  c.lineWidth = HINT_HALO_PX * k; c.strokeStyle = WHITE_CSS;
  lines.forEach((line, i) => c.strokeText(line, glow, lineY(i)));
  if (icon) drawCamera(c, ix, iy, iconH, k, false, true);
  c.shadowColor = "transparent";
  c.fillStyle = EDGE_LABEL_CSS;
  lines.forEach((line, i) => c.fillText(line, glow, lineY(i)));
  if (icon) drawCamera(c, ix, iy, iconH, k, icon === "camera-off");
  return canvas;
}

/** `sprite` showing `canvas` drawn at `k` canvas pixels per CSS pixel (its old picture let go). */
function showCanvas(sprite, canvas, k) {
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.anisotropy = 4;
  sprite.material.map?.dispose();
  sprite.material.map = texture;
  sprite.material.needsUpdate = true;
  sprite.userData.px = canvas.height / k;                   // sprite height in CSS px (with the shadow margin)
  sprite.userData.aspect = canvas.width / canvas.height;
}

/** A sprite over the scene (a label: no depth, drawn last). */
function labelSprite() {
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ depthTest: false, depthWrite: false, transparent: true }));
  sprite.renderOrder = RENDER_ORDER.pinButton;
  return sprite;
}

/**
 * A hint over the scene that keeps its size on screen (the viewer scales it): `text` as the pins' "Drag to move",
 * anchored at its bottom middle.
 */
export function hintSprite(text) {
  const sprite = labelSprite();
  showCanvas(sprite, drawHint(text, 4), 4);
  sprite.center.set(0.5, 0);
  sprite.userData.text = text;
  return sprite;
}

/**
 * Hint `sprite` (hintSprite) saying `text` now, with `icon` after it (drawHint; null: none), drawn at `k` canvas
 * pixels per CSS pixel (as sharp as its size on screen needs).
 */
export function setHintText(sprite, text, icon = null, k = 4) {
  if (sprite.userData.text === text && sprite.userData.icon === icon && sprite.userData.k === k) return;
  showCanvas(sprite, drawHint(text, k, icon), k);
  sprite.userData.k = k;
  Object.assign(sprite.userData, { text, icon });
}

/**
 * Draw pin head `head` (makePin) and its hint at `k` canvas pixels per CSS pixel: as sharp as its size on screen
 * needs (Viewer sizePinButtons).
 */
export function drawPinAt(head, k) {
  const { pin, hint, name, dark } = head.userData;
  showCanvas(pin, drawPin(name, dark, k), k);
  showCanvas(hint, drawHint("Drag to move", k), k);
  head.userData.k = k;
}

/**
 * A pin group for `which` ("start" | "goal") at `position` [x, y, z] (metres).  group.userData.head: the end point
 * (an empty object at the stem's top) holding the pin (userData.pin, its anchor the end point), its hint
 * (userData.hint, shown by the viewer while the page has just opened; userData.hint.userData.above: its anchor's
 * height under it, in its own heights) and the pin's size on screen at its smallest (userData.size: { w, h, tip, gap }
 * CSS px: the head's width and height, the point under it, the gap down to the end point).
 */
export function makePin(which, position) {
  const group = new THREE.Group();
  const z = position[2];
  const grey = new THREE.MeshBasicMaterial({ color: PIN_STEM_CSS });
  const stem = new THREE.Mesh(new THREE.CylinderGeometry(STEM_RADIUS_M, STEM_RADIUS_M, z, 8), grey);
  stem.rotation.x = Math.PI / 2;
  stem.position.z = z / 2;
  group.add(stem);
  // the bead at the end point, in the stem's grey and flat as the stem; drawn after the routes' see-through part
  // (routes.js), so the tube ends inside it do not show through it
  const bead = new THREE.Mesh(new THREE.SphereGeometry(BEAD_RADIUS_M, 24, 16),
                              new THREE.MeshBasicMaterial({ color: PIN_STEM_CSS, transparent: true }));
  bead.renderOrder = RENDER_ORDER.pinBead;
  bead.position.z = z;
  group.add(bead);
  const start = which === "start";
  const head = new THREE.Object3D();
  head.position.z = z;
  Object.assign(head.userData, { name: start ? "Start" : "Goal", dark: !start });
  Object.assign(head.userData, { pin: labelSprite(), hint: labelSprite() });
  drawPinAt(head, 4);
  head.userData.pin.center.set(0.5, 0);                     // the anchor (the end point) at the bottom middle
  head.userData.pin.userData.gap = TIP_GAP_PX;
  head.userData.raise = 0;                 // how far (CSS px) the pin and its hint are raised to clear the bead
  const h = HEAD_PX * (head.userData.dark ? DARK_GROWTH : 1);
  head.userData.size = { w: headWidths[head.userData.name], h, tip: TIP_PX, gap: TIP_GAP_PX };
  const hint = head.userData.hint;
  hint.userData.above = (TIP_GAP_PX + TIP_PX + h + HINT_GAP_PX - HINT_GLOW_PX) / hint.userData.px;
  hint.center.set(0.5, -hint.userData.above);               // the hint's bottom above the pin's head
  head.add(head.userData.pin, hint);
  group.add(head);
  group.userData.head = head;
  // the dot on the ground under the end point, in the stem's grey; one pass, as a slice (materials.js sliceMaterial):
  // it is flat
  const dot = new THREE.Mesh(new THREE.CircleGeometry(DOT_RADIUS_M, 24), new THREE.MeshBasicMaterial({
    color: PIN_STEM_CSS, side: THREE.DoubleSide, depthWrite: false, forceSinglePass: true }));
  dot.position.z = GROUND_Z.pinFoot;
  group.add(dot);
  group.position.set(position[0], position[1], 0);
  return group;
}
