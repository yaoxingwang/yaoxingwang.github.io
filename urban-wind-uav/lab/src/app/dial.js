// The wind direction dial: a disc with ticks and labels and one pointer where the wind blows, turned with the camera
// (top-down from its azimuth) so the pointer shows the wind's direction on screen; its number field keeps the map
// angle.  Pressing or dragging the disc sets the direction in STEP_DEGREES steps (not while locked: Ground truth on).
import { INFLOW_ARROW_CSS, MUTED_CSS, CANVAS_FONT, WHITE_CSS, TRACK_CSS, MAJOR_TICK_CSS } from "../shared/style.js";
import { context2d } from "./dom.js";

const STEP_DEGREES = 5;          // the dial sets the direction in these steps (also the number field's step)
const HUB_SHARE = 0.12;          // the hub (no direction) is this share of the dial's size around its centre
const REDRAW_DEG = 0.05;         // a camera turn of the dial smaller than this is not redrawn

/** The wind direction dial (see the file header). */
export class Dial {
  /**
   * The dial on `canvas` with its number field `number`.  onChange(theta) while it turns (the direction shown),
   * onCommit() when a press or a typed number ends with a new direction.
   */
  constructor(canvas, number, onChange, onCommit) {
    Object.assign(this, { canvas, number, onChange, onCommit, degrees: 0, locked: false });
    // the dial turns with the camera (top-down from its azimuth): `view` = screen angle (degrees, counter-clockwise
    // from right) of map 0 degrees, so the dial's pointer points the way the wind goes on screen.  The number field
    // keeps the map angle.
    this.view = 0;
    // the pointer's offset from the dial's centre (CSS pixels), and the dial's box
    const offset = (event) => {
      const rect = canvas.getBoundingClientRect();
      return [event.clientX - rect.left - rect.width / 2, event.clientY - rect.top - rect.height / 2, rect];
    };
    const angle = (event) => {
      const [dx, dy] = offset(event);
      return Math.round((Math.atan2(-dy, dx) * 180 / Math.PI - this.view) / STEP_DEGREES) * STEP_DEGREES;
    };
    let dragging = false;
    // a press on the hub has no direction: ignored
    const onHub = (event) => {
      const [dx, dy, rect] = offset(event);
      return Math.hypot(dx, dy) < HUB_SHARE * Math.min(rect.width, rect.height);
    };
    let pressedAt = 0;
    canvas.addEventListener("pointerdown", (e) => {
      if (this.locked || onHub(e)) return;
      dragging = true;
      pressedAt = this.degrees;
      canvas.setPointerCapture(e.pointerId);
      this.set(angle(e));
    });
    canvas.addEventListener("pointermove", (e) => { if (dragging && !onHub(e)) this.set(angle(e)); });
    // a press that leaves the direction as it was computes nothing again
    const stop = (e) => {
      if (!dragging) return;
      dragging = false;
      canvas.releasePointerCapture(e.pointerId);
      if (this.degrees !== pressedAt) this.onCommit();
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
    number.addEventListener("change", () => {
      const before = this.degrees;
      this.set(Math.round(parseFloat(number.value)));     // whole degrees, as the field shows
      this.showNumber();
      if (this.degrees !== before) this.onCommit();
    });
    this.draw();
  }

  /** The number field and the dial's screen-reader label show the direction (whole degrees). */
  showNumber() {
    const degrees = Math.round(this.degrees);
    this.number.value = String(degrees);
    this.canvas.setAttribute("aria-label", `Wind direction ${degrees}°`);     // what a screen reader says
  }

  /** The camera turned: `degrees` = the screen angle of map 0 degrees (see `view`). */
  turnWithCamera(degrees) {
    if (!(Math.abs(degrees - this.view) > REDRAW_DEG)) return;
    this.view = degrees;
    this.draw();
  }

  /** Show `degrees` without reporting a change. */
  silentSet(degrees) {
    this.degrees = degrees;
    this.showNumber();
    this.draw();
  }

  /** Turn to `degrees` (brought into -180..180) and report the change. */
  set(degrees) {
    if (!Number.isFinite(degrees)) { this.showNumber(); return; }
    let d = ((degrees % 360) + 360) % 360;
    if (d > 180) d -= 360;
    if (d === this.degrees) return;
    this.degrees = d;
    this.showNumber();
    this.draw();
    this.onChange(d * Math.PI / 180);
  }

  /**
   * Map angles (0 = map +x, 90 = map +y; no N: +y is not verified as north) drawn turned by `view` (the camera
   * azimuth, top-down), so the pointer shows where the wind blows on screen.  Labels inside the ring, upright;
   * read-only with Ground truth on (the same look, not greyed).
   */
  draw() {
    if (this.canvas.clientWidth < 8) return;
    const { c, w, h } = context2d(this.canvas);
    c.font = `500 12px ${CANVAS_FONT}`;
    // the disc uses the whole canvas, so it is large enough to drag
    const cx = w / 2, cy = h / 2, radius = Math.min(h, w) / 2 - 2;
    c.fillStyle = WHITE_CSS;
    c.beginPath(); c.arc(cx, cy, radius, 0, Math.PI * 2); c.fill();
    // one pointer, only where the wind blows: a slim needle from a round base at the centre to the ring, in the
    // inflow arrows' colour as they appear; drawn first, under ticks and numbers
    const r = (this.degrees + this.view) * Math.PI / 180, ux = Math.cos(r), uy = -Math.sin(r), nx = -uy, ny = ux;
    const front = radius - 4, base = 4.5;          // the pointer's tip, and the half width of its base
    c.fillStyle = INFLOW_ARROW_CSS;
    c.beginPath(); c.moveTo(cx + ux * front, cy + uy * front);
    c.lineTo(cx + nx * base, cy + ny * base); c.lineTo(cx - nx * base, cy - ny * base); c.closePath(); c.fill();
    c.beginPath(); c.arc(cx, cy, base, 0, Math.PI * 2); c.fill();
    c.strokeStyle = TRACK_CSS; c.lineWidth = 1;
    c.beginPath(); c.arc(cx, cy, radius - 0.5, 0, Math.PI * 2); c.stroke();
    c.lineCap = "round";
    for (let a = 0; a < 360; a += 15) {
      const major = a % 90 === 0, rr = (a + this.view) * Math.PI / 180, len = major ? 6 : 3;
      c.strokeStyle = major ? MAJOR_TICK_CSS : TRACK_CSS; c.lineWidth = major ? 1.5 : 1;
      c.beginPath();
      c.moveTo(cx + Math.cos(rr) * (radius - 1), cy - Math.sin(rr) * (radius - 1));
      c.lineTo(cx + Math.cos(rr) * (radius - 1 - len), cy - Math.sin(rr) * (radius - 1 - len));
      c.stroke();
    }
    c.textAlign = "center"; c.textBaseline = "middle"; c.fillStyle = MUTED_CSS;
    for (const [label, a] of [["0", 0], ["90", 90], ["180", 180], ["−90", 270]]) {
      const rr = (a + this.view) * Math.PI / 180, cs = Math.cos(rr), sn = Math.sin(rr);
      // inside the ring, just within the major tick: half the label's extent along this direction from it
      const half = Math.abs(cs) * c.measureText(label).width / 2 + Math.abs(sn) * 6;
      c.fillText(label, cx + cs * (radius - 9 - half), cy - sn * (radius - 9 - half) + 0.5);
    }
  }
}
