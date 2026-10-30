// The city's size on its edges: a fixed scale bar would not change with the zoom, so the city's own side lengths are
// written on its edges instead - text only, no lines, lying flat on the ground outside the middle of each edge, along
// it and upright seen from outside the city, a fixed size in the scene.  The user turns the scene, so there is a label
// for each of the four edges and only the two edges nearest the camera show theirs (at most two): the lengths always
// sit on the near side.
import * as THREE from "../../vendor/three.module.js";
import { CELL_M, kmText } from "../shared/units.js";
import { CANVAS_FONT, EDGE_LABEL_CSS } from "../shared/style.js";
import { reducedMotion } from "../shared/platform.js";
import { RENDER_ORDER, GROUND_Z } from "./materials.js";
import { project } from "./projection.js";

// a label's text height on the ground (m) for a city of REFERENCE_SIDE_M, and its opacity: the city's size is the
// page's least important mark, so it stays quieter than the routes, the pins and the arrows (regular weight,
// EDGE_LABEL_CSS at this opacity); a larger city's labels (and their gap) grow with its side, so they look the same in
// its opening view, which is that much farther away
const LABEL_M = 18, LABEL_OPACITY = 0.45, REFERENCE_SIDE_M = 1200;
const LABEL_TEXEL_PX = 96;             // its texture's text height (pixels: sharp also seen close)
const FADE_S = 0.15;                   // a label fades out, then the other edge's fades in, over this each
const GAP_M = 10;                      // from the city's edge to a label's text box (m), outside its edge's middle
// no ping-pong: the shown edge of a pair changes only once the camera is SWITCH_DEG past the city's middle line (seen
// from above, from the city's centre), and a label off the free area shows again only SHOW_MARGIN_PX inside it
const SWITCH_DEG = 10, SHOW_MARGIN_PX = 12;

/** The city's side lengths on its edges, shown or faded every frame (place). */
export class EdgeLabels {
  /** Labels of `viewer` (its camera decides which show). */
  constructor(viewer) {
    this.viewer = viewer;
    this.labels = null;
    // per dimension: the edge wanted (the camera's side), the edge shown and its opacity
    this.dims = { x: { want: "s", shown: "s", alpha: 1 }, y: { want: "e", shown: "e", alpha: 1 } };
  }

  /** The four labels of a city as a group (the viewer's "edgeLabels" layer). */
  build(city) {
    const W = city.nx * CELL_M, H = city.ny * CELL_M, grow = Math.max(W, H) / REFERENCE_SIDE_M;
    const textM = LABEL_M * grow;
    const off = GAP_M * grow + textM * 1.25 / 2;     // the text box's middle (its height: textM x 1.25, below)
    const group = new THREE.Group();
    group.userData.labels = [];
    // [x, y, length, edge]: south, north (east-west length), west, east (north-south length)
    const edges = [[W / 2, -off, W, "s"], [W / 2, H + off, W, "n"], [-off, H / 2, H, "w"], [W + off, H / 2, H, "e"]];
    for (const [x, y, metres, edge] of edges) {
      const text = `${kmText(metres)} km`;
      const canvas = document.createElement("canvas"), context = canvas.getContext("2d");
      const font = `400 ${LABEL_TEXEL_PX}px ${CANVAS_FONT}`;
      context.font = font;
      canvas.width = Math.ceil(context.measureText(text).width * 1.06);
      canvas.height = Math.ceil(LABEL_TEXEL_PX * 1.25);
      context.font = font; context.fillStyle = EDGE_LABEL_CSS;
      context.textBaseline = "middle"; context.textAlign = "center";
      context.fillText(text, canvas.width / 2, canvas.height / 2);
      const texture = new THREE.CanvasTexture(canvas);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.anisotropy = 8;              // read at a slant on the ground
      // flat on the ground (just above it), hidden by buildings in front as the ground is
      const h = textM * canvas.height / LABEL_TEXEL_PX, w = h * canvas.width / canvas.height;
      const material = new THREE.MeshBasicMaterial({ map: texture, transparent: true, depthWrite: false });
      const label = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material);
      label.renderOrder = RENDER_ORDER.edgeLabels;
      label.position.set(x, y, GROUND_Z.edgeLabel);
      // along its edge, its top towards the city: upright seen from outside
      label.rotation.z = { s: 0, n: Math.PI, e: Math.PI / 2, w: -Math.PI / 2 }[edge];
      label.userData = { edge };
      group.add(label);
      group.userData.labels.push(label);
    }
    this.labels = group.userData.labels;
    return group;
  }

  /**
   * The labels this frame: each fixed on the ground; shown: the near edge of each pair
   * (wantedEdges), not while its place is off the free area (onScreen); a change fades the old label out over FADE_S,
   * then the new one in, so at most two show.  `delta` = seconds since the last frame (Infinity: at once).
   */
  place(delta = Infinity) {
    const viewer = this.viewer, labels = this.labels;
    if (!labels || !viewer.city) return;
    this.wantedEdges();
    const step = Number.isFinite(delta) && !reducedMotion() ? delta / FADE_S : Infinity;
    for (const d of Object.values(this.dims)) {
      if (d.shown !== d.want) {
        d.alpha = Math.max(0, d.alpha - step);
        if (d.alpha === 0) d.shown = d.want;
      } else {
        d.alpha = Math.min(1, d.alpha + step);
      }
    }
    for (const label of labels) {
      const { edge } = label.userData, d = edge === "s" || edge === "n" ? this.dims.x : this.dims.y;
      label.userData.seen = this.onScreen(label, label.userData.seen ?? true);
      label.userData.alpha = d.shown === edge && label.userData.seen
        ? Math.min(1, (label.userData.alpha ?? 0) + step, d.alpha) : Math.max(0, (label.userData.alpha ?? 0) - step);
      label.material.opacity = LABEL_OPACITY * label.userData.alpha;
      label.visible = label.userData.alpha > 0;
    }
  }

  /**
   * The near edge of each pair, by the direction the camera looks from (its azimuth: the view's centre may be off the
   * city's, beside the panel): south or north, east or west; kept until that direction is SWITCH_DEG past the middle.
   */
  wantedEdges() {
    const a = this.viewer.azimuth * Math.PI / 180, m = Math.sin(SWITCH_DEG * Math.PI / 180);
    const sn = Math.sin(a), cs = Math.cos(a), { x, y } = this.dims;
    if (x.want === "s" ? sn > m : sn < -m) x.want = x.want === "s" ? "n" : "s";
    if (y.want === "e" ? cs < -m : cs > m) y.want = y.want === "e" ? "w" : "e";
  }

  /**
   * Whether label `label`'s place is on the canvas and not under the panel (its left inset); a label shown so far
   * (wasSeen) goes when its place leaves that, one hidden comes back only SHOW_MARGIN_PX inside it.
   */
  onScreen(label, wasSeen) {
    const { size, insets, camera } = this.viewer, p = label.position;
    const [sx, sy, depth] = project(camera, p.x, p.y, p.z, size.width, size.height);
    const m = wasSeen ? 0 : SHOW_MARGIN_PX;
    return depth <= 1 && sx >= insets.left + m && sx <= size.width - m && sy >= m && sy <= size.height - m;
  }
}
