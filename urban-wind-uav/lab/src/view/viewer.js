// The 3-D scene (three.js): buildings, the mission corridor, start and goal pins, the planned routes and the drone
// flying one, the wind particles, the speed slices and the inflow arrows.  City metres, z up.  The scene fills the
// page; the cards float over it, and the camera's view offset centres the orbit target in the free area between them
// (setInsets).  The page uses only the public methods below.
import * as THREE from "../../vendor/three.module.js";
import { CELL_M, cellCentreM } from "../shared/units.js";
import { BACKGROUND, BASE_PLATE, CORRIDOR_GROUND, BASE_EDGE, CORRIDOR_LINE, pixelRatio } from "../shared/style.js";
import { reducedMotion } from "../shared/platform.js";
import { yieldTurn } from "../shared/turn.js";
import { paletteTexture, buildingMaterial, RENDER_ORDER, GROUND_Z } from "./materials.js";
import { particleMesh, planeFrameMesh, sliceGroup, sliceTexels, verticalSliceGroup, inflowArrowGroup, ARROW_LENGTH_M,
         ARROW_GAP_M, ARROW_HEIGHT_M, PARTICLE_WIDTH_PX } from "./wind_layers.js";
import { buildingGeometry } from "./buildings.js";
import { makePin, pinZoom, drawPinAt, hintSprite, setHintText, BEAD_RADIUS_M, BEAD_CLEAR_PX } from "./pins.js";
import { routeMeshes, ROUTE_LAPSE } from "./routes.js";
import { droneGroup, droneFlight, DRONE_TRAIL, DRONE_SPAN_M } from "./drone.js";
import { EdgeLabels } from "./edge_labels.js";
import { attachInteraction } from "./interaction.js";
import { project, metresPerPixelAt } from "./projection.js";

// The scene is drawn at 30 frames per second, not 60 (about half the GPU and CPU work), on a fixed schedule: a frame
// is due ANIMATION_FRAME_MS after the last one was due and is drawn on the screen refresh nearest that time, so a
// screen of any refresh rate gets 30 a second and a callback run a little late does not put a frame off.  The time
// step is the time since the last drawn frame, so every animation keeps its speed.
const ANIMATION_FRAME_MS = 1000 / 30;
const REFRESH_WINDOW = 8;              // the refresh interval: the shortest of this many recent ones (one the browser
                                       // skips doubles an interval)
// a refresh this share of an interval before the due time draws, and one later than this starts the schedule again:
// a little more than half, so on screens whose refreshes fall half an interval from it (75, 165 Hz) the timestamps'
// jitter does not decide
const DUE_REACH = 0.6;
const MAX_STEP_S = 0.1;                // the longest time step of one frame (a hidden tab, a long task)
// the light: a sky / ground hemisphere (a cool ground bounce) and a white sun; colour stays for the data
const SKY_LIGHT = 0xffffff, GROUND_LIGHT = 0xe0e6eb, HEMISPHERE_INTENSITY = 2.1;
const SUN_LIGHT = 0xffffff, SUN_INTENSITY = 1.6;
// the buildings' shadows: the sun's shadow map (texels a side), its bias, and the faint shadow on the ground
const SHADOW_MAP_PX = 2048, SHADOW_NORMAL_BIAS = 1, SHADOW_COLOUR = 0x52616b, SHADOW_OPACITY = 0.14;
const HOVER_SCALE = 1.08;              // a hovered pin grows by this (also while it is dragged)
const HOVER_LIFT_PX = 3;               // and lifts this far
const PIN_HIT_MARGIN_PX = 8;           // a pin is grabbed this far outside its head too (about 44 px high)
const POINT_KEEP_PX = 10;              // but not within this distance of its end point (where the routes meet)
// the flying drone is hovered and clicked within its whole span on screen, at least DRONE_HIT_MIN_PX; once hovered,
// it stays so while the pointer is within DRONE_KEEP times that (it moves under a still pointer)
const DRONE_HIT_MIN_PX = 40, DRONE_KEEP = 1.5;
const DRONE_HOVER_SCALE = 1.1;         // a hovered drone grows by this, and says what a click does above it
// that hint's canvas bottom above the drone's centre on screen: DRONE_HINT_RISE of its span (seen from above the
// drone looks flatter than its span), less DRONE_HINT_SINK_PX (CSS px at its smallest: the canvas's glow margin,
// pins.js HINT_GLOW_PX, so the text sits just above the drone)
const DRONE_HINT_RISE = 0.3, DRONE_HINT_SINK_PX = 4;
// the hints over the pins and the drone appear and go with a quick fade of HINT_FADE_MS (fadeHint)
const HINT_FADE_MS = 180;
// "Drag to move" over the start pin from when the drone's own hint ends (or the drone is clicked, or the page asks:
// showDragHint) until a pin is dragged, or for DRAG_HINT_MS counted while the view is not on the drone (then the pins
// are likely in view); once gone, not shown by itself again; over a pin always while the pointer is on it
const DRAG_HINT_MS = 30000;
// the drone's hint shown by itself (announceDrone: during the opening flight), DRONE_ANNOUNCE_AFTER_MS to
// DRONE_ANNOUNCE_UNTIL_MS from then; gone when the drone is clicked
const DRONE_ANNOUNCE_AFTER_MS = 3000, DRONE_ANNOUNCE_UNTIL_MS = 8000;
const PIN_LAYERS = Object.freeze({ start: "startPin", goal: "goalPin" });   // the pins' layer names
const CORRIDOR_LINE_OPACITY = 0.35;    // the corridor's ground line
// particle width x (1 / metres per pixel)^POWER, kept within MIN..MAX: mildly following the zoom
const PARTICLE_ZOOM_POWER = 0.35, PARTICLE_ZOOM_MIN = 0.75, PARTICLE_ZOOM_MAX = 1.8;
// The inflow arrows reach their gap plus their length outside the city's upwind side, ARROW_HEIGHT_M up: the opening
// view frames the city and, on that side and at that height, 6 m more than the arrows.
const FRAME_MARGIN_M = ARROW_GAP_M + ARROW_LENGTH_M + 6;
// the camera: field of view (degrees), near and far planes (m); fog from FOG_NEAR_M to FOG_FAR_M away
const FOV_DEG = 36, NEAR_M = 2, FAR_M = 30000;
const FOG_NEAR_M = 9000, FOG_FAR_M = 26000;
// the view before the first city is framed (frameCity sets the real one)
const START_DISTANCE_M = 2500, START_AZIMUTH_DEG = -60, START_ELEVATION_DEG = 38;
// the opening view of a city: the camera's direction, the box framed (up to FRAME_TOP_M, about the flight band's
// top) and its margin (a share of the free area on each side)
const OPENING_AZIMUTH_DEG = -90, OPENING_ELEVATION_DEG = 48, FRAME_TOP_M = 120, FRAME_FIT_MARGIN = 0.02;
// fitProjected: the distance found by bisection within this range (m), in this many halvings, in this many rounds
const FIT_DISTANCE_MIN_M = 50, FIT_DISTANCE_MAX_M = 40000, FIT_HALVINGS = 40, FIT_ROUNDS = 3;
const CAMERA_ANIMATION_MS = 600;       // an eased setInsets
const VIEW_SWITCH_MS = 600;            // the whole city to the followed drone (followDrone) and back (resetView)
// the drone is gone at least this long (s on screen) at the goal before it flies again, and until its route has
// faded in whole (DRONE_TRAIL past the goal)
const DRONE_PAUSE_S = 0.6;
const NO_CUT_M = 1e9;                  // a route cut there is drawn whole
// following the drone (followDrone): the distance it is framed at (m) and its elevation (degrees: a low, near level
// view of the flight), from the view's own azimuth
const FOLLOW_DISTANCE_M = 800, FOLLOW_ELEVATION_DEG = OPENING_ELEVATION_DEG;     // as the whole city's view
// following, the drone somewhere new (another route, a new lap, new results): the view dashes there in DASH_MS (so
// quick that it reads as a glance, not a flight), pulled back by up to DASH_LIFT of its distance halfway for a way of
// DASH_LIFT_AT x that distance or more (less sweeps across the screen)
const DASH_MS = 200, DASH_LIFT = 0.3, DASH_LIFT_AT = 1.5;
const ROUTE_FADE_OUT_S = 0.2;          // before each flight the flown route fades out over this long (s on screen)
/** The camera animations' ease: share e (0..1) of the way at share t (0..1) of the time. */
const easeInOut = (t) => 0.5 - 0.5 * Math.cos(Math.PI * t);

/** The 3-D scene drawn on `canvas` (see the file header). */
export class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.setCursor("grab");
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.setPixelRatio(pixelRatio());
    // the buildings' shadows are drawn once per city (setCity), not every frame: the city does not move
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.shadowMap.autoUpdate = false;
    canvas.addEventListener("webglcontextrestored", () => { this.renderer.shadowMap.needsUpdate = true; });
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(BACKGROUND);
    this.scene.fog = new THREE.Fog(BACKGROUND, FOG_NEAR_M, FOG_FAR_M);
    this.camera = new THREE.PerspectiveCamera(FOV_DEG, 1, NEAR_M, FAR_M);
    this.camera.up.set(0, 0, 1);
    this.target = new THREE.Vector3();
    this.distance = START_DISTANCE_M;
    this.azimuth = START_AZIMUTH_DEG;      // degrees, from +x counter-clockwise, camera position
    this.elevation = START_ELEVATION_DEG;
    this.homeView = null;              // the opening view of the city (frameCity)
    this.userMoved = false;            // the user has moved the camera since the opening view
    this.animation = null;             // the camera animation running (animateTo), or null
    this.insets = { left: 0, right: 0, top: 0, bottom: 0 };
    this.insetsWanted = { ...this.insets };   // where an eased setInsets is going
    // the canvas's size in CSS pixels as of the last resize (read there, not in every frame: a read can make the page
    // lay itself out first)
    this.size = { width: 1, height: 1 };
    // Lighting: the hemisphere and sun intensities render roofs near white and lit and shaded walls in two distinct
    // light greys.  The sun comes along the diagonal (-1, -1), so the x- and y-facing 4 m steps of walls facing it get
    // the same light.
    this.scene.add(new THREE.HemisphereLight(SKY_LIGHT, GROUND_LIGHT, HEMISPHERE_INTENSITY));
    const sun = new THREE.DirectionalLight(SUN_LIGHT, SUN_INTENSITY);
    sun.position.set(-1, -1, 2.2);
    sun.castShadow = true;
    sun.shadow.mapSize.set(SHADOW_MAP_PX, SHADOW_MAP_PX);
    sun.shadow.normalBias = SHADOW_NORMAL_BIAS;
    this.sun = sun;
    this.scene.add(sun, sun.target);
    this.palette = paletteTexture();
    this.city = null;
    this.objects = {};                 // the scene's named objects (add / remove): name -> three.js object
    this.routes = new Map();
    this.routeTime = { value: 0 };     // the flight time shown on the routes (s), advanced by ROUTE_LAPSE per second
    this.selectedRoute = null;
    this.hiddenRoutes = new Set();     // the routes not drawn (their check boxes; showRoutes)
    this.routesStale = false;          // the routes are to be built anew at the end of this task (selectRoute)
    this.frameWork = new Map();        // name -> work for the start of the next drawn frame (beforeFrame)
    this.drone = null;                 // the drone's flight (flyDrone): { id, flight, time (s of flight) }, or null
    this.flying = false;               // a flight is on (flyDrone .. stopDrone), also with the drone away (hideDrone)
    // the view follows the drone (followDrone): { catching: the ease onto it { from, fromDistance, started } or null,
    // last: the drone's place at the last frame followed, or null; dash: the quick move to its new place (trackDrone)
    // { from, base, started, ms, lift } or null }, or null
    this.follow = null;
    // the visitor wants the view to follow the drone's flights: on at first (the opening flight is followed); only a
    // click on the drone (or the middle button) changes it, so a flight stopped and started again is followed again
    this.followWanted = true;
    this.followPending = false;        // a flight started: the view follows once its drone appears (flyDrone)
    this.onLap = null;                 // called when a flight has reached the goal and its route is whole (app)
    this.onUnfollow = null;            // called when the visitor stops the view following the drone (app)
    // the flown route (materials.js cutAhead): the drone's place along it (m), the trail behind it, the share shown
    this.routeCut = { at: { value: NO_CUT_M }, gap: { value: DRONE_TRAIL.gapM }, fade: { value: DRONE_TRAIL.fadeM },
                      show: { value: 1 } };
    this.pins = null;
    this.hoverPin = null;
    this.hoverDrone = null;            // the pointer [x, y] over the flying drone (highlightDrone), or null
    // the start pin's hint (showDragHint): asked for, over, its time left (ms) and when that was last counted
    this.dragHint = { armed: false, done: false, left: DRAG_HINT_MS, countedAt: null };
    this.droneAnnounce = { from: Infinity, until: -Infinity };   // the drone's hint shown by itself (announceDrone)
    this.pinZooms = { start: 1, goal: 1 };   // each pin's size on screen as a multiple of its smallest (sizePinButtons)
    this.corridorLineVisible = true;
    this.inflowKey = null;             // what the inflow arrows were built for (setInflowArrows)
    this.inflowTheta = null;           // their wind direction (radians), null before the first
    this.shaderAnchors = new Map();
    this.listeners = {};
    this.edgeLabels = new EdgeLabels(this);
    attachInteraction(this);
    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
    this.startLoop();
  }

  /**
   * The frame loop at 30 frames per second (ANIMATION_FRAME_MS, timed by the refreshes' own times); an error in one
   * frame is reported once and does not stop it.
   */
  startLoop() {
    let last = performance.now(), refreshed = null, due = -Infinity, tickError = false, renderError = false;
    const intervals = [];
    const loop = (time) => {
      requestAnimationFrame(loop);       // asked for first, so an error in this frame cannot stop the loop
      if (refreshed !== null) {
        intervals.push(time - refreshed);
        if (intervals.length > REFRESH_WINDOW) intervals.shift();
      }
      refreshed = time;
      // the refresh nearest the due time draws (from DUE_REACH of a refresh interval before it); one on time keeps the
      // schedule, a later one (after a skipped refresh, a long task, a hidden tab) starts it again from itself, so
      // two frames never come close together
      const reach = intervals.length ? DUE_REACH * Math.min(...intervals) : 0;
      if (time < due - reach) return;
      due = time - due > reach ? time + ANIMATION_FRAME_MS : due + ANIMATION_FRAME_MS;
      const delta = Math.min(MAX_STEP_S, Math.max(0, time - last) / 1000);
      last = time;
      try {
        this.tick(delta);
      } catch (error) {
        if (!tickError) console.error("render loop:", error);
        tickError = true;
      }
      try {
        this.renderer.render(this.scene, this.camera);
      } catch (error) {
        if (!renderError) console.error("render:", error);
        renderError = true;
      }
      this.keepPrograms();
    };
    requestAnimationFrame(loop);
  }

  /**
   * Every compiled shader program is kept for the life of the page: three.js deletes a program when its last material
   * is disposed, so each mission would compile some again, and reading the compile result is a synchronous call that
   * can wait seconds for the GPU while it runs the wind prediction (about ten programs).
   */
  keepPrograms() {
    for (const p of this.renderer.info.programs) {
      if (!p.keptForever) {
        p.keptForever = true;
        p.usedTimes++;
      }
    }
  }

  /**
   * Compile, without holding up the page, the shader programs that would otherwise be compiled in the frame that first
   * draws their layer (a first vertical slice, particle plane or flight would wait for the compiler there): hidden
   * samples of those layers, made as the real ones are, their programs started (renderer.compile: the browser compiles
   * them in the background, nothing here waits for the result) and kept, the samples removed.  For the page, once, when
   * it is idle after its first results; a failure only leaves the compiling to the first use.
   */
  async warmShaders() {
    const plane = { origin: [0, 0], u: [1, 0], s0: 0, s1: 1, z0: 0, z1: 1 };
    const route = { colour: BASE_EDGE, points: [[0, 0, 0], [1, 0, 0], [2, 0, 0]], speeds: [1, 1, 1] };
    const samples = new THREE.Group();
    samples.add(verticalSliceGroup(new Float32Array(8), 2, 2, plane, [0, 1], this.palette),
                planeFrameMesh([[0, 0, 0], [1, 0, 0], [1, 1, 0]]), droneGroup(BASE_EDGE),
                ...routeMeshes(route, true, 1, this.routeTime, this.routeCut));
    samples.visible = false;
    this.add("shaderSamples", samples);
    try {
      // one sample at a time, a turn for the page between them: each start is a short task of its own
      for (const sample of samples.children) {
        this.renderer.compile(sample, this.camera, this.scene);
        await yieldTurn();
      }
      this.keepPrograms();
    } catch (error) {
      console.warn("shader warm-up:", error.message);
    }
    this.remove("shaderSamples");
  }

  /**
   * Run `work` at the start of the next drawn frame, before it is drawn; asked for again under the same `name` before
   * then, only the latest runs (a dragged slider changes more often than frames are drawn: each frame shows its last).
   */
  beforeFrame(name, work) { this.frameWork.set(name, work); }

  /** Call `handler` on every event `name`: "camera" (the view moved), "pinDragStart", "pinDrag", "pinDragEnd". */
  on(name, handler) { (this.listeners[name] ??= []).push(handler); }

  /** Call the handlers of event `name` (view modules only). */
  emit(name, ...args) { for (const handler of this.listeners[name] ?? []) handler(...args); }

  // ---------- camera ----------
  /**
   * Screen space taken by the floating cards and bars (CSS pixels).  eased: the view slides there over
   * CAMERA_ANIMATION_MS (as animateTo; at once with reduced motion).  The same insets again change nothing (a running
   * slide goes on).
   */
  setInsets(left, right, top, bottom, eased = false) {
    const to = { left, right, top, bottom };
    if (Object.keys(to).every((k) => to[k] === this.insetsWanted[k])) return;
    this.insetsWanted = to;
    const ms = eased && !reducedMotion() ? CAMERA_ANIMATION_MS : 0;
    const from = { ...this.insets }, started = performance.now();
    const step = () => {
      if (this.insetsWanted !== to) return;            // newer insets meanwhile
      const t = ms > 0 ? Math.min(1, (performance.now() - started) / ms) : 1, e = easeInOut(t);
      for (const k of Object.keys(to)) this.insets[k] = t < 1 ? from[k] + (to[k] - from[k]) * e : to[k];
      this.resize();
      if (t < 1) requestAnimationFrame(step);
    };
    if (ms > 0) requestAnimationFrame(step);
    else step();
  }

  /** The canvas changed size or the insets changed: the renderer, the camera and the view offset follow. */
  resize() {
    const width = this.canvas.clientWidth || 1, height = this.canvas.clientHeight || 1;
    this.size = { width, height };
    this.renderer.setSize(width, height, false);
    this.camera.aspect = width / height;
    // the orbit target sits in the middle of the free area, not of the canvas
    const { left, right, top, bottom } = this.insets;
    const sx = (left - right) / 2, sy = (top - bottom) / 2;
    if (Math.abs(sx) > 0.5 || Math.abs(sy) > 0.5) this.camera.setViewOffset(width, height, -sx, -sy, width, height);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
    const resolution = this.resolution();
    for (const object of Object.values(this.objects)) {
      object.traverse?.((child) => child.material?.uniforms?.resolution?.value.copy(resolution));
    }
    // the opening view, if the user has not moved it, is fitted again to the new size (the city stays in the free area)
    if (this.homeView && !this.userMoved && !this.animation && this.city) {
      const { azimuth, elevation } = this.homeView;
      if (azimuth === this.azimuth && elevation === this.elevation) this.frameCity({ azimuth, elevation });
    }
    this.emit("camera");
  }

  /** Half the drawing buffer size (the particle ribbons' screen scale). */
  resolution() {
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    return new THREE.Vector2(size.x / 2, size.y / 2);
  }

  /** The camera at the current target, distance and angles, without a "camera" event (fitting tries views). */
  placeCamera() {
    const az = this.azimuth * Math.PI / 180, el = this.elevation * Math.PI / 180;
    this.camera.position.set(
      this.target.x + this.distance * Math.cos(el) * Math.cos(az),
      this.target.y + this.distance * Math.cos(el) * Math.sin(az),
      this.target.z + this.distance * Math.sin(el));
    this.camera.lookAt(this.target);
    this.camera.updateMatrixWorld();
  }

  /** The camera placed (placeCamera), and the "camera" event. */
  updateCamera() {
    this.placeCamera();
    this.emit("camera");
  }

  /**
   * The opening view of a city, independent of the mission: the whole city centred in the free area, square to the
   * screen (seen from the city's -y side, map x to the right), OPENING_ELEVATION_DEG up so the building heights still
   * read, with FRAME_FIT_MARGIN around it.
   */
  frameCity({ azimuth = OPENING_AZIMUTH_DEG, elevation = OPENING_ELEVATION_DEG } = {}) {
    if (!this.city) return;
    this.animation = null;               // a camera ease running (resetView) ends: this view instead
    this.homeView = this.openingView(azimuth, elevation);
    this.azimuth = azimuth; this.elevation = elevation;
    this.target.copy(this.homeView.target);
    this.distance = this.homeView.distance;
    this.userMoved = false;
    this.updateCamera();
  }

  /**
   * The opening view from `azimuth`, `elevation` (frameCity): the city up to FRAME_TOP_M, and its plate moved upwind
   * by FRAME_MARGIN_M at the inflow arrows' height (all round before they are first shown).  Returns
   * { target, distance, azimuth, elevation }; the camera stays as it is.
   */
  openingView(azimuth, elevation) {
    const sx = this.city.nx * CELL_M, sy = this.city.ny * CELL_M, e = FRAME_MARGIN_M, theta = this.inflowTheta;
    const plate = [[0, 0], [sx, 0], [0, sy], [sx, sy]], points = [];
    for (const [x, y] of plate) points.push([x, y, 0], [x, y, FRAME_TOP_M]);
    if (theta === null) {
      for (const [x, y] of plate) points.push([x ? x + e : -e, y ? y + e : -e, ARROW_HEIGHT_M]);
    } else {
      for (const [x, y] of plate) points.push([x - e * Math.cos(theta), y - e * Math.sin(theta), ARROW_HEIGHT_M]);
    }
    const fit = this.fitProjected(points, azimuth, elevation, FRAME_FIT_MARGIN);
    return { target: new THREE.Vector3(...fit.target), distance: fit.distance, azimuth, elevation };
  }

  /**
   * Fit `points` (metres) into the free area with `margin` (0.1 = 10 % each side) by projecting them through the real
   * camera: binary search on the distance, then the target is moved so their projected box is centred in the free
   * area; FIT_ROUNDS rounds.  Keeps azimuth and elevation, leaves the camera as it was.  Returns { target, distance }.
   */
  fitProjected(points, azimuth, elevation, margin) {
    const save = this.currentView();
    const { width: W, height: H } = this.size;
    const { left, right, top, bottom } = this.insets;
    const fw = W - left - right, fh = H - top - bottom, cx = left + fw / 2, cy = top + fh / 2;
    const box = () => {
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (const [x, y, z] of points) {
        const [px, py] = project(this.camera, x, y, z, W, H);
        x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
      }
      return [x0, x1, y0, y1];
    };
    this.azimuth = azimuth; this.elevation = elevation;
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]), zs = points.map((p) => p[2]);
    const middle = (values) => (Math.min(...values) + Math.max(...values)) / 2;
    this.target.set(middle(xs), middle(ys), middle(zs));
    for (let round = 0; round < FIT_ROUNDS; round++) {
      let lo = FIT_DISTANCE_MIN_M, hi = FIT_DISTANCE_MAX_M;
      for (let k = 0; k < FIT_HALVINGS; k++) {
        this.distance = Math.sqrt(lo * hi);
        this.placeCamera();
        const [x0, x1, y0, y1] = box();
        const fits = x1 - x0 <= fw * (1 - 2 * margin) && y1 - y0 <= fh * (1 - 2 * margin);
        if (fits) {
          hi = this.distance;
        } else {
          lo = this.distance;
        }
      }
      this.distance = hi;
      this.placeCamera();
      const [x0, x1, y0, y1] = box();
      const dx = (x0 + x1) / 2 - cx, dy = (y0 + y1) / 2 - cy;
      const mpp = this.metresPerPixel(), az = this.azimuth * Math.PI / 180;
      const rightV = new THREE.Vector3(-Math.sin(az), Math.cos(az), 0);
      const fwd = new THREE.Vector3(-Math.cos(az), -Math.sin(az), 0);
      // fwd points away from the camera (screen up): a box below the centre (dy > 0) needs the target moved toward
      // the camera, i.e. along -fwd
      const lift = Math.max(0.2, Math.sin(this.elevation * Math.PI / 180));
      this.target.addScaledVector(rightV, dx * mpp).addScaledVector(fwd, -dy * mpp / lift);
    }
    const result = { target: this.target.toArray(), distance: this.distance };
    Object.assign(this, { distance: save.distance, azimuth: save.azimuth, elevation: save.elevation });
    this.target.copy(save.target);
    this.placeCamera();
    return result;
  }

  /** Ease the camera to a new view over VIEW_SWITCH_MS (with reduced motion: there at the next frame). */
  animateTo(target, distance, azimuth, elevation) {
    const from = this.currentView();
    const turn = ((azimuth - from.azimuth) % 360 + 540) % 360 - 180;
    const started = performance.now();
    const ms = reducedMotion() ? 0 : VIEW_SWITCH_MS;
    const token = this.animation = {};
    const step = () => {
      if (this.animation !== token) return;
      const t = ms > 0 ? Math.min(1, (performance.now() - started) / ms) : 1;
      const e = easeInOut(t);
      this.target.set(from.target.x + (target[0] - from.target.x) * e, from.target.y + (target[1] - from.target.y) * e,
                      from.target.z + (target[2] - from.target.z) * e);
      this.distance = from.distance + (distance - from.distance) * e;
      this.azimuth = from.azimuth + turn * e;
      this.elevation = from.elevation + (elevation - from.elevation) * e;
      this.updateCamera();
      if (t < 1) requestAnimationFrame(step);
      else this.animation = null;      // done: a resize may fit the opening view again
    };
    requestAnimationFrame(step);
  }

  /** The current view: target (a copy), distance, azimuth, elevation. */
  currentView() {
    return { target: this.target.clone(), distance: this.distance, azimuth: this.azimuth, elevation: this.elevation };
  }

  /**
   * A middle click (a double tap, 0 or Home): while the drone flies, the view follows it, and the next one ends that;
   * otherwise back to the opening view.
   */
  returnView() {
    this.endDroneAnnounce();                       // the drone clicked (or the view reset): its own hint is done
    if (this.flying && !this.follow) {
      this.followWanted = true;
      this.followDrone();
      return;
    }
    if (this.follow) {
      this.followWanted = false;
      this.onUnfollow?.();
    }
    this.follow = null;
    this.resetView();
  }

  /** Back to the city's opening view (for the wind direction now: the inflow arrows' side). */
  resetView() {
    if (!this.homeView) return;
    this.homeView = this.openingView(this.homeView.azimuth, this.homeView.elevation);
    const h = this.homeView;
    this.userMoved = false;
    this.animateTo(h.target.toArray(), h.distance, h.azimuth, h.elevation);
  }

  /** The camera as plain numbers: azimuth and elevation (degrees), distance (m). */
  cameraState() {
    return { azimuth: this.azimuth, elevation: this.elevation, distance: this.distance };
  }

  /** Screen position (CSS pixels) of a point in metres. */
  toScreen([x, y, z]) {
    const r = this.canvas.getBoundingClientRect(), [px, py] = project(this.camera, x, y, z, r.width, r.height);
    return [r.left + px, r.top + py];
  }

  /** The camera ray through the page point (clientX, clientY). */
  ray(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - rect.left) / rect.width) * 2 - 1,
                                  -((clientY - rect.top) / rect.height) * 2 + 1);
    const ray = new THREE.Raycaster();
    ray.setFromCamera(ndc, this.camera);
    return ray;
  }

  /** City metres [x, y] of the point at `height` under a screen position, or null. */
  groundPoint(clientX, clientY, height) {
    const point = new THREE.Vector3();
    const plane = new THREE.Plane(new THREE.Vector3(0, 0, 1), -height);
    return this.ray(clientX, clientY).ray.intersectPlane(plane, point) ? [point.x, point.y] : null;
  }

  /**
   * The view centre stays over the city (otherwise a few drags while zoomed out could take the view kilometres away
   * with nothing to bring it back).
   */
  keepTargetOnCity() {
    if (!this.city) return;
    this.target.x = Math.min(this.city.nx * CELL_M, Math.max(0, this.target.x));
    this.target.y = Math.min(this.city.ny * CELL_M, Math.max(0, this.target.y));
  }

  /** Metres per screen pixel at the orbit target. */
  metresPerPixel() {
    return metresPerPixelAt(this.camera, this.distance, this.size.height);
  }

  /** The canvas cursor (CSS cursor name). */
  setCursor(cursor) { this.canvas.style.cursor = cursor; }

  /** The GPU's name as WebGL reports it (UNMASKED_RENDERER_WEBGL), or "" when the browser hides it. */
  rendererName() {
    const gl = this.renderer.getContext();
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    return ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : "";
  }

  // ---------- layers ----------
  /**
   * Take a layer out of the scene and free what it owns on the GPU: geometry, material and the textures it alone uses
   * (palettes are shared and kept).  For a custom shader, the first compiled material of each shader source is kept
   * (not disposed, its textures released) as an anchor: three.js numbers a shader source by the materials that use it,
   * so with the anchor alive every later material of that source reuses the compiled program instead of compiling a
   * new one (a compile can block the page while the GPU predicts the wind).
   */
  remove(name) {
    const object = this.objects[name];
    if (!object) return;
    this.scene.remove(object);
    object.traverse?.((child) => {
      if (!child.isSprite) child.geometry?.dispose();     // every sprite shares three.js's one sprite geometry
      const m = child.material;
      if (!m || m.userData?.shared) return;
      const own = (x) => x?.isTexture && !x.userData?.shared;
      const textures = [m.map, ...Object.values(m.uniforms ?? {}).map((u) => u.value)].filter(own);
      for (const texture of textures) texture.dispose();
      if (m.isShaderMaterial) {
        const key = `${m.vertexShader}\u0000${m.fragmentShader}`;
        // only a compiled material holds its source in three.js's shader cache
        if (!this.shaderAnchors.has(key) && this.renderer.properties.get(m).currentProgram) {
          for (const u of Object.values(m.uniforms ?? {})) if (own(u.value)) u.value = null;
          this.shaderAnchors.set(key, m);
          return;
        }
      }
      m.dispose?.();
    });
    delete this.objects[name];
  }

  /** Put `object` in the scene as layer `name` (replacing that layer).  remove / add / showLayer: view modules only. */
  add(name, object) {
    this.remove(name);
    this.objects[name] = object;
    this.scene.add(object);
    return object;
  }

  /** Whether layer `name` shows (view modules only). */
  showLayer(name, on) { if (this.objects[name]) this.objects[name].visible = on; }

  // ---------- the wind layers, for the page ----------
  /** Whether the particles show. */
  showParticles(on) { this.showLayer("particles", on); }

  /** Whether the slice shows. */
  showSlice(on) { this.showLayer("slice", on); }

  /** The particles leave the scene. */
  clearParticles() { this.remove("particles"); }

  /** The slice leaves the scene. */
  clearSlice() { this.remove("slice"); }

  /** The wind of the last computation leaves the scene: its particles and its slice. */
  clearWind() {
    this.clearParticles();
    this.clearSlice();
  }

  /** The prediction sweep leaves the scene. */
  clearSweep() { this.remove("sweep"); }

  /** A new city: its base plate with a 1 px outline, its edge labels and its buildings (city/buildings.js surfaces). */
  setCity(city, buildings) {
    this.city = city;
    const spanX = city.nx * CELL_M, spanY = city.ny * CELL_M;
    const group = new THREE.Group();
    const plate = new THREE.Mesh(new THREE.PlaneGeometry(spanX, spanY),
                                 new THREE.MeshBasicMaterial({ color: BASE_PLATE }));
    plate.position.set(spanX / 2, spanY / 2, GROUND_Z.plate);
    group.add(plate);
    const outline = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(
      [[0, 0], [spanX, 0], [spanX, spanY], [0, spanY]].map(([x, y]) => new THREE.Vector3(x, y, GROUND_Z.plateEdge))),
      new THREE.LineBasicMaterial({ color: BASE_EDGE }));
    group.add(outline);
    this.add("plate", group);
    this.add("edgeLabels", this.edgeLabels.build(city));
    this.edgeLabels.place();
    const mesh = new THREE.Mesh(buildingGeometry(buildings), buildingMaterial());
    mesh.castShadow = true;
    this.add("buildings", mesh);
    this.shadeCity(spanX, spanY);
  }

  /**
   * The buildings' shadows on the ground of a spanX x spanY m city: a faint shadow-only plane (the wind layers and the
   * routes stay as they are), the sun moved along its direction to frame the city, and the shadow map drawn once.
   */
  shadeCity(spanX, spanY) {
    const shadow = new THREE.Mesh(new THREE.PlaneGeometry(spanX, spanY),
      new THREE.ShadowMaterial({ color: SHADOW_COLOUR, opacity: SHADOW_OPACITY, depthWrite: false }));
    shadow.position.set(spanX / 2, spanY / 2, GROUND_Z.shadow);
    shadow.receiveShadow = true;
    this.add("groundShadow", shadow);
    const span = Math.max(spanX, spanY), camera = this.sun.shadow.camera;
    this.sun.target.position.set(spanX / 2, spanY / 2, 0);
    this.sun.position.set(spanX / 2 - span, spanY / 2 - span, 2.2 * span);   // the direction (-1, -1, 2.2) as before
    camera.left = camera.bottom = -0.8 * span;
    camera.right = camera.top = 0.8 * span;
    camera.near = 1;
    camera.far = 5 * span;
    camera.updateProjectionMatrix();
    this.renderer.shadowMap.needsUpdate = true;
  }

  /** Nothing of a mission is left: routes, the drone (its flight goes on), particles, slices, sweep, pins, corridor. */
  clearMission() {
    this.setRoutes([], null);
    this.hideDrone();
    this.clearWind();
    this.clearSweep();
    for (const name of [PIN_LAYERS.start, PIN_LAYERS.goal, "corridor"]) this.remove(name);
    this.pins = null;
  }

  /** The corridor (cell units): a 1 px ground line and its ground, lighter than the plate. */
  setCorridor(polygonCells) {
    const points = polygonCells.map(([ix, iy]) => new THREE.Vector2(cellCentreM(ix), cellCentreM(iy)));
    const group = new THREE.Group();
    const linePoints = points.map((p) => new THREE.Vector3(p.x, p.y, GROUND_Z.corridorLine));
    const outline = new THREE.BufferGeometry().setFromPoints(linePoints);
    const line = new THREE.LineLoop(outline,
      new THREE.LineBasicMaterial({ color: CORRIDOR_LINE, transparent: true, opacity: CORRIDOR_LINE_OPACITY,
                                    depthWrite: false }));
    line.renderOrder = RENDER_ORDER.corridorLine;
    line.visible = this.corridorLineVisible;
    group.add(line);
    const fill = new THREE.Mesh(new THREE.ShapeGeometry(new THREE.Shape(points)), new THREE.MeshBasicMaterial({
      color: CORRIDOR_GROUND, depthWrite: false }));
    fill.position.z = GROUND_Z.corridorGround;
    group.add(fill);
    this.add("corridor", group);
  }

  /** The corridor's ground line (setCorridor), or undefined. */
  corridorLine() { return this.objects.corridor?.children[0]; }

  /** Whether the corridor's outline shows (it is hidden under a slice); kept when the corridor is rebuilt. */
  setCorridorLineVisible(on) {
    this.corridorLineVisible = on;
    const line = this.corridorLine();
    if (line) line.visible = on;
  }

  /** Start and goal pins at their flight altitude [x, y, z]. */
  setPins(start, goal) {
    this.pins = { start, goal };
    for (const [which, position] of [["start", start], ["goal", goal]]) {
      this.add(PIN_LAYERS[which], makePin(which, position));
    }
    this.sizePinButtons();
  }

  /** Move the pin `which` ("start" | "goal") to (x, y) metres while it is dragged (labels keep clear of it there). */
  movePin(which, x, y) {
    if (!this.pins) return;                   // the mission was cleared meanwhile (another city)
    this.objects[PIN_LAYERS[which]]?.position.set(x, y, 0);
    this.pins[which] = [x, y, this.pins[which][2]];
  }

  /** The end point of a pin, holding its pin sprite (pins.js makePin). */
  pinHead(which) { return this.objects[PIN_LAYERS[which]].userData.head; }

  /**
   * The hovered drone: DRONE_HOVER_SCALE larger, and above it, a constant size on screen, what a click does ("Click to
   * follow", or "Click to stop following" while the view follows it); otherwise as ever, the hint hidden.
   */
  placeDroneHint() {
    const group = this.objects.drone, hint = this.objects.droneHint;
    if (!group || !hint) return;
    // the drone flown away from under a still pointer: no longer hovered
    if (this.hoverDrone && !this.droneUnder(...this.hoverDrone, DRONE_KEEP)) {
      this.hoverDrone = null;
      this.setCursor("grab");
    }
    const hovered = Boolean(this.hoverDrone) && group.visible, now = performance.now(), a = this.droneAnnounce;
    const announced = group.visible && now >= a.from && now < a.until;
    if (Number.isFinite(a.until) && now >= a.until) this.endDroneAnnounce();     // its time is up
    group.scale.setScalar(group.userData.scale * (hovered ? DRONE_HOVER_SCALE : 1));
    this.fadeHint(hint, hovered || announced);
    if (!hint.visible) return;
    hint.position.copy(group.position);
    const m = metresPerPixelAt(this.camera, this.camera.position.distanceTo(group.position), this.size.height);
    // sized as the pins and their hints (pins.js pinZoom: a size in the scene, kept between its smallest and largest
    // on screen), drawn as sharp as that needs
    const zoom = pinZoom(m), sharp = 2 ** Math.ceil(Math.log2(Math.max(4, zoom * pixelRatio())));
    if (this.follow) setHintText(hint, "Click to\nstop following", "camera-off", sharp);
    else setHintText(hint, "Click to\nfollow", "camera", sharp);
    const scale = zoom * hint.userData.px * m;
    hint.scale.set(scale * hint.userData.aspect, scale, 1);
    const rise = DRONE_HOVER_SCALE * DRONE_HINT_RISE * DRONE_SPAN_M / m - zoom * DRONE_HINT_SINK_PX;     // CSS px
    hint.center.y = -rise / (zoom * hint.userData.px);
  }

  /**
   * Whether the flying drone is under the pointer: within its span on screen (at least DRONE_HIT_MIN_PX), times
   * `widen` (DRONE_KEEP for a drone already hovered).
   */
  droneUnder(clientX, clientY, widen = 1) {
    const group = this.objects.drone;
    if (!this.flying || !group?.visible) return false;
    const rect = this.canvas.getBoundingClientRect(), p = group.getWorldPosition(new THREE.Vector3());
    const [px, py, depth] = project(this.camera, p.x, p.y, p.z, rect.width, rect.height);
    if (depth > 1) return false;
    const m = metresPerPixelAt(this.camera, this.camera.position.distanceTo(p), rect.height);
    const reach = widen * Math.max(DRONE_HIT_MIN_PX, DRONE_SPAN_M / m);
    return Math.hypot(clientX - rect.left - px, clientY - rect.top - py) <= reach;
  }

  /**
   * The hover highlight of the flying drone, the pointer at [x, y] (client pixels) over it, or null: DRONE_HOVER_SCALE
   * larger, with what a click does (placeDroneHint).
   */
  highlightDrone(at) { this.hoverDrone = at; }

  /** The pins' "Drag to move" hint ends now (a pin is being dragged): it fades out and is not shown again. */
  dismissDragHint() {
    this.dragHint.done = true;
  }

  /** "Drag to move" over the start pin from now until a pin is dragged or its time is up (once). */
  showDragHint() {
    this.dragHint.armed = true;
  }

  /** The drone's hint (what a click does) by itself, DRONE_ANNOUNCE_AFTER_MS to DRONE_ANNOUNCE_UNTIL_MS from now. */
  announceDrone() {
    const now = performance.now();
    this.droneAnnounce = { from: now + DRONE_ANNOUNCE_AFTER_MS, until: now + DRONE_ANNOUNCE_UNTIL_MS };
  }

  /** The drone's own hint (announceDrone) is over. */
  endDroneAnnounce() {
    this.droneAnnounce.until = -Infinity;
    this.showDragHint();                           // next: the start pin's hint
  }

  /**
   * Hint sprite `sprite` faded towards shown (on) or gone, HINT_FADE_MS for the whole way, from the last frame it was
   * faded; its opacity set, hidden at 0 (view modules only).
   */
  fadeHint(sprite, on) {
    const now = performance.now(), u = sprite.userData;
    const step = (now - (u.fadedAt ?? now)) / HINT_FADE_MS;
    u.fadedAt = now;
    u.shown = Math.min(1, Math.max(0, (u.shown ?? 0) + (on ? step : -step)));
    sprite.material.opacity = u.shown;
    sprite.visible = u.shown > 0;
  }

  /** The hover highlight of a pin: HOVER_SCALE times larger, HOVER_LIFT_PX higher. */
  highlightPin(which) {
    if (this.hoverPin === which) return;
    this.hoverPin = which;
    this.sizePinButtons();
  }

  /**
   * The pin whose head and point (PIN_HIT_MARGIN_PX around them) are under the pointer; never within POINT_KEEP_PX of
   * an end point.
   */
  pinUnder(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    let best = null, bestD = Infinity;
    for (const [which, name] of Object.entries(PIN_LAYERS)) {
      const head = this.objects[name]?.visible && this.objects[name].userData.head;
      if (!head) continue;
      const { x, y, z } = head.getWorldPosition(new THREE.Vector3());
      const [px, py, depth] = project(this.camera, x, y, z, rect.width, rect.height);
      if (depth > 1) continue;
      const mx = clientX - rect.left - px, my = clientY - rect.top - py;      // from the end point
      if (Math.hypot(mx, my) <= POINT_KEEP_PX) continue;
      const hovered = this.hoverPin === which, zoom = this.pinZooms[which];
      const k = zoom * (hovered ? HOVER_SCALE : 1), lift = zoom * (hovered ? HOVER_LIFT_PX : 0) + head.userData.raise;
      const { w, h, tip, gap } = head.userData.size;
      const top = -(lift + k * (gap + tip + h)), bottom = -(lift + k * gap);
      const dx = Math.max(0, Math.abs(mx) - k * w / 2), dy = Math.max(0, top - my, my - bottom);
      if (dx <= PIN_HIT_MARGIN_PX && dy <= PIN_HIT_MARGIN_PX && dx + dy < bestD) { bestD = dx + dy; best = which; }
    }
    return best;
  }

  /**
   * Size the pins (pins.js pinZoom): constant on screen at their smallest, larger with the scene when the view is
   * closer (and larger and lifted while hovered), each at its own distance, redrawn as sharp as that needs, raised so
   * the tip stays BEAD_CLEAR_PX above the bead, with their hint while it lasts (DRAG_HINT_MS from the first pins) and
   * over a hovered pin (above it as it grows and lifts).
   */
  sizePinButtons() {
    const { height } = this.size;
    // the start pin's hint: its time counted while the view is not on the drone
    const d = this.dragHint, now = performance.now();
    if (d.armed && !d.done && !this.follow && d.countedAt !== null) d.left -= now - d.countedAt;
    d.countedAt = now;
    if (d.left <= 0) d.done = true;
    const dragHint = d.armed && !d.done;
    const metresPerPx = (object) => metresPerPixelAt(this.camera,
      this.camera.position.distanceTo(object.getWorldPosition(new THREE.Vector3())), height);
    for (const [which, name] of Object.entries(PIN_LAYERS)) {
      const head = this.objects[name]?.userData.head;
      if (!head) continue;
      const { pin, hint: label } = head.userData, hovered = this.hoverPin === which, m = metresPerPx(head);
      const zoom = this.pinZooms[which] = pinZoom(m);
      // canvas pixels per CSS pixel: a power of two, at least the pixels it covers on screen
      const sharp = 2 ** Math.ceil(Math.log2(Math.max(4, zoom * HOVER_SCALE * pixelRatio())));
      if (head.userData.k !== sharp) drawPinAt(head, sharp);
      // the tip above the bead (the bead grows with the scene, the gap under the tip with the pin)
      const raise = head.userData.raise = Math.max(0, BEAD_RADIUS_M / m + BEAD_CLEAR_PX - zoom * pin.userData.gap);
      const k = zoom * (hovered ? HOVER_SCALE : 1), scale = k * pin.userData.px * m;
      pin.scale.set(scale * pin.userData.aspect, scale, 1);
      pin.center.y = -((hovered ? zoom * HOVER_LIFT_PX : 0) + raise) / (k * pin.userData.px);
      const labelScale = zoom * label.userData.px * m;
      label.scale.set(labelScale * label.userData.aspect, labelScale, 1);
      const grown = hovered ? zoom * (HOVER_LIFT_PX + (HOVER_SCALE - 1) * pin.userData.px) : 0;   // CSS px
      label.center.y = -label.userData.above - (raise + grown) / (zoom * label.userData.px);
      this.fadeHint(label, hovered || (which === "start" && dragHint));
    }
  }

  // ---------- routes ----------
  /**
   * The routes [{ id, colour, points, speeds }] with route `selected` highlighted, the ids in `hidden` not drawn.  The
   * same routes again (the same ids and point arrays) only change the selection and which are drawn.
   */
  showRoutes(routes, selected, hidden) {
    this.hiddenRoutes = new Set(hidden);
    const same = this.routes.size === routes.length
      && routes.every((r) => this.routes.get(r.id)?.userData.route.points === r.points);
    if (same) this.selectRoute(selected);
    else this.setRoutes(routes, selected);
  }

  /** No routes in the scene. */
  clearRoutes() { this.setRoutes([], null); }

  /** The routes built anew (showRoutes), route `selected` highlighted. */
  setRoutes(routes, selected) {
    for (const name of [...this.routes.keys()]) this.remove(`route:${name}`);
    this.routes.clear();
    for (const route of routes) {
      if (!route.points || route.points.length < 2) continue;
      const group = new THREE.Group();
      group.userData = { route };
      this.routes.set(route.id, group);
      this.add(`route:${route.id}`, group);
    }
    if (!this.routes.size) { this.selectedRoute = null; return; }
    this.selectRoute(selected);
  }

  /**
   * Route `id` selected: thicker and in front where routes overlap.  Every route is built anew at the end of the
   * current task (a microtask), once however often this is asked for in it (new routes and a flight starting on them).
   */
  selectRoute(id) {
    this.selectedRoute = this.routes.has(id) ? id : null;
    if (this.routesStale) return;
    this.routesStale = true;
    queueMicrotask(() => this.buildRoutes());
  }

  /** Every route built anew (selectRoute). */
  buildRoutes() {
    this.routesStale = false;
    const casing = this.metresPerPixel();         // the white casing about 1 px wide on screen
    for (const [key, group] of this.routes) {
      for (const child of [...group.children]) {
        group.remove(child);
        child.geometry.dispose();
        child.material.dispose();
      }
      if (this.hiddenRoutes.has(key)) continue;
      const cut = this.drone?.id === key ? this.routeCut : null;
      group.add(...routeMeshes(group.userData.route, key === this.selectedRoute, casing, this.routeTime, cut));
    }
  }

  // ---------- the drone ----------
  /**
   * The drone flies along route `id` (shown, view/routes.js) from its start, at the time its arrows show (ROUTE_LAPSE x
   * the planned flight), again and again, and the route shows only behind it: flight = { points, speeds, thrusts,
   * colour } (view/drone.js droneFlight).  A new flight starts over; fadeOut: the route is on screen whole, so it
   * fades out first (else the drone starts at once and the route is never shown whole).
   */
  flyDrone(id, flight, fadeOut) {
    const group = droneGroup(flight.colour);
    group.userData.scale = group.scale.x;          // its scale unhovered
    this.add("drone", group);
    this.add("droneHint", hintSprite(""));
    this.flying = true;
    // fading: s on screen left of the route's fade-out before the drone starts
    this.drone = { id, flight: droneFlight(flight), time: 0, fading: fadeOut ? ROUTE_FADE_OUT_S : 0 };
    this.routeCut.at.value = NO_CUT_M;
    this.routeCut.show.value = 1;
    // the visitor wants flights followed: the view eases onto the drone once it appears (stepDrone)
    this.followPending = this.followWanted && !this.follow;
    this.stepDrone(0);
    this.selectRoute(this.selectedRoute);          // the routes built anew: the flown one cut at the drone
  }

  /** The drone leaves the scene for now (a new computation): its route stays as drawn until the routes are replaced. */
  hideDrone() {
    this.drone = null;
    this.remove("drone");
    this.remove("droneHint");
  }

  /** The flight ends: the drone leaves the scene, the view no longer follows it, its route is drawn whole again. */
  stopDrone() {
    const flown = this.drone, followed = Boolean(this.follow);
    this.drone = null;
    this.flying = false;
    this.follow = null;
    this.followPending = false;
    if (followed) this.resetView();                // the view on the drone has nothing to show now: the whole city
    this.remove("drone");
    this.remove("droneHint");
    this.routeCut.at.value = NO_CUT_M;
    this.routeCut.show.value = 1;
    if (flown) this.selectRoute(this.selectedRoute);
  }

  /**
   * The drone `delta` s on: its route fading out before a flight; the drone on its route; at the goal gone until its
   * route is whole, then the route fades out and the drone starts again (view modules only).
   */
  stepDrone(delta) {
    const drone = this.drone, { flight } = drone, group = this.objects.drone;
    if (drone.fading > 0) {
      drone.fading = Math.max(0, drone.fading - delta);
      this.routeCut.show.value = drone.fading / ROUTE_FADE_OUT_S;
      group.visible = false;
      if (drone.fading > 0) return;
      drone.time = 0;                    // faded out: the flight starts
      this.routeCut.show.value = 1;
    } else if (this.followPending) {     // the flight's first frame with its drone: the view eases onto it
      this.followPending = false;
      this.followDrone();
    } else {
      drone.time += delta * ROUTE_LAPSE;
      const trail = flight.alongAt(drone.time) > flight.length + DRONE_TRAIL.gapM + DRONE_TRAIL.fadeM;
      if (trail && drone.time > flight.duration + DRONE_PAUSE_S * ROUTE_LAPSE) {
        this.onLap?.();                  // a flight done (the page's opening ends there); it may stop the drone
        if (this.drone !== drone) return;
        drone.fading = ROUTE_FADE_OUT_S;  // the route whole: it fades out next
        return;
      }
    }
    group.visible = drone.time <= flight.duration;
    // the route up to the drone, fading in behind it; past the goal the fade goes on until the route is solid
    this.routeCut.at.value = flight.alongAt(drone.time);
    if (group.visible) flight.place(group, drone.time);
  }

  /**
   * The view follows the drone from now on: its centre eased onto the drone, its distance to FOLLOW_DISTANCE_M and its
   * elevation to FOLLOW_ELEVATION_DEG over VIEW_SWITCH_MS, then moved with the drone; the user moves, turns and
   * zooms it as ever (view/interaction.js) and it keeps that place relative to the drone.
   */
  followDrone() {
    this.animation = null;
    this.userMoved = true;
    this.follow = { catching: { from: this.target.clone(), fromDistance: this.distance, fromElevation: this.elevation,
                                started: reducedMotion() ? -Infinity : performance.now() },
                    last: null, dash: null };
  }

  /** The user moves the view: the ease onto the drone (followDrone) ends where it is; following goes on. */
  endFollowEase() {
    if (this.follow?.catching) this.follow.catching = null;
  }

  /**
   * Each frame while following: the view moved by as much as the drone moved since the last frame (it keeps the place
   * the user gave it relative to the drone), eased onto the drone after followDrone; when the drone goes somewhere new
   * (another route, a new lap, new results) the view dashes there (dashTo), towards the flight's start while the drone
   * is away before it (the route fading out), towards the drone once it is back (view modules only).
   */
  trackDrone() {
    const group = this.objects.drone, follow = this.follow, drone = this.drone;
    const shown = drone && group?.visible;
    if (!shown) {
      follow.catching = null;              // the ease onto the drone cut short: a dash takes over when it is back
      // the drone gone before a flight (its route fading out): the view sets off for the flight's start now
      if (drone && drone.fading > 0 && !follow.dash) this.dashTo(this.flightStart(drone));
      follow.last = null;
      if (follow.dash) this.stepDash(this.flightStart(drone));
      return;
    }
    const at = group.position, c = follow.catching;
    if (c) {
      const t = Math.min(1, (performance.now() - c.started) / VIEW_SWITCH_MS), e = easeInOut(t);
      this.target.lerpVectors(c.from, at, e);
      this.distance = c.fromDistance + (FOLLOW_DISTANCE_M - c.fromDistance) * e;
      this.elevation = c.fromElevation + (FOLLOW_ELEVATION_DEG - c.fromElevation) * e;
      if (t >= 1) follow.catching = null;
    } else if (follow.dash) {
      this.stepDash(at);
    } else if (follow.last) {
      this.target.add(at).sub(follow.last);
    } else {
      this.dashTo(at);                     // back after a while (a new lap, new results): a dash there
      this.stepDash(at);
    }
    follow.last = (follow.last ?? new THREE.Vector3()).copy(at);
    this.placeCamera();
  }

  /** Where flight `drone` starts (its drone's place at its time 0), in a vector kept for this. */
  flightStart(drone) {
    this.startProbe ??= new THREE.Object3D();
    if (drone) drone.flight.place(this.startProbe, 0);
    return this.startProbe.position;
  }

  /** A dash of the followed view to `goal` (DASH_MS; its lift set by how far it is: see DASH_LIFT). */
  dashTo(goal) {
    const way = Math.hypot(goal.x - this.target.x, goal.y - this.target.y);
    const ms = reducedMotion() ? 0 : DASH_MS;
    const lift = DASH_LIFT * Math.min(1, way / (DASH_LIFT_AT * this.distance));
    this.follow.dash = { from: this.target.clone(), base: this.distance, started: performance.now(), ms, lift };
  }

  /** The dash's frame towards `goal` (it may move): eased in and out, pulled back halfway; done at its end. */
  stepDash(goal) {
    const d = this.follow.dash, t = d.ms > 0 ? Math.min(1, (performance.now() - d.started) / d.ms) : 1;
    this.target.lerpVectors(d.from, goal, easeInOut(t));
    this.distance = d.base * (1 + d.lift * Math.sin(Math.PI * t));
    if (t >= 1) {
      this.follow.dash = null;
      this.distance = d.base;
    }
    this.placeCamera();
  }

  /**
   * One drawn frame after `delta` seconds: the page's work for it (beforeFrame), the route marks, the drone, the edge
   * labels, the particles.
   */
  tick(delta) {
    const work = [...this.frameWork.values()];
    this.frameWork.clear();
    for (const run of work) {
      try {
        run();
      } catch (error) {                // reported each time, and the frame goes on
        console.error(error);
      }
    }
    this.routeTime.value += delta * ROUTE_LAPSE;
    if (this.drone) this.stepDrone(delta);
    this.placeDroneHint();
    if (this.follow) this.trackDrone();
    this.edgeLabels.place(delta);
    const particles = this.objects.particles;
    if (particles?.userData.particles) this.stepParticles(particles, delta);
    this.sizePinButtons();
  }

  // ---------- particles ----------
  /**
   * Wind particles (particles/system.js), computed in a worker (particles/worker.js) so the page never freezes:
   * `system` = { id, count, arrays, ribbons, step } where arrays = the first trail-point arrays [points (position,
   * speed), alphas] from the worker.  Each frame the viewer passes the time step and a spare set of arrays to
   * step(delta, arrays); the worker fills them and sends them back (transferred, no copy); the viewer swaps them in
   * (receiveParticles).  The ribbons (their arrays from the worker too): wind_layers.js particleMesh.
   */
  setParticles(system, range) {
    const mesh = particleMesh(system, range, this.palette, this.resolution(), this.renderer.getPixelRatio());
    mesh.userData.particles = system;
    // the frames' bookkeeping: the arrays for the next step, a step being computed, the time not yet stepped
    mesh.userData.steps = { spare: system.arrays.map((a) => new Float32Array(a.length)), pending: false, acc: 0 };
    if (this.objects.particles) mesh.visible = this.objects.particles.visible;
    this.add("particles", mesh);
    this.stepParticles(mesh, 0);
  }

  /** Trail-point arrays from the particle worker: swapped into the textures; the ones shown so far become the spare. */
  receiveParticles(id, arrays) {
    const mesh = this.objects.particles, sys = mesh?.userData.particles;
    if (!sys || sys.id !== id) return;
    const u = mesh.material.uniforms, steps = mesh.userData.steps;
    steps.spare = [u.points.value, u.alphas.value].map((t, q) => {
      const old = t.image.data;
      t.image.data = arrays[q]; t.needsUpdate = true;
      return old;
    });
    steps.pending = false;
  }

  /** Ask the worker for the next particle step (one at a time) and follow the zoom with the ribbon width. */
  stepParticles(mesh, delta) {
    const sys = mesh.userData.particles, steps = mesh.userData.steps;
    // the step is computed in the worker; while it works, the time adds up (at most MAX_STEP_S per step)
    if (mesh.visible) {
      steps.acc = Math.min(MAX_STEP_S, steps.acc + delta);
      if (!steps.pending && steps.spare) {
        steps.pending = true;
        sys.step(steps.acc, steps.spare);
        steps.spare = null; steps.acc = 0;
      }
    }
    // width in screen pixels, mildly following the zoom (1 at 1 m per pixel)
    const follow = Math.pow(1 / this.metresPerPixel(), PARTICLE_ZOOM_POWER);
    const zoom = Math.min(PARTICLE_ZOOM_MAX, Math.max(PARTICLE_ZOOM_MIN, follow));
    mesh.material.uniforms.width.value = PARTICLE_WIDTH_PX * zoom * this.renderer.getPixelRatio();
    mesh.material.uniforms.resolution.value = this.resolution();
  }

  /** The particles' birth plane while its slider is held (wind_layers.js planeFrameMesh); points null removes it. */
  setPlaneFrame(points) {
    if (!points) { this.remove("planeFrame"); return; }
    this.add("planeFrame", planeFrameMesh(points));
  }

  // ---------- slices ----------
  /**
   * A slice of the prediction sweep while the wind is predicted, from the sweep worker's half-float texels
   * (wind_layers.js sliceGroup; clearSweep takes it away).
   */
  setSweepSlice(texels, layout, height, range) {
    this.add("sweep", sliceGroup(texels, layout, height, range, this.palette));
  }

  /** A horizontal speed slice: data [speed, mask] per cell of `layout`'s grid (Float32Array). */
  setSlice(data, layout, height, range) {
    this.add("slice", sliceGroup(sliceTexels(data, layout.gx * layout.gy), layout, height, range, this.palette));
  }

  /** A vertical speed slice (wind_layers.js verticalSliceGroup). */
  setVerticalSlice(data, ns, nz, plane, range) {
    this.add("slice", verticalSliceGroup(data, ns, nz, plane, range, this.palette));
  }

  /** The inflow arrows of the wind direction theta (wind_layers.js inflowArrowGroup), built again only on a change. */
  setInflowArrows(theta) {
    if (!this.city) return;
    const W = this.city.nx * CELL_M, H = this.city.ny * CELL_M;
    const key = `${W}x${H}:${theta.toFixed(4)}`;
    if (this.inflowKey === key && this.objects.inflow) return;
    this.inflowKey = key;
    this.inflowTheta = theta;
    this.add("inflow", inflowArrowGroup(W, H, theta));
  }
}
