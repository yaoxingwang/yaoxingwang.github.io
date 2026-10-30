// Pointer and wheel control of the 3-D view: left drag pans (shift: turns), right or middle drag turns, the wheel
// zooms toward the pointer, a middle click returns to the opening view (while the drone flies it makes the view
// follow the drone, and the next one ends that; a left click on the flying drone does the same), and the start and
// goal pins are dragged.  On a touch screen: one
// finger as the left button; two fingers pinch to zoom about their midpoint, twist to turn and move up or down
// together to tilt; a double tap as a middle click.  From the keyboard (the canvas focused): arrow keys move the view,
// Shift + arrow keys turn and tilt it, + and - zoom, 0 or Home as a middle click.  While the view follows the drone
// all of these work as ever (it keeps where the user put it relative to the drone: Viewer trackDrone), and any of them
// ends the view's ease onto the drone.
import * as THREE from "../../vendor/three.module.js";

// A press that moves less than this (CSS pixels) is a click: a hand tremor would otherwise move a pin and plan again.
const CLICK_PX = 5;
// orbit: degrees of azimuth and of elevation per pixel dragged; the elevation stays in its range (degrees)
const ORBIT_DEG_PER_PX = 0.3, TILT_DEG_PER_PX = 0.25;
const MIN_ELEVATION_DEG = 6, MAX_ELEVATION_DEG = 88;
// pan: a ground point beyond FAR_DISTANCES viewing distances is too far to follow; then the screen rule moves
// SCREEN_PAN_RATE x the distance per pixel (along the view up to 1 / MIN_TILT_SINE times that); a step moves at most
// PAN_STEP_LIMIT x the distance
const FAR_DISTANCES = 6, SCREEN_PAN_RATE = 0.0012, MIN_TILT_SINE = 0.35, PAN_STEP_LIMIT = 0.25;
// zoom: the distance changes by exp(ZOOM_RATE x wheel delta) within its range (m), toward the point under the cursor
// when it is within ZOOM_TOWARD_DISTANCES viewing distances of the target
const ZOOM_RATE = 0.0011, MIN_DISTANCE_M = 80, MAX_DISTANCE_M = 15000, ZOOM_TOWARD_DISTANCES = 3;
// two taps within this time (ms) and distance (CSS pixels) of each other are a double tap
const DOUBLE_TAP_MS = 300, DOUBLE_TAP_PX = 24;
// keyboard: an arrow key moves the view as a drag of KEY_PAN_PX does, turns it KEY_TURN_DEG or tilts it KEY_TILT_DEG
// with Shift; + and - zoom as one wheel step of KEY_WHEEL_DELTA
const KEY_PAN_PX = 40, KEY_TURN_DEG = 10, KEY_TILT_DEG = 5, KEY_WHEEL_DELTA = 100;
const ARROWS = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };

/** Wire the canvas of `viewer` (view/viewer.js) to its camera and pins. */
export function attachInteraction(viewer) {
  const canvas = viewer.canvas;
  // the press being dragged: mode null (none) | "pin" | "pan" | "orbit" | "pinch" (two fingers); the pointer's last
  // position and the pixels it moved; the button; the height of the plane a pan grabs; the dragged pin and its grab
  // offset (grabOffset)
  const drag = { mode: null, last: null, moved: 0, button: 0, grabZ: 0, pin: null, grab: null };
  // the fingers on the canvas (pointer id -> [x, y]), the two-finger gesture's last span, angle and midpoint, and the
  // last tap (a double tap's first half)
  const touch = { points: new Map(), pinch: null, lastTap: null };
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());
  // no browser autoscroll on a middle-button drag
  canvas.addEventListener("mousedown", (event) => { if (event.button === 1) event.preventDefault(); });
  canvas.addEventListener("pointerdown", (event) => {
    if (event.pointerType === "touch") touchDown(viewer, drag, touch, event); else press(viewer, drag, event);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (touch.points.has(event.pointerId)) touchMove(viewer, drag, touch, event); else move(viewer, drag, event);
  });
  for (const type of ["pointerup", "pointercancel"]) {
    canvas.addEventListener(type, (event) => {
      if (touch.points.has(event.pointerId)) touchUp(viewer, drag, touch, event); else release(viewer, drag, event);
    });
  }
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoomAbout(viewer, Math.exp(event.deltaY * ZOOM_RATE), [event.clientX, event.clientY]);
  }, { passive: false });
  canvas.addEventListener("keydown", (event) => keyControl(viewer, event));
}

/** A key on the focused canvas: move, turn, tilt or zoom the view, or return to the opening view. */
function keyControl(viewer, event) {
  if (event.ctrlKey || event.metaKey || event.altKey) return;     // the browser's own shortcuts (page zoom, ...)
  const arrow = ARROWS[event.key];
  if (arrow) {
    event.preventDefault();
    viewer.animation = null;
    viewer.userMoved = true;
    viewer.endFollowEase();
    if (event.shiftKey) {
      viewer.azimuth -= arrow[0] * KEY_TURN_DEG;
      const elevation = viewer.elevation - arrow[1] * KEY_TILT_DEG;
      viewer.elevation = Math.min(MAX_ELEVATION_DEG, Math.max(MIN_ELEVATION_DEG, elevation));
    } else {
      // the view moves toward the arrow: the city as if dragged the other way, from the view's centre on screen
      const [cx, cy] = viewer.toScreen([viewer.target.x, viewer.target.y, viewer.target.z]);
      const dx = -arrow[0] * KEY_PAN_PX, dy = -arrow[1] * KEY_PAN_PX;
      const [mx, my] = panStep(viewer, [cx + dx, cy + dy], dx, dy, viewer.target.z);
      viewer.target.x += mx; viewer.target.y += my;
      viewer.keepTargetOnCity();
    }
    viewer.updateCamera();
  } else if (event.key === "+" || event.key === "=" || event.key === "-") {
    event.preventDefault();
    zoomAbout(viewer, Math.exp((event.key === "-" ? 1 : -1) * KEY_WHEEL_DELTA * ZOOM_RATE));     // about the target
  } else if (event.key === "0" || event.key === "Home") {
    event.preventDefault();
    viewer.returnView();
  }
}

/** A press: on a pin's button (left button) its drag starts; otherwise a pan or a turn of the view. */
function press(viewer, drag, event) {
  drag.last = [event.clientX, event.clientY];
  drag.moved = 0;
  drag.button = event.button;
  drag.grabZ = viewer.target.z;          // pan: the plane at the orbit target's height is "grabbed"
  viewer.animation = null;
  viewer.endFollowEase();
  viewer.canvas.setPointerCapture(event.pointerId);
  viewer.highlightDrone(null);           // (a press on the drone: a click on it, released at once)
  const pin = event.button === 0 ? viewer.pinUnder(event.clientX, event.clientY) : null;
  if (pin) {
    drag.grab = grabOffset(viewer, pin, event);
    drag.mode = "pin"; drag.pin = pin;
    viewer.setCursor("move");
    viewer.dismissDragHint();
    viewer.emit("pinDragStart", pin);
    return;
  }
  // left drag pans, right or middle drag turns the view (shift + left also turns)
  drag.mode = (event.button === 0 && !event.shiftKey) ? "pan" : "orbit";
  viewer.setCursor("grabbing");
}

/** A pointer move: hovering (cursor and pin highlight), or the current drag (not a two-finger one: touchMove). */
function move(viewer, drag, event) {
  if (drag.mode === "pinch") return;
  if (!drag.mode) {
    // cursor: move (and a highlighted pin) over a pin, pointer over the flying drone (a click follows it), otherwise
    // grab (the view drags)
    const pin = viewer.pinUnder(event.clientX, event.clientY);
    const drone = !pin && viewer.droneUnder(event.clientX, event.clientY);
    viewer.highlightPin(pin);
    viewer.highlightDrone(drone ? [event.clientX, event.clientY] : null);
    viewer.setCursor(pin ? "move" : drone ? "pointer" : "grab");
    return;
  }
  const dx = event.clientX - drag.last[0], dy = event.clientY - drag.last[1];
  drag.moved += Math.abs(dx) + Math.abs(dy);
  drag.last = [event.clientX, event.clientY];
  if (drag.mode === "pin") {
    // a hand tremor on a click is not a move (a few pixels would move the point and recompute the routes)
    if (drag.moved < CLICK_PX) return;
    // the end point under the pointer (by the grab offset) on the plane at its height
    const p = viewer.groundPoint(event.clientX + drag.grab.dx, event.clientY + drag.grab.dy, drag.grab.z);
    if (p) viewer.emit("pinDrag", drag.pin, p);
    viewer.setCursor("move");
    return;
  }
  viewer.userMoved = true;
  if (drag.mode === "orbit") {
    viewer.azimuth -= dx * ORBIT_DEG_PER_PX;
    const elevation = viewer.elevation + dy * TILT_DEG_PER_PX;
    viewer.elevation = Math.min(MAX_ELEVATION_DEG, Math.max(MIN_ELEVATION_DEG, elevation));
  } else {
    const [mx, my] = panStep(viewer, drag.last, dx, dy, drag.grabZ);
    viewer.target.x += mx; viewer.target.y += my;
    viewer.keepTargetOnCity();
  }
  viewer.updateCamera();
}

/**
 * The press ends (or is cancelled): a pin drag ends; a middle click without a drag, or a left one on the flying
 * drone: Viewer returnView.
 */
function release(viewer, drag, event) {
  const canvas = viewer.canvas;
  if (canvas.hasPointerCapture?.(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  const { mode, moved, button } = drag;
  endDrag(viewer, drag);
  const click = mode !== "pin" && moved < CLICK_PX;
  if (click && (button === 1 || (button === 0 && viewer.droneUnder(event.clientX, event.clientY)))) viewer.returnView();
}

/** The current drag ends (a pin's drag reports its end), before whatever its listeners do. */
function endDrag(viewer, drag) {
  const { mode, pin } = drag;
  drag.mode = null; drag.pin = null; drag.grab = null;
  viewer.setCursor("grab");
  if (mode === "pin") viewer.emit("pinDragEnd", pin);
}

/** The two fingers' span (px), angle (degrees) and midpoint. */
function pinchOf(points) {
  const [[x0, y0], [x1, y1]] = points.values();
  return { span: Math.hypot(x1 - x0, y1 - y0), angle: Math.atan2(y1 - y0, x1 - x0) * 180 / Math.PI,
           mid: [(x0 + x1) / 2, (y0 + y1) / 2] };
}

/** A finger lands: the first one as the left button; a second one ends that drag and starts the two-finger gesture. */
function touchDown(viewer, drag, touch, event) {
  touch.points.set(event.pointerId, [event.clientX, event.clientY]);
  if (touch.points.size === 1) {
    press(viewer, drag, event);
    return;
  }
  viewer.canvas.setPointerCapture(event.pointerId);
  if (touch.points.size !== 2) return;            // a third finger is held, not followed
  endDrag(viewer, drag);
  drag.mode = "pinch";
  touch.pinch = pinchOf(touch.points);
}

/** A finger moves: the one-finger drag, or the two-finger gesture (zoom, turn and tilt by its change since last). */
function touchMove(viewer, drag, touch, event) {
  touch.points.set(event.pointerId, [event.clientX, event.clientY]);
  if (drag.mode !== "pinch") {
    move(viewer, drag, event);
    return;
  }
  if (touch.points.size !== 2) return;
  const now = pinchOf(touch.points), was = touch.pinch;
  touch.pinch = now;
  if (now.span > 0 && was.span > 0) zoomAbout(viewer, was.span / now.span, now.mid);
  let turn = now.angle - was.angle;
  if (turn > 180) turn -= 360;
  if (turn < -180) turn += 360;
  viewer.azimuth += turn;                         // the city turns with the fingers
  const elevation = viewer.elevation + (now.mid[1] - was.mid[1]) * TILT_DEG_PER_PX;
  viewer.elevation = Math.min(MAX_ELEVATION_DEG, Math.max(MIN_ELEVATION_DEG, elevation));
  viewer.updateCamera();
}

/**
 * A finger lifts: during the two-finger gesture, the fingers left go on from where they are (no jump): two (a third
 * lifted) the gesture, one a pan; the last finger ends the drag as the left button does (a tap on the flying drone:
 * a left click on it), and a tap close after another (DOUBLE_TAP_MS, DOUBLE_TAP_PX) acts as a middle click (Viewer
 * returnView; not after a tap on the drone, which did that already).
 */
function touchUp(viewer, drag, touch, event) {
  touch.points.delete(event.pointerId);
  if (drag.mode === "pinch") {
    if (viewer.canvas.hasPointerCapture?.(event.pointerId)) viewer.canvas.releasePointerCapture(event.pointerId);
    if (touch.points.size === 2) {
      touch.pinch = pinchOf(touch.points);
    } else if (touch.points.size === 1) {
      drag.mode = "pan";
      drag.last = [...touch.points.values()][0];
      drag.moved = CLICK_PX;                     // not a tap any more
      drag.grabZ = viewer.target.z;
    }
    return;
  }
  const tap = drag.mode === "pan" && drag.moved < CLICK_PX;
  const onDrone = tap && viewer.droneUnder(event.clientX, event.clientY);
  release(viewer, drag, event);
  if (!tap || onDrone) {
    touch.lastTap = null;
    return;
  }
  const now = performance.now(), here = [event.clientX, event.clientY], last = touch.lastTap;
  const near = last && Math.hypot(here[0] - last.at[0], here[1] - last.at[1]) < DOUBLE_TAP_PX;
  if (near && now - last.time < DOUBLE_TAP_MS) {
    touch.lastTap = null;
    viewer.returnView();
  } else {
    touch.lastTap = { time: now, at: here };
  }
}

/**
 * Where the end point of `pin` is on screen relative to the pointer, and its height: the user holds the button in
 * the air at the flight height (no jump to the ground); while dragging, the end point follows the pointer by that
 * offset on the horizontal plane at its own height, so the button stays under the pointer and nothing jumps.
 */
function grabOffset(viewer, pin, event) {
  const head = viewer.pinHead(pin).getWorldPosition(new THREE.Vector3());
  const [sx, sy] = viewer.toScreen([head.x, head.y, head.z]);
  return { dx: sx - event.clientX, dy: sy - event.clientY, z: head.z };
}

/**
 * One pan step for a pointer move (dx, dy) ending at `last`: the point of the plane at height grabZ under the cursor
 * stays under the cursor.  Near the horizon that point runs away to infinity, so each step is limited to
 * PAN_STEP_LIMIT viewing distances; where the plane is not hit (cursor above the horizon) a screen-space rule is used.
 * Returns the target's move [mx, my] (metres).
 */
function panStep(viewer, last, dx, dy, grabZ) {
  const p0 = viewer.groundPoint(last[0] - dx, last[1] - dy, grabZ);
  const p1 = viewer.groundPoint(last[0], last[1], grabZ);
  const camera = viewer.camera.position;
  const far = (p) => !p || Math.hypot(p[0] - camera.x, p[1] - camera.y) > FAR_DISTANCES * viewer.distance;
  let mx, my;
  if (!far(p0) && !far(p1)) {
    mx = p0[0] - p1[0];
    my = p0[1] - p1[1];
  } else {
    const scale = viewer.distance * SCREEN_PAN_RATE, az = viewer.azimuth * Math.PI / 180;
    const k = 1 / Math.max(MIN_TILT_SINE, Math.sin(viewer.elevation * Math.PI / 180));
    mx = -dx * scale * -Math.sin(az) + dy * scale * k * -Math.cos(az);
    my = -dx * scale * Math.cos(az) + dy * scale * k * -Math.sin(az);
  }
  const len = Math.hypot(mx, my), cap = PAN_STEP_LIMIT * viewer.distance;
  if (len > cap) { mx *= cap / len; my *= cap / len; }
  return [mx, my];
}

/**
 * Zoom by `factor` (the distance's ratio) toward the screen point `at` [x, y], which stays fixed (wheel, pinch), or
 * about the view's target (no `at`: the keyboard).
 */
function zoomAbout(viewer, factor, at = null) {
  viewer.animation = null;
  viewer.endFollowEase();
  const next = Math.min(MAX_DISTANCE_M, Math.max(MIN_DISTANCE_M, viewer.distance * factor));
  const p = at && viewer.groundPoint(at[0], at[1], viewer.target.z);
  // near the horizon the point under the cursor is very far away: no zoom toward it then
  if (p && Math.hypot(p[0] - viewer.target.x, p[1] - viewer.target.y) < ZOOM_TOWARD_DISTANCES * viewer.distance) {
    const k = 1 - next / viewer.distance;
    viewer.target.x += (p[0] - viewer.target.x) * k;
    viewer.target.y += (p[1] - viewer.target.y) * k;
    viewer.keepTargetOnCity();
  }
  viewer.distance = next;
  viewer.userMoved = true;
  viewer.updateCamera();
}
