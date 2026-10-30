// The drone flying along the selected route: a quadrotor (body, four arms, four rotor discs) in the route's colour, its
// frame darker, aligned with the line from start to goal (its frame's x axis along it) and never turned about the
// vertical (the energy model has no yaw), tilted so that its rotors' axis is the thrust direction (a multirotor's
// thrust is along that axis).  Like the route tubes it is larger than life (a real
// one, about 1 m, would not show) and has one size in metres in every city and view; it shows pale through buildings
// as the routes do.
import * as THREE from "../../vendor/three.module.js";
import { interpolateAlong } from "../shared/geometry.js";
import { routeCurve, flightTimes } from "./routes.js";
import { RENDER_ORDER } from "./materials.js";

// the drone's size: rotor tip to rotor tip along a diagonal (m)
/** The drone's span (metres): two hub distances and two rotor radii. */
export const DRONE_SPAN_M = 60;
/** Its route behind it: none up to its rear rotors (gapM), then fading in over fadeM (m along the route). */
export const DRONE_TRAIL = Object.freeze({ gapM: DRONE_SPAN_M / 2, fadeM: 35 });
// its parts in units of the hub distance (centre to a rotor's axis): body, arms, rotor discs and hubs
const BODY = { width: 0.62, height: 0.22 }, ARM = { width: 0.16, height: 0.1 };
const ROTOR = { radius: 0.5, height: 0.03, lift: 0.12 }, HUB = { radius: 0.1, height: 0.14 };
const FRAME_SHADE = 0.55;               // the frame: the route's colour times this
const ROTOR_OPACITY = 0.75, GHOST_OPACITY = 0.25;
const UP = new THREE.Vector3(0, 0, 1);

/** The drone's meshes in units of the hub distance: [geometry, colour, opacity] per part, for a route of `colour`. */
function parts(colour) {
  const frame = new THREE.Color(colour).multiplyScalar(FRAME_SHADE);
  const out = [[new THREE.BoxGeometry(BODY.width, BODY.width, BODY.height), frame, 1]];
  for (let k = 0; k < 4; k++) {
    const angle = Math.PI / 4 + k * Math.PI / 2, x = Math.cos(angle), y = Math.sin(angle);
    const arm = new THREE.BoxGeometry(1, ARM.width, ARM.height);
    arm.rotateZ(angle);
    arm.translate(x / 2, y / 2, 0);
    const hub = new THREE.CylinderGeometry(HUB.radius, HUB.radius, HUB.height, 16);
    hub.rotateX(Math.PI / 2);
    hub.translate(x, y, HUB.height / 2);
    const rotor = new THREE.CylinderGeometry(ROTOR.radius, ROTOR.radius, ROTOR.height, 40);
    rotor.rotateX(Math.PI / 2);
    rotor.translate(x, y, ROTOR.lift);
    out.push([arm, frame, 1], [hub, frame, 1], [rotor, colour, ROTOR_OPACITY]);
  }
  return out;
}

/**
 * The drone of a route of `colour` as a three.js group (hidden until placed): each part solid, and pale where
 * something nearer covers it (depth test GREATER, as the routes' see-through part, view/routes.js).
 */
export function droneGroup(colour) {
  const group = new THREE.Group();
  for (const [geometry, shade, opacity] of parts(colour)) {
    const solid = new THREE.Mesh(geometry, new THREE.MeshStandardMaterial({
      color: shade, roughness: 0.6, metalness: 0.1, transparent: opacity < 1, opacity, depthWrite: opacity === 1 }));
    const ghost = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
      color: shade, transparent: true, opacity: GHOST_OPACITY, depthWrite: false, depthFunc: THREE.GreaterDepth }));
    ghost.renderOrder = RENDER_ORDER.routeGhost;
    group.add(solid, ghost);
  }
  group.scale.setScalar(DRONE_SPAN_M / (2 * (1 + ROTOR.radius)));     // the span: two hub distances, two radii
  group.visible = false;
  return group;
}

/**
 * A flight along a route: flight = { points [[x, y, z], ...], speeds (planned ground speeds), thrusts [[tx, ty, tz],
 * ...] } per point (and the route's colour, for droneGroup).  Returns { duration (s of flight), length (m along the
 * curve the route's tube follows), place(group, t), alongAt(t) }: place puts the drone where it is t seconds after the
 * start, on that curve and at the time its arrows show (view/routes.js); alongAt is that place in metres along the
 * curve (past the goal, going on at the flight's mean speed).
 */
export function droneFlight({ points, speeds, thrusts }) {
  const curve = routeCurve(points);
  const { L, T } = flightTimes(points, speeds);
  const total = L[L.length - 1] || 1;
  const axis = [0, 1, 2].map((c) => thrusts.map((t) => t[c]));
  const direction = new THREE.Vector3(), tilt = new THREE.Quaternion();
  // the frame's x axis (between two arms) along the line from start to goal, for the whole flight
  const [a, b] = [points[0], points[points.length - 1]];
  const heading = new THREE.Quaternion().setFromAxisAngle(UP, Math.atan2(b[1] - a[1], b[0] - a[0]));
  const duration = T[T.length - 1], length = curve.getLength();
  const share = (t) => Math.min(1, Math.max(0, interpolateAlong(T, L, t) / total));
  return {
    duration, length,
    alongAt: (t) => (t <= duration ? share(t) * length : length + (t - duration) * length / (duration || 1)),
    place(group, t) {
      curve.getPointAt(share(t), group.position);
      direction.set(...axis.map((values) => interpolateAlong(T, values, t))).normalize();
      group.quaternion.multiplyQuaternions(tilt.setFromUnitVectors(UP, direction), heading);
    },
  };
}
