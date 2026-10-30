// The planned routes: solid tubes in their method's colour with arrow marks moving at the planned ground speed, a thin
// white casing, and a pale see-through part where buildings hide them.
import * as THREE from "../../vendor/three.module.js";
import { interpolateAlong } from "../shared/geometry.js";
import { WHITE } from "../shared/style.js";
import { arrowMaterial, cutAhead, RENDER_ORDER } from "./materials.js";

// The route arrows move ROUTE_LAPSE x faster than the drone would (the particles have their own time lapse), one mark
// every ROUTE_MARK_S s of flight (at the optimizer's cruise speed, planner/trajopt.js, about 50 m).
export const ROUTE_LAPSE = 7;
const ROUTE_MARK_S = 4;
// Tube radius in true metres, the same in every city and mission (thicker tubes on the large city would hide its
// size); the selected route and the others get these shares of it.
const ROUTE_RADIUS_M = 3.6, CHOSEN_RADIUS_SHARE = 1.3, OTHER_RADIUS_SHARE = 0.6;
// the tubes: segments along a route (3 per planned point, at most MAX_SEGMENTS), sides around the tube and its casing
const MAX_SEGMENTS = 800, TUBE_SIDES = 12, CASING_SIDES = 10;
// the arrows' flight time: speeds below MIN_MARK_SPEED (m/s) count as that (no endless segment)
const MIN_MARK_SPEED = 0.5;

/** The smooth curve a route is drawn along (and the drone flies along): through its points [[x, y, z], ...]. */
export function routeCurve(points) {
  return new THREE.CatmullRomCurve3(points.map((p) => new THREE.Vector3(p[0], p[1], p[2])), false, "centripetal");
}

/**
 * A route's flight: at each point the distance flown along its straight segments L (m) and the flight time T (s) at
 * the planned ground speeds (per point).
 */
export function flightTimes(points, speeds) {
  const L = [0], T = [0];
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i], ds = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    L.push(L[i - 1] + ds);
    T.push(T[i - 1] + ds / Math.max(MIN_MARK_SPEED, 0.5 * (speeds[i - 1] + speeds[i])));
  }
  return { L, T };
}

/** Flight time (s) at each tube vertex from the route's points and planned ground speeds (flightTimes). */
function tubeTimes(geometry, route, curveLength) {
  const { L, T } = flightTimes(route.points, route.speeds);
  const along = geometry.attributes.along.array, t = new Float32Array(along.length);
  const k = L[L.length - 1] / (curveLength || 1);
  for (let q = 0; q < along.length; q++) t[q] = interpolateAlong(L, T, along[q] * k);
  geometry.setAttribute("tAlong", new THREE.BufferAttribute(t, 1));
}

/**
 * `curve` as TubeGeometry reads it (its Frenet frames, its points by arc length), each computed once for the tubes
 * along it that have the same number of segments, and copied out after: a later tube is the same as on its own.
 */
function sharedSamples(curve) {
  let frames = null;
  const points = new Map();
  return {
    computeFrenetFrames: (segments, closed) => (frames ??= curve.computeFrenetFrames(segments, closed)),
    getPointAt: (u, target) => {
      if (!points.has(u)) points.set(u, curve.getPointAt(u));
      return target.copy(points.get(u));
    },
  };
}

/**
 * A tube along `curve` (sharedSamples) with an `along` attribute: metres along the route (its u coordinate times
 * `length`).
 */
function tube(curve, segments, radius, radial, length) {
  const geometry = new THREE.TubeGeometry(curve, segments, radius, radial, false);
  const uv = geometry.attributes.uv.array;
  const along = new Float32Array(uv.length / 2);
  for (let i = 0; i < along.length; i++) along[i] = uv[2 * i] * length;
  geometry.setAttribute("along", new THREE.BufferAttribute(along, 1));
  return geometry;
}

// a route (its object, as long as it is shown) -> its curve's length and samples (sharedSamples): the meshes are built
// again for another selection or a flight, along the same curve
const curves = new WeakMap();

/**
 * The meshes of one route: route = { colour, points [[x, y, z], ...], speeds }; chosen: the selected route (thicker,
 * in front where routes overlap); casing: the white casing's width in metres (about 1 px); time: the shared uniform of
 * the shown flight time; cut: for the route the drone flies, its uniforms (materials.js cutAhead), else null.
 */
export function routeMeshes(route, chosen, casing, time, cut) {
  if (!curves.has(route)) {
    const curve = routeCurve(route.points);
    curves.set(route, { length: curve.getLength(), samples: sharedSamples(curve) });
  }
  const { length, samples } = curves.get(route);     // the tubes and casings: the same frames and points
  const segments = Math.min(MAX_SEGMENTS, route.points.length * 3);
  const radius = (chosen ? CHOSEN_RADIUS_SHARE : OTHER_RADIUS_SHARE) * ROUTE_RADIUS_M;
  const colour = route.colour;
  const geometry = tube(samples, segments, radius, TUBE_SIDES, length);
  tubeTimes(geometry, route, length);
  const body = new THREE.Mesh(geometry, arrowMaterial({
    color: colour, emissive: colour, emissiveIntensity: chosen ? 0.3 : 0.2, roughness: 0.5, depthWrite: true,
    // where routes overlap, the selected one (normally Ours) is drawn in front of the others
    polygonOffset: chosen, polygonOffsetFactor: chosen ? -4 : 0, polygonOffsetUnits: chosen ? -4 : 0 },
    ROUTE_MARK_S, time));
  const casingMaterial = new THREE.MeshBasicMaterial({ color: WHITE, side: THREE.BackSide, depthWrite: true });
  const outline = new THREE.Mesh(tube(samples, segments, radius + casing, CASING_SIDES, length), casingMaterial);
  // the see-through part (routes show through buildings, particles do not): the same tube drawn only where something
  // nearer already covers it (depth test GREATER), pale, writing no depth; pulled a little towards the camera so the
  // route's own visible surface never passes the test
  const ghost = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: colour, transparent: true, opacity: 0.1, depthWrite: false, depthFunc: THREE.GreaterDepth,
    polygonOffset: true, polygonOffsetFactor: -8, polygonOffsetUnits: -8 }));
  ghost.renderOrder = RENDER_ORDER.routeGhost;
  if (cut) for (const mesh of [body, outline, ghost]) cutAhead(mesh.material, cut);
  return [body, outline, ghost];
}
