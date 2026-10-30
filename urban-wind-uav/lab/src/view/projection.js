// Camera maths shared by the view modules: points on screen, and the size of a screen pixel in the scene.
import * as THREE from "../../vendor/three.module.js";

const projected = new THREE.Vector3();          // project's working vector

/**
 * [x, y, depth] of the point (x, y, z) (metres) seen by `camera` in a w x h pixel viewport (depth > 1: behind),
 * written into `out` when given (a search over many points makes no garbage), else into a new array.
 */
export function project(camera, x, y, z, w, h, out = [0, 0, 0]) {
  const p = projected.set(x, y, z).project(camera);
  out[0] = (p.x + 1) / 2 * w; out[1] = (1 - p.y) / 2 * h; out[2] = p.z;
  return out;
}

/** Metres per screen pixel at distance `d` from `camera`, on a canvas `height` pixels high. */
export const metresPerPixelAt = (camera, d, height) => 2 * d * Math.tan(camera.fov * Math.PI / 360) / height;
