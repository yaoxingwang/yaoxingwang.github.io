// The ribbons the particle trails are drawn as (view/wind_layers.js particleMesh): two vertices per trail point, the
// arrays that depend only on the particle count, built by the particle worker with a system's first arrays.
import { TRAIL } from "./system.js";

/** A ribbon's width at its tail, a share of its head's (the width shrinks linearly along the trail). */
const PARTICLE_TAIL_TAPER = 0.45;

/**
 * The ribbons of `count` particles: per vertex its side (-1, 1) and its width share (taper), and the triangle index
 * (two triangles between consecutive trail points).  { side, taper (Float32Array), index (Uint32Array) }.
 */
export function ribbonArrays(count) {
  const points = count * TRAIL, verts = points * 2;
  const side = new Float32Array(verts), taper = new Float32Array(verts);
  for (let j = 0; j < points; j++) {
    const k = j % TRAIL, w = PARTICLE_TAIL_TAPER + (1 - PARTICLE_TAIL_TAPER) * (1 - k / (TRAIL - 1));
    side[2 * j] = -1; side[2 * j + 1] = 1; taper[2 * j] = w; taper[2 * j + 1] = w;
  }
  const index = new Uint32Array(count * (TRAIL - 1) * 6);
  let q = 0;
  for (let i = 0; i < count; i++) for (let k = 0; k < TRAIL - 1; k++) {
    const a = 2 * (i * TRAIL + k), b = a + 2;
    index[q++] = a; index[q++] = a + 1; index[q++] = b;
    index[q++] = a + 1; index[q++] = b + 1; index[q++] = b;
  }
  return { side, taper, index };
}
