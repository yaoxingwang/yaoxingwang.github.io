// The prediction sweep's slices, prepared off the main thread (on the large city each slice would block the page
// for a noticeable time): per prediction the corridor mask and the roof under each cell; per layer the speeds (the
// weighted sums finished as the prediction finishes them) and the half-float texels of the slice texture.
// Messages: { city }; { roofs: { key, windPoly, gx, gy, gridX0, gridY0, theta } }; a layer
// { id, key, layer, sums, gx, gy, target, scale } -> { id, layer, texels }, or { id, stale } for a layer of an
// older prediction than the last roofs, or { id, error } when the slice failed (the worker goes on).
import { CELL_VALUES } from "../shared/units.js";
import { windFrame } from "../shared/geometry.js";
import { errorText } from "../shared/errors.js";
import { cellAt, forCellsInside } from "../city/cells.js";
import { MIN_BLEND_WEIGHT } from "../gpu/field_post.js";
import { setSliceTexel } from "./half_float.js";

let city = null, roofs = null, roofsKey = null;

self.onmessage = (event) => {
  const m = event.data;
  if (m.city) { city = m.city; return; }
  if (m.roofs) {
    // per cell: the roof (layers) under it, -1 outside the corridor
    const { key, windPoly, gx, gy, gridX0, gridY0, theta } = m.roofs;
    const frame = windFrame(theta, city.nx, city.ny);         // as the prediction's layout (gpu/tiling.js)
    roofs = new Float32Array(gx * gy).fill(-1);
    forCellsInside(windPoly, { gx, gy, gridX0, gridY0 }, frame, (o, x, y) => {
      const cell = cellAt(city, x, y);
      roofs[o] = cell < 0 ? Infinity : city.layers[cell];
    });
    roofsKey = key;
    return;
  }
  if (m.key !== roofsKey) { self.postMessage({ id: m.id, stale: true }); return; }
  try {
    const texels = sliceTexels(m);
    self.postMessage({ id: m.id, layer: m.layer, texels }, [texels.buffer]);
  } catch (error) {
    self.postMessage({ id: m.id, error: errorText(error) });
  }
};

/** A layer message's slice: its half-float RGBA texels. */
function sliceTexels({ layer, sums: s, gx, gy, target, scale }) {
  // weighted sums to winds (reference units scaled and centred as the model's target)
  const c = target.center, k = target.scale;
  for (let i = 0; i < s.length; i += CELL_VALUES) {
    const w = s[i + 3];
    if (w > MIN_BLEND_WEIGHT) {
      for (let j = 0; j < 3; j++) s[i + j] = s[i + j] / w * k[j] + c[j];
    } else {
      s[i] = 0; s[i + 1] = 0; s[i + 2] = 0; s[i + 3] = 0;
    }
  }
  // speed and mask as the RGBA half-float texels of a slice texture (the same encoding as the page's slices)
  const texels = new Uint16Array(gx * gy * 4);         // RGBA
  for (let cell = 0; cell < gx * gy; cell++) {
    const i = cell * CELL_VALUES, roof = roofs[cell];
    // outside the corridor, in a building (its roof reaches the layer's centre, as city/cells.js solidAt) or no patch
    if (roof < 0 || roof >= layer + 0.5 || !(s[i + 3] > 0)) continue;
    setSliceTexel(texels, cell, scale * Math.hypot(s[i], s[i + 1], s[i + 2]));
  }
  return texels;
}
