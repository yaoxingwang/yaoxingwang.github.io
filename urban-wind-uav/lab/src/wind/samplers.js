// Wind samplers in the city frame (shared/units.js): sampler(x, y, z) -> [wx, wy, wz] in m/s at a point in metres
// (sampler(x, y, z, out): written into the array out and returned, for a caller that samples many points).
// The predicted wind, the CFD ground truth, the height profile and no wind; and the plain-data descriptors that carry
// a wind to a worker (windFromDescriptor).
import { cellIndexOf, layerIndexOf } from "../shared/units.js";

// How far (cells) a sample of the predicted wind looks for the nearest predicted cell when none of its 8 neighbours
// is predicted.
const MAX_RING = 16;

/** `out` set to no wind, and returned. */
const calm = (out) => { out[0] = 0; out[1] = 0; out[2] = 0; return out; };

/**
 * Trilinear sample of a [y][z][x][stride] grid at fractional indices.  Outside the grid the position is clamped to it
 * (edge values, not zero wind: the field beyond the predicted or stored region is unknown, not calm).  With
 * `weighted` (stride 4, the 4th value = prediction weight, 0 = not predicted) only predicted neighbours are averaged;
 * if none of the 8 is predicted, the nearest predicted cell of the same layer within MAX_RING cells is used.  Returns
 * null only if nothing is found.
 *
 * ringCache: the predicted cells are the same on every layer (every patch covers all of them), so the cell found for
 * (iy, ix) is kept and used for the other heights after checking it on this layer; -2: not searched yet, -1: nothing
 * found within MAX_RING (the search runs again).
 */
function trilinear(values, ny, nz, nx, fy, fz, fx, out, stride = 3, weighted = false, ringCache = null) {
  fy = Math.min(Math.max(fy, 0), ny - 1);
  fz = Math.min(Math.max(fz, 0), nz - 1);
  fx = Math.min(Math.max(fx, 0), nx - 1);
  const y0 = Math.max(0, Math.min(ny - 2, Math.floor(fy))), z0 = Math.max(0, Math.min(nz - 2, Math.floor(fz))),
        x0 = Math.max(0, Math.min(nx - 2, Math.floor(fx)));
  const a = fy - y0, b = fz - z0, c = fx - x0;
  out[0] = 0; out[1] = 0; out[2] = 0;
  let total = 0;
  for (let dy = 0; dy < 2; dy++) for (let dz = 0; dz < 2; dz++) for (let dx = 0; dx < 2; dx++) {
    const w = (dy ? a : 1 - a) * (dz ? b : 1 - b) * (dx ? c : 1 - c);
    if (!w) continue;
    const i = (((y0 + dy) * nz + z0 + dz) * nx + x0 + dx) * stride;
    if (weighted && !(values[i + 3] > 0)) continue;
    out[0] += w * values[i]; out[1] += w * values[i + 1]; out[2] += w * values[i + 2];
    total += w;
  }
  if (total > 0) { out[0] /= total; out[1] /= total; out[2] /= total; return out; }
  // no predicted neighbour: the nearest predicted cell of the nearest layer, ring by ring
  const iy = Math.round(fy), iz = Math.round(fz), ix = Math.round(fx);
  const key = iy * nx + ix;
  if (ringCache) {
    const hit = ringCache[key];
    if (hit >= 0) {
      const yy = Math.floor(hit / nx), xx = hit - yy * nx, i = ((yy * nz + iz) * nx + xx) * stride;
      if (values[i + 3] > 0) { out[0] = values[i]; out[1] = values[i + 1]; out[2] = values[i + 2]; return out; }
    }
  }
  for (let r = 1; r <= MAX_RING; r++) {
    for (let yy = iy - r; yy <= iy + r; yy++) for (let xx = ix - r; xx <= ix + r; xx++) {
      if (Math.max(Math.abs(yy - iy), Math.abs(xx - ix)) !== r || yy < 0 || xx < 0 || yy >= ny || xx >= nx) continue;
      const i = ((yy * nz + iz) * nx + xx) * stride;
      if (values[i + 3] > 0) {
        if (ringCache && ringCache[key] === -2) ringCache[key] = yy * nx + xx;
        out[0] = values[i]; out[1] = values[i + 1]; out[2] = values[i + 2]; return out;
      }
    }
  }
  if (ringCache && ringCache[key] === -2) ringCache[key] = -1;
  return null;
}

/**
 * The predicted wind (gpu/predictor.js result: wind-frame u, w, v and weight on [gy][gz][gx]) scaled by `scale`, as a
 * sampler in the city frame.
 */
export function predictedWind(result, city, theta, scale) {
  const { gx, gy, gz, values, layout, firstLayer, valuesPerCell } = result;
  const cY = (city.ny - 1) / 2, cX = (city.nx - 1) / 2;
  const cos = Math.cos(theta), sin = Math.sin(theta);
  const tmp = [0, 0, 0];
  let ringCache = null;          // made on the first sample (trilinear)
  return (x, y, z, out = [0, 0, 0]) => {
    const ix = cellIndexOf(x), iy = cellIndexOf(y);
    const dx = ix - cX, dy = iy - cY;
    const sx = cX + dx * cos + dy * sin, sy = cY - dx * sin + dy * cos;     // shared/geometry.js windFrame toWind
    if (!ringCache) ringCache = new Int32Array(gy * gx).fill(-2);
    const s = trilinear(values, gy, gz, gx, sy - layout.gridY0, layerIndexOf(z) - firstLayer,
                        sx - layout.gridX0, tmp, valuesPerCell, true, ringCache);   // weighted: the 4th value
    if (!s) return calm(out);                      // nothing predicted near: no wind
    const u = s[0], w = s[1], v = s[2];           // along the wind, vertical, across the wind
    out[0] = scale * (u * cos - v * sin); out[1] = scale * (u * sin + v * cos); out[2] = scale * w;
    return out;
  };
}

/** The CFD ground-truth wind (wind/cfd.js grid: every layerStride-th layer, inflow along +x; stored u, w, v). */
export function cfdWind(grid) {
  const { values, ny, nz, nx, layerStride, firstLayer } = grid;
  const tmp = [0, 0, 0];
  return (x, y, z, out = [0, 0, 0]) => {
    const s = trilinear(values, ny, nz, nx, cellIndexOf(y), (layerIndexOf(z) - firstLayer) / layerStride,
                        cellIndexOf(x), tmp);      // unweighted: found at every number, not at NaN
    if (!s) return calm(out);
    out[0] = s[0]; out[1] = s[2]; out[2] = s[1];
    return out;
  };
}

/** The along-wind speed of a height profile (manifest.profile) scaled by `scale`, blowing toward theta. */
export function profileWind(profile, theta, scale) {
  const values = profile.along_wind_m_s;
  const first = profile.first_layer;
  const cos = Math.cos(theta), sin = Math.sin(theta);
  return (_x, _y, z, out = [0, 0, 0]) => {    // the same at every (x, y)
    const f = Math.min(values.length - 1, Math.max(0, layerIndexOf(z) - first));
    const i = Math.min(values.length - 2, Math.floor(f));
    const s = scale * (values[i] + (f - i) * (values[i + 1] - values[i]));
    out[0] = s * cos; out[1] = s * sin; out[2] = 0;
    return out;
  };
}

/** No wind (the wind-agnostic route). */
export const noWind = () => (_x, _y, _z, out = [0, 0, 0]) => calm(out);

/**
 * The descriptor (below) of a predicted field (gpu/predictor.js result) of a city (nx, ny) for wind direction theta
 * and speed scale.
 */
export function predictedDescriptor(prediction, city, theta, scale) {
  const p = prediction;
  return { kind: "predicted", values: p.values, gx: p.gx, gy: p.gy, gz: p.gz, gridX0: p.layout.gridX0,
           gridY0: p.layout.gridY0, firstLayer: p.firstLayer, layerStride: p.layerStride,
           valuesPerCell: p.valuesPerCell, ny: city.ny, nx: city.nx, theta, scale };
}

/**
 * A sampler from a wind descriptor, the plain data that carries a wind to a worker:
 *   { kind: "predicted", values, gx, gy, gz, gridX0, gridY0, firstLayer, layerStride, valuesPerCell, nx, ny, theta,
 *     scale }
 *   { kind: "cfd", grid }   { kind: "profile", profile, theta, scale }   { kind: "none" }
 * The kinds are winds, not planning methods: the methods (app/config.js METHODS) ours, cfd, profile and agnostic plan
 * in the predicted, cfd, profile and no wind.
 */
export function windFromDescriptor(d) {
  if (d.kind === "predicted") {
    const field = { gx: d.gx, gy: d.gy, gz: d.gz, values: d.values, layout: { gridX0: d.gridX0, gridY0: d.gridY0 },
                    firstLayer: d.firstLayer, valuesPerCell: d.valuesPerCell };
    return predictedWind(field, { ny: d.ny, nx: d.nx }, d.theta, d.scale);
  }
  if (d.kind === "cfd") return cfdWind(d.grid);
  if (d.kind === "profile") return profileWind(d.profile, d.theta, d.scale);
  return noWind();
}
