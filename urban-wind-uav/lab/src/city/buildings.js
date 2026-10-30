// The buildings of a city as surfaces (positions and shading normals) from its height map: the arithmetic only, so a
// worker can build them (height_map_worker.js); view/buildings.js makes the mesh.
import { CELL_M, LAYER_M } from "../shared/units.js";

/**
 * The shading normals of walls (lighting only, the faces stay where they are): the facade direction from a smoothed
 * building footprint (box blur, radius 2 cells), so a diagonal facade built from 4 m x- and y-facing steps is lit as
 * one surface instead of in stripes.  Falls back to the face normal where the smoothed direction disagrees with it (by
 * more than about 70 degrees).  Returns facade(cx, cy, face): the normal of a wall whose middle is at (cx, cy) (cell
 * units) with face normal `face`.
 */
function facadeNormals(city) {
  const { nx, ny, layers } = city;
  const occ = new Float32Array(nx * ny);
  for (let i = 0; i < nx * ny; i++) occ[i] = layers[i] > 0 ? 1 : 0;
  const blur = (src, along) => {
    const out = new Float32Array(nx * ny), r = 2;
    for (let y = 0; y < ny; y++) for (let x = 0; x < nx; x++) {
      let s2 = 0;
      for (let k = -r; k <= r; k++) {
        const xx = along ? Math.min(nx - 1, Math.max(0, x + k)) : x;
        const yy = along ? y : Math.min(ny - 1, Math.max(0, y + k));
        s2 += src[yy * nx + xx];
      }
      out[y * nx + x] = s2 / (2 * r + 1);
    }
    return out;
  };
  const smooth = blur(blur(occ, true), false);
  const S = (x, y) => smooth[Math.min(ny - 1, Math.max(0, y)) * nx + Math.min(nx - 1, Math.max(0, x))];
  return (cx, cy, face) => {
    const x = Math.floor(cx), y = Math.floor(cy);
    const gx = (S(x + 1, y) - S(x - 1, y) + S(x + 1, y - 1) - S(x - 1, y - 1)) / 4;
    const gy = (S(x, y + 1) - S(x, y - 1) + S(x - 1, y + 1) - S(x - 1, y - 1)) / 4;
    const l = Math.hypot(gx, gy);
    if (l < 1e-4) return face;
    const n = [-gx / l, -gy / l, 0];
    return n[0] * face[0] + n[1] * face[1] > 0.35 ? n : face;
  };
}

/**
 * The buildings' surfaces from the height map (city frame, metres): a roof quad per run of equal-height cells along x,
 * and wall quads only where two neighbouring cells differ in height (from the lower to the higher roof), merged along
 * runs.  No two faces share a plane, so nothing can z-fight.  Returns { position, normal (Float32Array, three values
 * per vertex, two triangles per quad), sphere } (boundingSphere).
 */
export function buildingSurfaces(city) {
  const { nx, ny, layers } = city;
  const h = (x, y) => (x < 0 || y < 0 || x >= nx || y >= ny ? 0 : layers[y * nx + x] * LAYER_M);
  const pos = [], nor = [];
  const quad = (a, b, c, d, n) => {           // a b c d counter-clockwise seen from the normal side
    pos.push(...a, ...b, ...c, ...a, ...c, ...d);
    for (let k = 0; k < 6; k++) nor.push(...n);
  };
  const facade = facadeNormals(city);
  // the wall corner lists below run clockwise seen from their normal side: reversed here; shading normal from facade()
  const wall = (a, b, c, d, n) => quad(a, d, c, b, facade((a[0] + c[0]) / 2 / CELL_M, (a[1] + c[1]) / 2 / CELL_M, n));
  // roofs: runs along x
  for (let y = 0; y < ny; y++) {
    let x = 0;
    while (x < nx) {
      const z = h(x, y);
      let e = x + 1;
      while (e < nx && h(e, y) === z) e++;
      if (z > 0) {
        const x0 = x * CELL_M, x1 = e * CELL_M, y0 = y * CELL_M, y1 = (y + 1) * CELL_M;
        quad([x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z], [0, 0, 1]);
      }
      x = e;
    }
  }
  // walls facing -y / +y: between rows y-1 and y, runs along x of equal (low, high) pairs
  for (let y = 0; y <= ny; y++) {
    let x = 0;
    while (x < nx) {
      const below = h(x, y - 1), above = h(x, y);
      let e = x + 1;
      while (e < nx && h(e, y - 1) === below && h(e, y) === above) e++;
      const Y = y * CELL_M, x0 = x * CELL_M, x1 = e * CELL_M;
      if (above > below) wall([x1, Y, below], [x0, Y, below], [x0, Y, above], [x1, Y, above], [0, -1, 0]);
      else if (below > above) wall([x0, Y, above], [x1, Y, above], [x1, Y, below], [x0, Y, below], [0, 1, 0]);
      x = e;
    }
  }
  // walls facing -x / +x: between columns x-1 and x, runs along y
  for (let x = 0; x <= nx; x++) {
    let y = 0;
    while (y < ny) {
      const left = h(x - 1, y), right = h(x, y);
      let e = y + 1;
      while (e < ny && h(x - 1, e) === left && h(x, e) === right) e++;
      const X = x * CELL_M, y0 = y * CELL_M, y1 = e * CELL_M;
      if (right > left) wall([X, y0, left], [X, y1, left], [X, y1, right], [X, y0, right], [-1, 0, 0]);
      else if (left > right) wall([X, y1, right], [X, y0, right], [X, y0, left], [X, y1, left], [1, 0, 0]);
      y = e;
    }
  }
  const position = new Float32Array(pos);
  return { position, normal: new Float32Array(nor), sphere: boundingSphere(position) };
}

/**
 * The bounding sphere of `position` (three values per vertex) as three.js computeBoundingSphere makes it: the centre
 * of the box around the vertices (the origin without vertices), the radius to the farthest one.  { center, radius }.
 */
function boundingSphere(position) {
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < position.length; i += 3) {
    for (let c = 0; c < 3; c++) {
      lo[c] = Math.min(lo[c], position[i + c]);
      hi[c] = Math.max(hi[c], position[i + c]);
    }
  }
  const center = position.length ? lo.map((v, c) => (v + hi[c]) * 0.5) : [0, 0, 0];
  let farthest = 0;
  for (let i = 0; i < position.length; i += 3) {
    const dx = center[0] - position[i], dy = center[1] - position[i + 1], dz = center[2] - position[i + 2];
    farthest = Math.max(farthest, dx * dx + dy * dy + dz * dz);
  }
  return { center, radius: Math.sqrt(farthest) };
}
