// The WGSL compute shaders of the wind model's geometry inputs (mddf.js): the whole-city directional scan and its
// Fourier sums, the packed field, the along-wind channels and the assemble pass that samples every input into model
// patches.  The generated source text is part of the numeric result (literal formatting decides the float32 values):
// change it only on purpose.
import { CELL_M, LAYER_M } from "../shared/units.js";
import { WORKGROUP } from "./context.js";

/** Threads per workgroup of the scan, accumulate, pack and along-wind shaders (mddf.js dispatches them so). */
export const MDDF_WORKGROUP = 64;
export const ANGLE_COUNT = 180;        // azimuths of the stored channels (2 degrees)
const DISTANCE_CUTOFF_M = 1600.0;      // distance when no roof is reached
const RAY_STEP_LIMIT = 640;            // cells a ray walks at most
export const LAYER_GROUP = 8;          // query heights per along-wind thread
export const UP_TILE = 32;             // along-wind channels on the wind-frame grid: tiles of 32 x 32 cells
/**
 * WGSL text of a short decimal constant (layer height, distances, the azimuth count): six decimals.  `lit` below is
 * for computed values that must parse to their exact float32; the two give different text, and the shader text is
 * part of the numeric result, so each keeps its own.
 */
const wgslFloat = (value) => value.toFixed(6);
/** WGSL literal that parses to the float32 nearest to `value` (as numpy float32 normalization). */
const lit = (value) => {
  const v = Math.fround(value);
  return Number.isInteger(v) && Math.abs(v) < 1e7 ? v.toFixed(1) : v.toExponential(9);
};

// Division rounded as IEEE float32 (WGSL "/" may be off by up to 2.5 ulp on the GPU, which breaks the
// equal-slope ties of the lower-obstruction rule against a CPU computation): one residual step.
const DIV = `
fn divr(a: f32, b: f32) -> f32 {
  let q = a / b;
  let r = fma(-q, b, a);
  return q + r / b;
}`;

// ---------------------------------------------------------------- SCAN
// Whole-field directional scan of the upper distance (signal 0): one whole-field pass per direction.  For one
// azimuth the padded grid is covered by parallel digital lines (minor offset round(slope * major), every cell on
// exactly one line); a thread walks one line against the azimuth for two query heights and keeps the nearest roof at
// or above each height ahead, so one pass gives every cell of the line its distance in that azimuth: 0 in such a
// roof, (steps - 1/2) x step length to the next one, the 1600 m cutoff when there is none.
// SCAN: a batch of azimuths -> raw[azimuth of the batch][height pair][pixel], two float16 distances per
// u32 (neighbouring threads: neighbouring lines).  ACCUMULATE: one thread per (height, pixel) adds the
// batch's Fourier terms to float32 sums[height][pixel][S].  The sums / 180 are then packed (PACKED FIELD).
export const scanShader = () => `
struct P { gx: u32, gy: u32, pixels: u32, layerFirst: u32,
           layerCount: u32, pairCount: u32, angleFirst: u32, angleCount: u32,
           lineMax: u32, pad0: u32, pad1: u32, pad2: u32 };
struct A { majorX: u32, majorSign: i32, minorSign: i32, maxShift: i32,
           lineCount: u32, offsetBase: u32, stepDist: f32, pad: u32 };
@group(0) @binding(0) var<storage, read> height: array<f32>;
@group(0) @binding(1) var<storage, read> layers: array<u32>;
@group(0) @binding(2) var<storage, read> offsets: array<i32>;
@group(0) @binding(3) var<storage, read> table: array<A>;
@group(0) @binding(4) var<storage, read_write> raw: array<u32>;
@group(0) @binding(5) var<uniform> u: P;
@compute @workgroup_size(${MDDF_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let g = gid.x + gid.y * groups.x * ${MDDF_WORKGROUP}u;
  let line = g % u.lineMax;
  let rest = g / u.lineMax;
  let pair = rest % u.pairCount;
  let ai = rest / u.pairCount;
  if (ai >= u.angleCount) { return; }
  let t = table[u.angleFirst + ai];
  if (line >= t.lineCount) { return; }
  let k0 = 2u * pair;
  let has1 = k0 + 1u < u.layerCount;
  let q0 = (f32(layers[u.layerFirst + k0]) + 0.5) * ${wgslFloat(LAYER_M)};
  let q1 = select(1.0e9, (f32(layers[u.layerFirst + min(k0 + 1u, u.layerCount - 1u)]) + 0.5) * ${wgslFloat(LAYER_M)}, has1);
  let cutoff = ${wgslFloat(DISTANCE_CUTOFF_M)};
  let majorSize = i32(select(u.gy, u.gx, t.majorX == 1u));
  let minorSize = i32(select(u.gx, u.gy, t.majorX == 1u));
  let lineOffset = i32(line) - t.maxShift;
  let out = (ai * u.pairCount + pair) * u.pixels;
  var o0 = -1;
  var o1 = -1;
  for (var m = majorSize - 1; m >= 0; m--) {
    let om = lineOffset + offsets[t.offsetBase + u32(m)];
    if (om < 0 || om >= minorSize) { continue; }
    let X = select(majorSize - 1 - m, m, t.majorSign > 0);
    let Y = select(minorSize - 1 - om, om, t.minorSign > 0);
    let x = select(Y, X, t.majorX == 1u);
    let y = select(X, Y, t.majorX == 1u);
    let cell = u32(y) * u.gx + u32(x);
    let h = height[cell];
    var d0 = cutoff;
    var d1 = cutoff;
    if (h >= q0) { o0 = m; d0 = 0.0; } else if (o0 >= 0) { d0 = min((f32(o0 - m) - 0.5) * t.stepDist, cutoff); }
    if (h >= q1) { o1 = m; d1 = 0.0; } else if (o1 >= 0) { d1 = min((f32(o1 - m) - 0.5) * t.stepDist, cutoff); }
    raw[out + cell] = pack2x16float(vec2<f32>(d0, d1));
  }
}`;

/** ACCUMULATE (above) for the stored channels `stored`. */
export function fourierAccumulateShader(stored) {
  // one line per stored channel: its share of the sums (the constant part, or one Fourier mode's x or y)
  const term = (c) => (c.part === "dc" ? "dc" : `${c.part}.${c.mode === 1 ? "x" : "y"}`);
  const adds = stored.map((c, i) => `  sums[e + ${i}u] += ${term(c)};`).join("\n");
  return `
struct Q { pixels: u32, layerCount: u32, pairCount: u32, angleFirst: u32, angleCount: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> raw: array<u32>;
@group(0) @binding(1) var<storage, read> trig: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> sums: array<f32>;
@group(0) @binding(3) var<uniform> u: Q;
@compute @workgroup_size(${MDDF_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let g = gid.x + gid.y * groups.x * ${MDDF_WORKGROUP}u;
  if (g >= u.pixels * u.layerCount) { return; }
  let k = g / u.pixels;
  let pixel = g % u.pixels;
  var dc = 0.0;
  var re = vec2<f32>(0.0);
  var im = vec2<f32>(0.0);
  for (var ai = 0u; ai < u.angleCount; ai++) {
    let v = unpack2x16float(raw[(ai * u.pairCount + k / 2u) * u.pixels + pixel]);
    let d = select(v.x, v.y, (k & 1u) == 1u);
    let w = trig[u.angleFirst + ai];
    dc += d;
    re += d * w.xz;
    im -= d * w.yw;
  }
  let e = (k * u.pixels + pixel) * ${stored.length}u;
${adds}
}`;
}

// ---------------------------------------------------------------- PACKED FIELD
// The stored channels of a whole city, compressed so that the largest city's field fits in memory and is still fast
// to decode.  Every value is rounded to a multiple of step = 2 x tolerance x the model's input scale of its channel
// (maximum error tolerance x scale), q = round(v / step).  Per (slot, channel) the grid is cut in 64 x 64-cell tiles
// of 8 x 8 blocks; a block stores its minimum q and the bit width of (q - minimum) (0 for a constant block), its 64
// values follow bit-packed.  Random access, decoded where it is read (assemble).
// Buffer (u32): [tile word offset, nTiles] [block header, nTiles x PACK_BLOCKS: width 4 b | minimum + 2048 12 b |
// word offset in the tile 16 b] [payload].
export const PACK_TILE = 64;
const PACK_BLOCK = 8;                                          // a block's side (cells)
const PACK_SIDE_BLOCKS = PACK_TILE / PACK_BLOCK;               // blocks along a tile's side
export const PACK_BLOCKS = PACK_SIDE_BLOCKS * PACK_SIDE_BLOCKS;    // blocks per tile: one header each
const BLOCK_CELLS = PACK_BLOCK * PACK_BLOCK;                   // values per block
const TILE_SHIFT = Math.log2(PACK_TILE), BLOCK_SHIFT = Math.log2(PACK_BLOCK), BLOCK_MASK = PACK_BLOCK - 1;
// a block header (u32, as WGSL text): the bit width in bits 0-3, the minimum + H_MIN_BIAS in bits 4-15 (clamped to
// 0..H_MIN_MAX), the block's word offset in its tile in bits 16-31
const H_WIDTH_MASK = "15u", H_MIN_SHIFT = "4u", H_MIN_MASK = "0xfffu", H_MIN_BIAS = "2048", H_MIN_MAX = "4095";
const H_OFFSET_SHIFT = "16u";
// a block's values fill whole 32-bit words (the payload pass writes a word only when it is full): its word count is
// width x BLOCK_CELLS / 32
if (BLOCK_CELLS % 32 !== 0) throw new Error("a pack block's values must fill whole 32-bit words");
/**
 * The packing passes (format above): payload false = every block's header (its minimum and bit width), true = the
 * blocks' values, bit-packed into the field after the headers.
 */
export const packShader = (stored, steps, tx, ty, payload) => `
struct E { gx: u32, gy: u32, pixels: u32, layerCount: u32, nTiles: u32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> sums: array<f32>;
@group(0) @binding(1) var<storage, read_write> ${payload ? "field" : "headers"}: array<u32>;
@group(0) @binding(2) var<uniform> u: E;
var<private> STEPS: array<f32, ${stored.length}> = array<f32, ${stored.length}>(${steps.map(lit).join(", ")});
fn qAt(k: u32, x: u32, y: u32, c: u32) -> i32 {
  return i32(round(sums[(k * u.pixels + y * u.gx + x) * ${stored.length}u + c] / ${wgslFloat(ANGLE_COUNT)} / STEPS[c]));
}
@compute @workgroup_size(${MDDF_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let g = gid.x + gid.y * groups.x * ${MDDF_WORKGROUP}u;
  if (g >= u.nTiles * ${PACK_BLOCKS}u) { return; }
  let t = g / ${PACK_BLOCKS}u;
  let b = g % ${PACK_BLOCKS}u;
  let tx = t % ${tx}u;
  let ty = (t / ${tx}u) % ${ty}u;
  let c = (t / ${tx * ty}u) % ${stored.length}u;
  let k = t / ${tx * ty * stored.length}u;
${payload ? `  let h = field[u.nTiles + t * ${PACK_BLOCKS}u + b];
  let w = h & ${H_WIDTH_MASK};
  if (w == 0u) { return; }
  let base = i32((h >> ${H_MIN_SHIFT}) & ${H_MIN_MASK}) - ${H_MIN_BIAS};
  var wi = field[t] + (h >> ${H_OFFSET_SHIFT});
  let x0 = tx * ${PACK_TILE}u + (b % ${PACK_SIDE_BLOCKS}u) * ${PACK_BLOCK}u;
  let y0 = ty * ${PACK_TILE}u + (b / ${PACK_SIDE_BLOCKS}u) * ${PACK_BLOCK}u;
  var acc = 0u;
  var nbits = 0u;
  for (var j = 0u; j < ${BLOCK_CELLS}u; j++) {
    let x = x0 + j % ${PACK_BLOCK}u; let y = y0 + j / ${PACK_BLOCK}u;
    var r = 0u;
    if (x < u.gx && y < u.gy) { r = u32(qAt(k, x, y, c) - base); }
    acc |= r << nbits;
    nbits += w;
    if (nbits >= 32u) {
      field[wi] = acc; wi += 1u; nbits -= 32u;
      acc = select(0u, r >> (w - nbits), nbits > 0u);
    }
  }` : `  let x0 = tx * ${PACK_TILE}u + (b % ${PACK_SIDE_BLOCKS}u) * ${PACK_BLOCK}u;
  let y0 = ty * ${PACK_TILE}u + (b / ${PACK_SIDE_BLOCKS}u) * ${PACK_BLOCK}u;
  var mn = 2147483647; var mx = -2147483647;
  for (var j = 0u; j < ${BLOCK_CELLS}u; j++) {
    let x = x0 + j % ${PACK_BLOCK}u; let y = y0 + j / ${PACK_BLOCK}u;
    if (x < u.gx && y < u.gy) { let q = qAt(k, x, y, c); mn = min(mn, q); mx = max(mx, q); }
  }
  if (mx < mn) { mn = 0; mx = 0; }
  let range = u32(mx - mn);
  let w = select(32u - countLeadingZeros(range), 0u, range == 0u);
  headers[t * ${PACK_BLOCKS}u + b] = w | (u32(clamp(mn + ${H_MIN_BIAS}, 0, ${H_MIN_MAX})) << ${H_MIN_SHIFT});`}
}`;

// Word offsets of the PACK_BLOCKS blocks of a tile (header bits 16-31) and the tile's word count.
export const packTileShader = () => `
@group(0) @binding(0) var<storage, read_write> headers: array<u32>;
@group(0) @binding(1) var<storage, read_write> tileWords: array<u32>;
@group(0) @binding(2) var<uniform> nTiles: vec4<u32>;
@compute @workgroup_size(${MDDF_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let t = gid.x + gid.y * groups.x * ${MDDF_WORKGROUP}u;
  if (t >= nTiles.x) { return; }
  var local = 0u;
  for (var b = 0u; b < ${PACK_BLOCKS}u; b++) {
    let h = headers[t * ${PACK_BLOCKS}u + b];
    headers[t * ${PACK_BLOCKS}u + b] = h | (local << ${H_OFFSET_SHIFT});
    local += ${BLOCK_CELLS / 32}u * (h & ${H_WIDTH_MASK});
  }
  tileWords[t] = local;
}`;

// ---------------------------------------------------------------- ALONG-WIND
// Thread per (tile of the list, cell of the tile, group of LAYER_GROUP layers).  The cell is a wind-frame cell; its
// position in the expanded grid is assemble's (the same float32 expression) and the upwind ray starts there.  Output
// element ((tile slot * layers + layer slot) * UP_TILE^2 + cell) * U + c, float16, two per u32.
export const alongWindShader = (up) => {
  const arr = `array<f32, ${LAYER_GROUP}>`;
  const value = { 0: "hs[0][k]", 1: "hs[1][k]", 2: "hs[2][k]", 3: "hs[3][k]" };
  // (the loop's `let z = query[k]` is not read; it stays: the text is part of the numeric result, see the file header)
  return `${DIV}
struct P { gx: u32, gy: u32, tileCount: u32, layerCount: u32,
           margin: u32, nx: u32, ny: u32, pad0: u32,
           upX0: i32, upY0: i32, pad1: u32, pad2: u32,
           cosU: f32, sinU: f32, cosT: f32, sinT: f32,
           centreX: f32, centreY: f32, pad3: f32, pad4: f32 };
@group(0) @binding(0) var<storage, read> height: array<f32>;
@group(0) @binding(1) var<storage, read> layers: array<u32>;
@group(0) @binding(2) var<storage, read_write> field: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read> tiles: array<i32>;
@group(0) @binding(4) var<uniform> u: P;
fn put(e: u32, v: f32) {
  let bits = pack2x16float(vec2<f32>(v, 0.0)) & 0xffffu;
  atomicOr(&field[e >> 1u], select(bits, bits << 16u, (e & 1u) == 1u));
}
@compute @workgroup_size(${MDDF_WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let g = gid.x + gid.y * groups.x * ${MDDF_WORKGROUP}u;
  let groupsPerPixel = (u.layerCount + ${LAYER_GROUP - 1}u) / ${LAYER_GROUP}u;
  let cells = ${UP_TILE * UP_TILE}u;
  if (g >= u.tileCount * cells * groupsPerPixel) { return; }
  let layerGroup = g % groupsPerPixel;
  let cell = (g / groupsPerPixel) % cells;
  let tile = g / (groupsPerPixel * cells);
  let sx = f32(u.upX0 + tiles[3u * tile] * ${UP_TILE} + i32(cell % ${UP_TILE}u));
  let sy = f32(u.upY0 + tiles[3u * tile + 1u] * ${UP_TILE} + i32(cell / ${UP_TILE}u));
  let slotTile = u32(tiles[3u * tile + 2u]);
  let dxw = sx - u.centreX;
  let dyw = sy - u.centreY;
  // expanded-grid position (cell centres at integers), as assemble's ix / iy
  let px = u.centreX + dxw * u.cosT - dyw * u.sinT + f32(u.margin);
  let py = u.centreY + dxw * u.sinT + dyw * u.cosT + f32(u.margin);
  let x0 = i32(floor(px + 0.5));
  let y0 = i32(floor(py + 0.5));
  // position inside its cell, 0..1 from the lower edge (0.5 at the centre)
  let fx = px + 0.5 - f32(x0);
  let fy = py + 0.5 - f32(y0);
  let first = layerGroup * ${LAYER_GROUP}u;
  let count = min(${LAYER_GROUP}u, u.layerCount - first);
  var query: ${arr};
  for (var k = 0u; k < ${LAYER_GROUP}u; k++) {
    query[k] = (f32(layers[min(first + k, u.layerCount - 1u)]) + 0.5) * ${wgslFloat(LAYER_M)};
  }
  // grid traversal along the single upwind azimuth, for up to LAYER_GROUP query heights at once (the cells the ray
  // crosses do not depend on the height, only where it stops and what it records)
  var hs: array<${arr}, 4>;
  {
    var x = x0; var y = y0;
    let nx = i32(u.gx);
    let ny = i32(u.gy);
    let gxr = u.cosU / ${wgslFloat(CELL_M)};
    let gyr = u.sinU / ${wgslFloat(CELL_M)};
    // first cell boundary crossings from the start position (fx, fy: its place within the start cell)
    var stepX = 0; var nextX = 1.0e20; var deltaX = 1.0e20;
    if (abs(gxr) >= 1.0e-10) {
      if (gxr > 0.0) { stepX = 1; nextX = divr(1.0 - fx, gxr); deltaX = divr(1.0, gxr); }
      else { stepX = -1; nextX = divr(-fx, gxr); deltaX = divr(-1.0, gxr); }
    }
    var stepY = 0; var nextY = 1.0e20; var deltaY = 1.0e20;
    if (abs(gyr) >= 1.0e-10) {
      if (gyr > 0.0) { stepY = 1; nextY = divr(1.0 - fy, gyr); deltaY = divr(1.0, gyr); }
      else { stepY = -1; nextY = divr(-fy, gyr); deltaY = divr(-1.0, gyr); }
    }
    let cutoff = ${wgslFloat(DISTANCE_CUTOFF_M)};
    var bestNum: ${arr};
    var bestDen: ${arr};
    var done: array<bool, ${LAYER_GROUP}>;
    for (var k = 0u; k < ${LAYER_GROUP}u; k++) {
      hs[0][k] = cutoff; hs[1][k] = 0.0; hs[2][k] = cutoff; hs[3][k] = -query[k];
      bestNum[k] = -1.0e20; bestDen[k] = 1.0;
      done[k] = k >= count;
    }
    var remaining = count;
    var entered = 0.0;
    for (var step = 0u; step < ${RAY_STEP_LIMIT}u; step++) {
      if (x < 0 || x >= nx || y < 0 || y >= ny || entered >= cutoff || remaining == 0u) { break; }
      let leaving = min(min(nextX, nextY), cutoff);
      let roof = height[u32(y) * u.gx + u32(x)];
      if (roof > 0.0) {
        for (var k = 0u; k < ${LAYER_GROUP}u; k++) {
          if (done[k]) { continue; }
          if (roof >= query[k]) {
            hs[0][k] = entered; hs[1][k] = roof - query[k]; hs[2][k] = entered; hs[3][k] = 0.0;
            done[k] = true;
            remaining -= 1u;
          } else {
            let num = roof - query[k];
            let den = max(leaving, 1.0e-8);
            if (num * bestDen[k] > bestNum[k] * den) { bestNum[k] = num; bestDen[k] = den; hs[2][k] = leaving; hs[3][k] = roof - query[k]; }
          }
        }
      }
      let crossX = nextX <= nextY + 1.0e-7;
      let crossY = nextY <= nextX + 1.0e-7;
      if (crossX) { x += stepX; nextX += deltaX; }
      if (crossY) { y += stepY; nextY += deltaY; }
      entered = leaving;
    }
  }
  for (var k = 0u; k < count; k++) {
    let z = query[k];
    let e0 = ((slotTile * u.layerCount + first + k) * cells + cell) * ${up.length}u;
${up.map((c, i) => `    put(e0 + ${i}u, ${value[c]});`).join("\n")}
  }
}`;
};

// ---------------------------------------------------------------- ASSEMBLE
// One thread per element of the model input [B, C, 1, Y, X].  Wind-frame
// position -> city position -> per-channel value from the manifest.
export const assembleShader = (store, inputPlan, packed) => {
  const S = inputPlan.stored.length, U = inputPlan.up.length;
  const { usesHeight } = inputPlan;
  const bindings = [["field", "array<u32>"]];
  if (usesHeight) bindings.push(["height", "array<f32>"]);
  bindings.push(["origins", "array<i32>"], ["slots", "array<i32>"]);
  if (U) bindings.push(["up", "array<u32>"], ["upTable", "array<i32>"]);
  const declare = bindings.map(([name, type], i) => `@group(0) @binding(${i}) var<storage, read> ${name}: ${type};`);
  declare.push(`@group(0) @binding(${bindings.length}) var<storage, read_write> out: array<${store}>;`);
  declare.push(`@group(0) @binding(${bindings.length + 1}) var<uniform> u: P;`);
  const norm = (expr, e) => `(${expr} - ${lit(e.center)}) / ${lit(e.scale)}`;
  const cases = inputPlan.inputs.map((e, c) => {
    let body;
    if (e.source === "r1") {
      if (e.part === "dc") {
        body = `value = ${norm(`sampleField(ry, rx, slot, ${inputPlan.storedIndex(e.signal, "dc", 0)}u)`, e)};`;
      } else {
        const re = inputPlan.storedIndex(e.signal, "re", e.mode), im = inputPlan.storedIndex(e.signal, "im", e.mode);
        body = `let re = sampleField(ry, rx, slot, ${re}u);
      let im = sampleField(ry, rx, slot, ${im}u);
      let angle = ${lit(e.mode)} * u.theta;
      let c = cos(angle); let s = sin(angle);
      value = ${norm(e.part === "im" ? "(re * s + im * c)" : "(re * c - im * s)", e)};`;
      }
    } else if (e.source === "geo") {
      const f = e.index === 0 ? "h / 100.0" : "asinh((h - z) / 10.0)";
      body = `let h = roofHeight(iy, ix);
      value = ${norm(`f16round(${f})`, e)};`;
    } else if (e.source === "up") {
      body = `value = ${norm(`upAtWind(sy, sx, slot, ${inputPlan.up.indexOf(e.index)}u)`, e)};`;
    } else if (e.source === "solid") {
      body = `let roofLayer = round(roofHeight(iy, ix) / ${wgslFloat(LAYER_M)});
      value = select(0.0, 1.0, roofLayer > f32(layer));`;
    } else if (e.axis === "y") {
      body = `value = (sy - u.centreY) / ${lit(e.scale)};`;
    } else if (e.axis === "z") {
      body = `value = (f32(layer) - ${lit(e.center)}) / ${lit(e.scale)};`;
    } else {
      body = `value = (sx - ${lit(e.center)}) / ${lit(e.scale)};`;
    }
    return `    case ${c}u: {  // ${e.name}
      ${body}
    }`;
  });
  return `${store === "f16" ? "enable f16;\n" : ""}
struct P {
  batch: u32, channels: u32, py: u32, px: u32,
  gx: u32, gy: u32, regionX: u32, regionY: u32,
  regionW: u32, regionH: u32, regionPixels: u32, margin: u32,
  cosT: f32, sinT: f32, theta: f32, centreY: f32,
  centreX: f32, packTiles: u32, pad1: f32, pad2: f32,
  upX0: i32, upY0: i32, upTW: u32, upTH: u32,
  upLayers: u32, pad3: u32, pad4: u32, pad5: u32,
};
${declare.join("\n")}

// Round to the nearest float16 value (ties to even), as numpy astype(float16).  Done with bit
// operations: the shader compiler may fold unpack2x16float(pack2x16float(v)) away.
fn f16round(v: f32) -> f32 {
  if (abs(v) < 6.103515625e-5) { return round(v * 16777216.0) / 16777216.0; }   // float16 subnormals
  let b = bitcast<u32>(v);
  return bitcast<f32>((b + 0x0fffu + ((b >> 13u) & 1u)) & 0xffffe000u);
}
var<private> STEPS: array<f32, ${S}> = array<f32, ${S}>(${packed.steps.map(lit).join(", ")});
// one value of the packed field (region = whole padded grid)
fn storedAt(pixel: u32, slot: u32, c: u32) -> f32 {
  let x = pixel % u.regionW;
  let y = pixel / u.regionW;
  let t = ((slot * ${S}u + c) * ${packed.ty}u + (y >> ${TILE_SHIFT}u)) * ${packed.tx}u + (x >> ${TILE_SHIFT}u);
  let h = field[u.packTiles + t * ${PACK_BLOCKS}u + ((y >> ${BLOCK_SHIFT}u) & ${BLOCK_MASK}u) * ${PACK_SIDE_BLOCKS}u + ((x >> ${BLOCK_SHIFT}u) & ${BLOCK_MASK}u)];
  let w = h & ${H_WIDTH_MASK};
  var q = i32((h >> ${H_MIN_SHIFT}) & ${H_MIN_MASK}) - ${H_MIN_BIAS};
  if (w > 0u) {
    let bit = (((y & ${BLOCK_MASK}u) * ${PACK_BLOCK}u) + (x & ${BLOCK_MASK}u)) * w;
    let word = field[t] + (h >> ${H_OFFSET_SHIFT}) + (bit >> 5u);
    let sh = bit & 31u;
    var r = field[word] >> sh;
    if (sh + w > 32u) { r |= field[word + 1u] << (32u - sh); }
    q += i32(r & ((1u << w) - 1u));
  }
  return f32(q) * STEPS[c];
}
${U ? `// along-wind channels computed at the wind-frame cells (no interpolation): tile table -> tile slot -> value
fn upAtWind(sy: f32, sx: f32, slot: u32, c: u32) -> f32 {
  let tx = i32(sx) - u.upX0;
  let ty = i32(sy) - u.upY0;
  if (tx < 0 || ty < 0 || u32(tx >> ${Math.log2(UP_TILE)}u) >= u.upTW || u32(ty >> ${Math.log2(UP_TILE)}u) >= u.upTH) { return 0.0; }
  let t = upTable[u32(ty >> ${Math.log2(UP_TILE)}u) * u.upTW + u32(tx >> ${Math.log2(UP_TILE)}u)];
  if (t < 0) { return 0.0; }
  let e = ((u32(t) * u.upLayers + slot) * ${UP_TILE * UP_TILE}u + u32(ty & ${UP_TILE - 1}) * ${UP_TILE}u + u32(tx & ${UP_TILE - 1})) * ${U}u + c;
  let pair = unpack2x16float(up[e >> 1u]);
  return select(pair.x, pair.y, (e & 1u) == 1u);
}` : ""}
${usesHeight ? `fn roofHeight(iy: f32, ix: f32) -> f32 {
  let hy = clamp(i32(round(iy)), 0, i32(u.gy) - 1);
  let hx = clamp(i32(round(ix)), 0, i32(u.gx) - 1);
  return height[u32(hy) * u.gx + u32(hx)];
}` : ""}

// Bilinear sample of the stored field at region coordinates (ry, rx).
fn sampleField(ry: f32, rx: f32, slot: u32, c: u32) -> f32 {
  let y0 = floor(ry); let x0 = floor(rx);
  let fy = ry - y0; let fx = rx - x0;
  let iy0 = u32(clamp(i32(y0), 0, i32(u.regionH) - 1));
  let ix0 = u32(clamp(i32(x0), 0, i32(u.regionW) - 1));
  let iy1 = min(iy0 + 1u, u.regionH - 1u);
  let ix1 = min(ix0 + 1u, u.regionW - 1u);
  let v00 = storedAt(iy0 * u.regionW + ix0, slot, c);
  let v01 = storedAt(iy0 * u.regionW + ix1, slot, c);
  let v10 = storedAt(iy1 * u.regionW + ix0, slot, c);
  let v11 = storedAt(iy1 * u.regionW + ix1, slot, c);
  return mix(mix(v00, v01, fx), mix(v10, v11, fx), fy);
}

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  let area = u.py * u.px;
  if (g >= u.batch * u.channels * area) { return; }
  let p = g % area;
  let channel = (g / area) % u.channels;
  let b = g / (area * u.channels);
  let x = p % u.px;
  let y = p / u.px;
  // Patch origin in wind-frame cells: [sy, layer, sx].
  let sy = f32(origins[b * 3u] + i32(y));
  let layer = u32(origins[b * 3u + 1u]);
  let sx = f32(origins[b * 3u + 2u] + i32(x));
  let dx = sx - u.centreX;
  let dy = sy - u.centreY;
  // City cell, then expanded-grid cell, then region cell.
  let ix = u.centreX + dx * u.cosT - dy * u.sinT + f32(u.margin);
  let iy = u.centreY + dx * u.sinT + dy * u.cosT + f32(u.margin);
  let ry = iy - f32(u.regionY);
  let rx = ix - f32(u.regionX);
  let z = (f32(layer) + 0.5) * ${wgslFloat(LAYER_M)};
  let slot = u32(slots[layer]);
  var value = 0.0;
  switch channel {
${cases.join("\n")}
    default: {}
  }
  out[g] = ${store === "f16" ? "f16(value)" : "value"};
}`;
};
