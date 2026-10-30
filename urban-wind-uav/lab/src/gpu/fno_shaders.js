// The WGSL compute shaders of the wind model's Fourier neural operator (fno.js).  The generated source text is part of
// the numeric result: change it only on purpose.
import { WORKGROUP } from "./context.js";

const COMMON = `
fn erf_approx(x: f32) -> f32 {
  let s = sign(x);
  let ax = abs(x);
  let t = 1.0 / (1.0 + 0.3275911 * ax);
  let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
      - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax);
  return s * y;
}
fn gelu(x: f32) -> f32 {
  return 0.5 * x * (1.0 + erf_approx(x * 0.7071067811865476));
}
`;

// Contract mode `mode` of a five-dimensional complex tensor with a factor.
export const CONTRACT = `
struct P { d0: u32, d1: u32, d2: u32, d3: u32,
           d4: u32, mode: u32, outer: u32, rank: u32,
           strideMode: u32, blockAbove: u32, total: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> facRe: array<f32>;
@group(0) @binding(3) var<storage, read> facIm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(5) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(6) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  if (g >= u.total) { return; }
  // Split the flat output index around the contracted mode.
  let inner = g % u.strideMode;
  let a = (g / u.strideMode) % u.outer;
  let above = g / (u.strideMode * u.outer);
  var accRe = 0.0;
  var accIm = 0.0;
  for (var r = 0u; r < u.rank; r++) {
    let source = (above * u.rank + r) * u.strideMode + inner;
    let xr = inRe[source];
    let xi = inIm[source];
    let fr = facRe[a * u.rank + r];
    let fi = facIm[a * u.rank + r];
    accRe += xr * fr - xi * fi;
    accIm += xr * fi + xi * fr;
  }
  outRe[g] = accRe;
  outIm[g] = accIm;
}`;

// InstanceNorm, one workgroup per (sample, channel).  It takes no parameters, so the dispatch size is the only input
// and there is no uniform binding.
//
// Two passes on purpose: the single-pass identity E[x^2] - mean^2 loses all significance when a channel has a large
// mean and a small spread, and the cancellation error can push the variance below zero, at which point inverseSqrt
// yields NaN for the whole channel.  Summing squared deviations cannot go negative.
//
// The epsilon is a parameter because the spectral branch stores its result pre-scaled: InstanceNorm is invariant to a
// constant factor k on its input provided epsilon is replaced by epsilon / k^2, which makes the scaling exact rather
// than an approximation.
function instanceNormShader(store, area, epsilon) {
  const { enable, load, save } = storage(store);
  return `${enable}
@group(0) @binding(0) var<storage, read> X: array<${store}>;
@group(0) @binding(1) var<storage, read_write> Y: array<${store}>;
var<workgroup> partial: array<f32, ${WORKGROUP}>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let base = wid.x * ${area}u;
  var s = 0.0;
  for (var i = lid.x; i < ${area}u; i += ${WORKGROUP}u) { s += ${load("X[base + i]")}; }
  partial[lid.x] = s;
  workgroupBarrier();
  var stride = ${WORKGROUP / 2}u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] += partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let mean = partial[0] / ${area}.0;
  workgroupBarrier();

  var q = 0.0;
  for (var i = lid.x; i < ${area}u; i += ${WORKGROUP}u) {
    let d = ${load("X[base + i]")} - mean;
    q += d * d;
  }
  partial[lid.x] = q;
  workgroupBarrier();
  stride = ${WORKGROUP / 2}u;
  while (stride > 0u) {
    if (lid.x < stride) { partial[lid.x] += partial[lid.x + stride]; }
    workgroupBarrier();
    stride = stride >> 1u;
  }
  let inverse = inverseSqrt(max(partial[0] / ${area}.0, 0.0) + ${epsilon.toExponential(12)});
  for (var i = lid.x; i < ${area}u; i += ${WORKGROUP}u) {
    Y[base + i] = ${save("(" + load("X[base + i]") + " - mean) * inverse")};
  }
}`;
}

// The spectral result peaks near 8.2e4 over the tested cities and heights, beyond the 65504 limit of f16.  Storing it
// pre-divided by this power of two keeps 25x of headroom while staying far above the subnormal range; the following
// InstanceNorm undoes it exactly through its epsilon.
const SPECTRAL_SCALE = 32;

// InstanceNorm's epsilon.
const NORM_EPSILON = 1.0e-5;

// Output channels per thread in the 1x1 convolution.
export const CONV_BLOCK = 16;

/** The WGSL of one storage type ("f16" | "f32"): the enable line, and a stored value read as f32 / written back. */
function storage(store) {
  return {
    enable: store === "f16" ? "enable f16;\n" : "",
    load: (expr) => store === "f16" ? `f32(${expr})` : expr,
    save: (expr) => store === "f16" ? `f16(${expr})` : expr,
  };
}

/**
 * 1x1 convolution.  Each thread computes CONV_BLOCK output channels of one pixel, so every input value is read once
 * per block instead of once per output channel; the block's weights sit in workgroup memory.  The WORKGROUP threads
 * of a workgroup share the sample and the block (the patch area is a multiple of WORKGROUP).  Same sum, in the same
 * order, as one thread per output value.
 */
function conv1x1Shader(spec) {
  const { store, patch } = spec;
  const area = patch * patch;
  const { enable, load, save } = storage(store);
  return `${enable}${COMMON}
struct P { batch: u32, cin: u32, cout: u32, act: u32 };
@group(0) @binding(0) var<storage, read> X: array<${store}>;
@group(0) @binding(1) var<storage, read> W: array<${store}>;
@group(0) @binding(2) var<storage, read> B: array<${store}>;
@group(0) @binding(3) var<storage, read_write> Y: array<${store}>;
@group(0) @binding(4) var<uniform> u: P;
const OB = ${CONV_BLOCK}u;
var<workgroup> w: array<f32, ${CONV_BLOCK * spec.cinMax}>;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(workgroup_id) wid: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear workgroup index over a 2-D grid (a single dimension is capped at 65535).
  let wg = wid.x + wid.y * groups.x;
  let blocks = (u.cout + OB - 1u) / OB;
  let perSample = blocks * ${area / WORKGROUP}u;
  if (wg >= u.batch * perSample) { return; }       // whole workgroups only: uniform
  let b = wg / perSample;
  let ob = (wg % perSample) / ${area / WORKGROUP}u;
  let p = (wg % ${area / WORKGROUP}u) * ${WORKGROUP}u + lid.x;
  let o0 = ob * OB;
  for (var k = lid.x; k < OB * u.cin; k += ${WORKGROUP}u) {
    let o = o0 + k / u.cin;
    w[k] = select(0.0, ${load("W[min(o, u.cout - 1u) * u.cin + k % u.cin]")}, o < u.cout);
  }
  workgroupBarrier();
  var acc: array<f32, ${CONV_BLOCK}>;
  for (var j = 0u; j < OB; j++) { acc[j] = ${load("B[min(o0 + j, u.cout - 1u)]")}; }
  let xBase = b * u.cin * ${area}u + p;
  for (var i = 0u; i < u.cin; i++) {
    let x = ${load("X[xBase + i * " + area + "u]")};
    for (var j = 0u; j < OB; j++) { acc[j] += w[j * u.cin + i] * x; }
  }
  for (var j = 0u; j < OB; j++) {
    let o = o0 + j;
    if (o >= u.cout) { break; }
    var v = acc[j];
    if (u.act == 1u) { v = gelu(v); }
    Y[(b * u.cout + o) * ${area}u + p] = ${save("v")};
  }
}`;
}

/** The gate: the first half of the channels times the sigmoid of the second half, through GELU. */
function gateShader(spec) {
  const { store, patch } = spec;
  const area = patch * patch;
  const { enable, load, save } = storage(store);
  return `${enable}${COMMON}
struct P { batch: u32, mid: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> G: array<${store}>;
@group(0) @binding(1) var<storage, read_write> H: array<${store}>;
@group(0) @binding(2) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  if (g >= u.batch * u.mid * ${area}u) { return; }
  let p = g % ${area}u;
  let m = (g / ${area}u) % u.mid;
  let b = g / (${area}u * u.mid);
  let value = ${load("G[(b * 2u * u.mid + m) * " + area + "u + p]")};
  let gateValue = ${load("G[(b * 2u * u.mid + u.mid + m) * " + area + "u + p]")};
  H[g] = ${save("gelu(value * (1.0 / (1.0 + exp(-gateValue))))")};
}`;
}

/** Forward DFT along x, truncated to modes2 columns. */
function forwardColumnsShader(spec) {
  const { store, patch, modes2 } = spec;
  const area = patch * patch;
  const { enable, load } = storage(store);
  return `${enable}
struct P { batch: u32, channels: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> X: array<${store}>;
@group(0) @binding(1) var<storage, read> tableCos: array<f32>;
@group(0) @binding(2) var<storage, read> tableSin: array<f32>;
@group(0) @binding(3) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(4) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(5) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  if (g >= u.batch * u.channels * ${patch * modes2}u) { return; }
  let k = g % ${modes2}u;
  let y = (g / ${modes2}u) % ${patch}u;
  let bc = g / ${patch * modes2}u;
  var accRe = 0.0; var accIm = 0.0;
  let base = bc * ${area}u + y * ${patch}u;
  for (var x = 0u; x < ${patch}u; x++) {
    let v = ${load("X[base + x]")};
    accRe += v * tableCos[x * ${modes2}u + k];
    accIm += v * tableSin[x * ${modes2}u + k];
  }
  outRe[g] = accRe; outIm[g] = accIm;
}`;
}

/**
 * Complex matrix product along y; used for both the forward reduction (patch -> 2*modes1 rows) and the inverse
 * expansion.
 */
function rowMixShader(spec) {
  const { modes2 } = spec;
  return `
struct P { batch: u32, channels: u32, rows: u32, inner: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> tableRe: array<f32>;
@group(0) @binding(3) var<storage, read> tableIm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(5) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(6) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  if (g >= u.batch * u.channels * u.rows * ${modes2}u) { return; }
  let k = g % ${modes2}u;
  let r = (g / ${modes2}u) % u.rows;
  let bc = g / (${modes2}u * u.rows);
  var accRe = 0.0; var accIm = 0.0;
  let inBase = bc * u.inner * ${modes2}u + k;
  let tableBase = r * u.inner;
  for (var j = 0u; j < u.inner; j++) {
    let ar = inRe[inBase + j * ${modes2}u];
    let ai = inIm[inBase + j * ${modes2}u];
    let cr = tableRe[tableBase + j];
    let ci = tableIm[tableBase + j];
    accRe += ar * cr - ai * ci;
    accIm += ar * ci + ai * cr;
  }
  outRe[g] = accRe; outIm[g] = accIm;
}`;
}

/** Channel mixing by the layer's spectral weights, per retained mode. */
function spectralMixShader(spec) {
  const { modes1, modes2 } = spec;
  const rows = 2 * modes1;
  return `
struct P { batch: u32, channels: u32, stackBase: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> wRe: array<f32>;
@group(0) @binding(3) var<storage, read> wIm: array<f32>;
@group(0) @binding(4) var<storage, read_write> outRe: array<f32>;
@group(0) @binding(5) var<storage, read_write> outIm: array<f32>;
@group(0) @binding(6) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  let block = ${rows * modes2}u;
  if (g >= u.batch * u.channels * block) { return; }
  let k = g % ${modes2}u;
  let s = (g / ${modes2}u) % ${rows}u;
  let o = (g / block) % u.channels;
  let b = g / (block * u.channels);
  let half = select(0u, 1u, s >= ${modes1}u);
  let sp = s - half * ${modes1}u;
  var accRe = 0.0; var accIm = 0.0;
  let modeArea = ${modes1 * modes2}u;
  let wBase = ((u.stackBase + half) * u.channels * u.channels + o) * modeArea
            + sp * ${modes2}u + k;
  let xBase = (b * u.channels * ${rows}u + s) * ${modes2}u + k;
  for (var i = 0u; i < u.channels; i++) {
    let xr = inRe[xBase + i * block];
    let xi = inIm[xBase + i * block];
    let wr = wRe[wBase + i * u.channels * modeArea];
    let wi = wIm[wBase + i * u.channels * modeArea];
    accRe += xr * wr - xi * wi;
    accIm += xr * wi + xi * wr;
  }
  outRe[g] = accRe; outIm[g] = accIm;
}`;
}

/** Inverse DFT along x back to the patch grid (real output). */
function inverseColumnsShader(spec) {
  const { store, patch, modes2 } = spec;
  const area = patch * patch;
  const { enable, save } = storage(store);
  return `${enable}
struct P { batch: u32, channels: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> inRe: array<f32>;
@group(0) @binding(1) var<storage, read> inIm: array<f32>;
@group(0) @binding(2) var<storage, read> tableCos: array<f32>;
@group(0) @binding(3) var<storage, read> tableSin: array<f32>;
@group(0) @binding(4) var<storage, read_write> Y: array<${store}>;
@group(0) @binding(5) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  if (g >= u.batch * u.channels * ${area}u) { return; }
  let p = g % ${area}u;
  let x = p % ${patch}u;
  let y = p / ${patch}u;
  let bc = g / ${area}u;
  var acc = 0.0;
  let base = (bc * ${patch}u + y) * ${modes2}u;
  for (var k = 0u; k < ${modes2}u; k++) {
    acc += inRe[base + k] * tableCos[k * ${patch}u + x]
         + inIm[base + k] * tableSin[k * ${patch}u + x];
  }
  Y[g] = ${save("acc * " + (1 / SPECTRAL_SCALE))};
}`;
}

/** The residual sum Y = A + B + Y, through GELU when u.act is 1. */
function residualShader(spec) {
  const { store } = spec;
  const { enable, load, save } = storage(store);
  return `${enable}${COMMON}
struct P { total: u32, act: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> A: array<${store}>;
@group(0) @binding(1) var<storage, read> B: array<${store}>;
@group(0) @binding(2) var<storage, read_write> Y: array<${store}>;
@group(0) @binding(3) var<uniform> u: P;
@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>,
        @builtin(num_workgroups) groups: vec3<u32>) {
  // Linear index over a 2-D grid: a single dimension is capped at
  // 65535 workgroups, which large batches exceed.
  let g = gid.x + gid.y * groups.x * ${WORKGROUP}u;
  if (g >= u.total) { return; }
  var acc = ${load("A[g]")} + ${load("B[g]")} + ${load("Y[g]")};
  if (u.act == 1u) { acc = gelu(acc); }
  Y[g] = ${save("acc")};
}`;
}

/**
 * Every pass's WGSL source by name (conv1x1, gate, instanceNorm, instanceNormScaled, forwardColumns, rowMix,
 * spectralMix, inverseColumns, residual) for spec { store ("f16" | "f32"), patch, modes1, modes2, cinMax }.
 */
export function shaders(spec) {
  const { store, patch } = spec, area = patch * patch;
  return {
    conv1x1: conv1x1Shader(spec),
    gate: gateShader(spec),
    instanceNorm: instanceNormShader(store, area, NORM_EPSILON),
    instanceNormScaled: instanceNormShader(store, area, NORM_EPSILON / (SPECTRAL_SCALE * SPECTRAL_SCALE)),
    forwardColumns: forwardColumnsShader(spec),
    rowMix: rowMixShader(spec),
    spectralMix: spectralMixShader(spec),
    inverseColumns: inverseColumnsShader(spec),
    residual: residualShader(spec),
  };
}
