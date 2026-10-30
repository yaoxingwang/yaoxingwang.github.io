// Tricubic B-spline fields: coefficients from grid samples (the layout and filter of the paper's interpolation: edge
// padding by SPLINE_PAD cells, then the mirror spline filter along every axis), and evaluation of the value and its
// gradient.  Layout of samples and coefficients: ((i0 * n1 + i1) * n2 + i2) * channels + ch.
import { bsplineWeights } from "../shared/numeric.js";

/** Padding (samples) added on every side of a coefficient box (the cubic B-spline's support around a sample). */
const SPLINE_PAD = 3;

/**
 * The spline of padded coefficients `coef` (dims = padded sizes, SPLINE_PAD more cells on every side), sample
 * spacing `spacing` (metres per axis; sample 0 at the local origin).  Returns evaluate(p, out, grad): the values at
 * local point p into out[channels], the gradient (per metre) into grad[ch * 3 + axis] when given.  Outside the grid
 * the point is clamped to it.
 */
export function makeSpline3(coef, dims, channels, spacing) {
  const pad = SPLINE_PAD;
  const [N0, N1, N2] = dims;
  const wx = new Float64Array(4), dx = new Float64Array(4), wy = new Float64Array(4), dy = new Float64Array(4),
        wz = new Float64Array(4), dz = new Float64Array(4);
  const lo = [1 - pad, 1 - pad, 1 - pad];
  const hi = [N0 - pad - 3, N1 - pad - 3, N2 - pad - 3];
  const S1 = N2 * channels, S0 = N1 * S1;
  return function evaluate(p, out, grad) {
    const f0 = p[0] / spacing[0], f1 = p[1] / spacing[1], f2 = p[2] / spacing[2];
    let b0 = Math.floor(f0), b1 = Math.floor(f1), b2 = Math.floor(f2);
    let t0 = f0 - b0, t1 = f1 - b1, t2 = f2 - b2;
    if (b0 < lo[0]) { b0 = lo[0]; t0 = 0; } else if (b0 > hi[0]) { b0 = hi[0]; t0 = 1; }
    if (b1 < lo[1]) { b1 = lo[1]; t1 = 0; } else if (b1 > hi[1]) { b1 = hi[1]; t1 = 1; }
    if (b2 < lo[2]) { b2 = lo[2]; t2 = 0; } else if (b2 > hi[2]) { b2 = hi[2]; t2 = 1; }
    bsplineWeights(t0, wx, dx); bsplineWeights(t1, wy, dy); bsplineWeights(t2, wz, dz);
    for (let ch = 0; ch < channels; ch++) {
      out[ch] = 0;
      if (grad) { grad[ch * 3] = 0; grad[ch * 3 + 1] = 0; grad[ch * 3 + 2] = 0; }
    }
    const base = (b0 - 1 + pad) * S0 + (b1 - 1 + pad) * S1 + (b2 - 1 + pad) * channels;
    for (let a = 0; a < 4; a++) {
      for (let b = 0; b < 4; b++) {
        const wab = wx[a] * wy[b], gab0 = dx[a] * wy[b], gab1 = wx[a] * dy[b];
        let o = base + a * S0 + b * S1;
        for (let c = 0; c < 4; c++, o += channels) {
          const w = wab * wz[c];
          if (grad) {
            const g0 = gab0 * wz[c], g1 = gab1 * wz[c], g2 = wab * dz[c];
            for (let ch = 0; ch < channels; ch++) {
              const v = coef[o + ch];
              out[ch] += w * v; grad[ch * 3] += g0 * v; grad[ch * 3 + 1] += g1 * v; grad[ch * 3 + 2] += g2 * v;
            }
          } else {
            for (let ch = 0; ch < channels; ch++) out[ch] += w * coef[o + ch];
          }
        }
      }
    }
    if (grad) for (let ch = 0; ch < channels; ch++) {
      grad[ch * 3] /= spacing[0]; grad[ch * 3 + 1] /= spacing[1]; grad[ch * 3 + 2] /= spacing[2];
    }
    return out;
  };
}

// The cubic B-spline interpolation filter (mirror boundary): one causal and one anti-causal recursion with the pole
// sqrt(3) - 2, the causal start from the first HORIZON samples.
const POLE = Math.sqrt(3) - 2;
const HORIZON = 30;
/** The n samples c[base + k stride] replaced in place by their cubic B-spline coefficients. */
function prefilterLine(c, n, base, stride) {
  const horizon = Math.min(n, HORIZON);
  let zn = POLE, sum = c[base];
  for (let k = 1; k < horizon; k++) { sum += zn * c[base + k * stride]; zn *= POLE; }
  let prev = 6 * sum;
  c[base] = prev;
  for (let k = 1; k < n; k++) { prev = 6 * c[base + k * stride] + POLE * prev; c[base + k * stride] = prev; }
  let next = (POLE / (POLE * POLE - 1)) * (c[base + (n - 1) * stride] + POLE * c[base + (n - 2) * stride]);
  c[base + (n - 1) * stride] = next;
  for (let k = n - 2; k >= 0; k--) { next = POLE * (next - c[base + k * stride]); c[base + k * stride] = next; }
}

/** Padded spline coefficients of a whole box of samples: { coef: Float32Array, dims: padded sizes }. */
export function coefficientsFromSamples(data, dims, channels) {
  const pad = SPLINE_PAD;
  const [n0, n1, n2] = dims;
  const N0 = n0 + 2 * pad, N1 = n1 + 2 * pad, N2 = n2 + 2 * pad;
  const c = new Float64Array(N0 * N1 * N2 * channels);
  for (let a = 0; a < N0; a++) {
    const sa = Math.min(n0 - 1, Math.max(0, a - pad));
    for (let b = 0; b < N1; b++) {
      const sb = Math.min(n1 - 1, Math.max(0, b - pad));
      for (let k = 0; k < N2; k++) {
        const sk = Math.min(n2 - 1, Math.max(0, k - pad));
        const o = ((a * N1 + b) * N2 + k) * channels, so = ((sa * n1 + sb) * n2 + sk) * channels;
        for (let ch = 0; ch < channels; ch++) c[o + ch] = data[so + ch];
      }
    }
  }
  const strides = [N1 * N2 * channels, N2 * channels, channels], sizes = [N0, N1, N2];
  for (let axis = 0; axis < 3; axis++) {
    const [p0, p1] = [0, 1, 2].filter((x) => x !== axis);
    for (let p = 0; p < sizes[p0]; p++) for (let q = 0; q < sizes[p1]; q++) for (let ch = 0; ch < channels; ch++) {
      prefilterLine(c, sizes[axis], p * strides[p0] + q * strides[p1] + ch, strides[axis]);
    }
  }
  return { coef: Float32Array.from(c), dims: [N0, N1, N2] };
}

// Work storage of coefficientsFromRegion, kept from plan to plan: a new large array per plan would leave its garbage
// collection to the planning that follows.
let regionWork = null, regionLine = null;

/**
 * coefficientsFromSamples on a region only: the samples (a, b, every k) with lo[a] <= b <= hi[a], a convex region.
 * Every line of the region is padded at its own two ends by SPLINE_PAD copies of its end value and filtered as a whole
 * line is filtered, so inside the region the coefficients are those of the whole-box spline up to the region
 * boundary's influence (a factor 0.268 per sample from it).  Coefficients outside the region are not computed (they
 * stay 0): the spline is meant to be read inside it, at least 2 samples from its boundary.  Same layout and padded
 * dims as coefficientsFromSamples.  alloc: (length) -> a zeroed Float32Array to fill (e.g. in a shared buffer), or
 * null for a new one.
 */
export function coefficientsFromRegion(data, dims, channels, lo, hi, alloc) {
  const pad = SPLINE_PAD;
  const [n0, n1, n2] = dims;
  const N0 = n0 + 2 * pad, N1 = n1 + 2 * pad, N2 = n2 + 2 * pad;
  const S0 = N1 * N2 * channels, S1 = N2 * channels;
  // region cells stored compactly: cell (a, b) at start[a] + b - lo[a], each with its N2 x channels values
  const start = new Int32Array(n0 + 1);
  for (let a = 0; a < n0; a++) start[a + 1] = start[a] + Math.max(0, hi[a] - lo[a] + 1);
  const size = start[n0] * S1;
  if (!regionWork || regionWork.length < size) regionWork = new Float64Array(Math.ceil(size * 1.25));
  const w = regionWork;
  // a block of whole cells (S1 values each) of one line, with `pad` cells before and after
  const blockLength = (Math.max(n0, n1) + 2 * pad) * S1;
  if (!regionLine || regionLine.length < blockLength) regionLine = new Float64Array(blockLength);
  const t = regionLine;
  const at = (a, b) => (start[a] + b - lo[a]) * S1;
  // samples, padded in height by edge copies (as coefficientsFromSamples)
  for (let a = 0; a < n0; a++) for (let b = lo[a]; b <= hi[a]; b++) {
    const o = at(a, b), so = (a * n1 + b) * n2 * channels;
    for (let k = 0; k < N2; k++) {
      const sk = Math.min(n2 - 1, Math.max(0, k - pad));
      for (let ch = 0; ch < channels; ch++) w[o + k * channels + ch] = data[so + sk * channels + ch];
    }
  }
  // `length` cells gathered into t after `pad` cells: padded at both ends by copies of the end cells, every one of the
  // S1 value lines filtered (stride S1)
  const filter = (length) => {
    for (let j = 0; j < pad; j++) {
      t.copyWithin(j * S1, pad * S1, (pad + 1) * S1);
      t.copyWithin((pad + length + j) * S1, (pad + length - 1) * S1, (pad + length) * S1);
    }
    for (let i = 0; i < S1; i++) prefilterLine(t, length + 2 * pad, i, S1);
  };
  // along a: every run of consecutive region cells of every row b (one run per row for a convex region)
  for (let b = 0; b < n1; b++) {
    for (let a = 0; a < n0; a++) {
      if (b < lo[a] || b > hi[a]) continue;
      let e = a;
      while (e + 1 < n0 && b >= lo[e + 1] && b <= hi[e + 1]) e++;
      for (let x = a; x <= e; x++) { const o = at(x, b); t.set(w.subarray(o, o + S1), (pad + x - a) * S1); }
      filter(e - a + 1);
      for (let x = a; x <= e; x++) w.set(t.subarray((pad + x - a) * S1, (pad + x - a + 1) * S1), at(x, b));
      a = e;
    }
  }
  // along b: every column a (contiguous in the work storage)
  for (let a = 0; a < n0; a++) {
    const length = hi[a] - lo[a] + 1;
    if (length <= 0) continue;
    const o = at(a, lo[a]);
    t.set(w.subarray(o, o + length * S1), pad * S1);
    filter(length);
    w.set(t.subarray(pad * S1, (pad + length) * S1), o);
  }
  // along k, then into the output (outside the region the output stays zero: a new array, or alloc's zeroed one)
  const coef = alloc ? alloc(N0 * N1 * N2 * channels) : new Float32Array(N0 * N1 * N2 * channels);
  for (let a = 0; a < n0; a++) for (let b = lo[a]; b <= hi[a]; b++) {
    const o = at(a, b), co = (a + pad) * S0 + (b + pad) * S1;
    for (let ch = 0; ch < channels; ch++) prefilterLine(w, N2, o + ch, channels);
    coef.set(w.subarray(o, o + S1), co);
  }
  return { coef, dims: [N0, N1, N2] };
}
