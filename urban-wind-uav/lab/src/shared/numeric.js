// Small numeric helpers shared by the page and the workers: rounding as numpy, float16 conversion, cubic B-spline
// weights.

/** Round half to even, as numpy.round and Python round (and WGSL round). */
export function roundHalfEven(x) {
  const r = Math.round(x);
  return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r;
}

let halfTable = null;
/** The float32 value of every float16 bit pattern (65536 entries, built once). */
export function halfToFloatTable() {
  if (halfTable) return halfTable;
  halfTable = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h >> 10) & 0x1f;
    const mantissa = h & 0x3ff;
    if (exponent === 0) halfTable[h] = sign * mantissa * 2 ** -24;
    else if (exponent === 31) halfTable[h] = mantissa ? NaN : sign * Infinity;
    else halfTable[h] = sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }
  return halfTable;
}

/** float16 bit patterns -> Float32Array. */
export function fromHalf(half) {
  const out = new Float32Array(half.length);
  const table = halfToFloatTable();
  for (let i = 0; i < half.length; i++) out[i] = table[half[i]];
  return out;
}

/** Numbers -> float16 bit patterns (rounded to nearest, ties away from zero). */
export function toHalf(values) {
  const out = new Uint16Array(values.length);
  const scratch = new Float32Array(1);
  const bits = new Uint32Array(scratch.buffer);
  for (let i = 0; i < values.length; i++) {
    scratch[0] = values[i];
    const x = bits[0];
    const sign = (x >>> 16) & 0x8000;
    const exponent = (x >>> 23) & 0xff;
    let mantissa = x & 0x7fffff;
    if (exponent === 0xff) { out[i] = sign | 0x7c00 | (mantissa ? 1 : 0); continue; }
    const e = exponent - 127 + 15;
    if (e >= 31) { out[i] = sign | 0x7c00; continue; }
    if (e <= 0) {
      if (e < -10) { out[i] = sign; continue; }
      mantissa |= 0x800000;
      const shift = 14 - e;
      let half = mantissa >>> shift;
      if ((mantissa >>> (shift - 1)) & 1) half += 1;
      out[i] = sign | half;
      continue;
    }
    let half = (e << 10) | (mantissa >>> 13);
    if (mantissa & 0x1000) half += 1;
    out[i] = sign | half;
  }
  return out;
}

/**
 * Uniform cubic B-spline weights of the four taps around a point at fraction t of its interval: values into w[4],
 * derivatives (per interval) into d[4].
 */
export function bsplineWeights(t, w, d) {
  const t2 = t * t, t3 = t2 * t, s = 1 - t;
  w[0] = s * s * s / 6; w[1] = (3 * t3 - 6 * t2 + 4) / 6; w[2] = (-3 * t3 + 3 * t2 + 3 * t + 1) / 6; w[3] = t3 / 6;
  d[0] = -0.5 * s * s; d[1] = 1.5 * t2 - 2 * t; d[2] = -1.5 * t2 + t + 0.5; d[3] = 0.5 * t2;
}
