// A city's height map as a preview image (app/picker.js): the arithmetic only, so the height-map worker draws it.

/**
 * The RGBA pixels (Uint8ClampedArray) of height map `map` ({ nx, ny, layers }) on W x H pixels (image row 0 = city y
 * max, as the dial's 90 degrees): per pixel the tallest cell under it (so small previews keep their buildings),
 * coloured along `ramp` ([[r, g, b], ...]) up to topLayers, `ground` ([r, g, b]) where there is no building.
 */
export function heightMapPixels(map, W, H, topLayers, ramp, ground) {
  const d = new Uint8ClampedArray(W * H * 4);
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const x0 = Math.floor(px / W * map.nx), x1 = Math.max(x0 + 1, Math.floor((px + 1) / W * map.nx));
    const y1 = map.ny - Math.floor(py / H * map.ny), y0 = Math.min(y1 - 1, map.ny - Math.floor((py + 1) / H * map.ny));
    let v = 0;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) v = Math.max(v, map.layers[y * map.nx + x]);
    const o = (py * W + px) * 4;
    if (v > 0) {
      // the lowest building is drawn at 25 % of the ramp so it stays visible
      const x = (0.25 + 0.75 * Math.sqrt(Math.min(1, v / topLayers))) * (ramp.length - 1);
      const i = Math.min(ramp.length - 2, Math.floor(x)), f = x - i;
      const lo = ramp[i], hi = ramp[i + 1];
      for (let ch = 0; ch < 3; ch++) d[o + ch] = Math.round(lo[ch] + f * (hi[ch] - lo[ch]));
    } else {
      d.set(ground, o);
    }
    d[o + 3] = 255;
  }
  return d;
}
