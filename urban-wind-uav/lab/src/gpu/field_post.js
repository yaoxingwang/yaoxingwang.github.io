// The predicted field after its read-back: float16 to float32, the layers between predicted ones, and the largest
// squared speed, for a band of rows of the [gy][gz][gx][4] field.  Every row is independent, so worker threads
// (field_worker.js) or the main thread in slices each take a band and the result is the same as one pass over all
// rows.
import { CELL_VALUES } from "../shared/units.js";
import { halfToFloatTable, roundHalfEven } from "../shared/numeric.js";

/**
 * A cell is predicted where its blend weight is above this (also in predictor_shaders.js FINALIZE_SHADER, written
 * from MIN_BLEND_WEIGHT_WGSL, and in the sweep worker's own finishing of the sums).
 */
export const MIN_BLEND_WEIGHT_WGSL = "1.0e-6";
export const MIN_BLEND_WEIGHT = Number(MIN_BLEND_WEIGHT_WGSL);
/**
 * The weight stored with a predicted cell is at least this (FINALIZE_SHADER), so that in the packed float16 field
 * every predicted cell has a weight clearly above 0 (readers of the packed field test weight > 0: wind/samplers.js,
 * planner/wind_grid.js) and every other cell 0.
 */
export const STORED_WEIGHT_FLOOR_WGSL = "1.0e-5";

/**
 * Where the compact table (predictor.js compactRows: the sums' and the read-back's layout) of a gy-row grid keeps its
 * sections: [rowOff, gy + 1] from 0, [rowX0, gy], [rowLen, gy], then each flight layer offset's slot.
 */
export const compactTableLayout = (gy) => ({ rowX0: gy + 1, rowLen: 2 * gy + 1, slots: 3 * gy + 1 });

/**
 * Rows [yLo, yHi) of the field.  p: { half (Uint16Array, the compact field), values (Float32Array, out), gy, gz, gx,
 * gridX0, gridY0, theta, nx, ny, layerMap, predicted (the predicted layers' offsets, ascending), firstLayer,
 * interpolate (bool), compactTable (compactTableLayout) }.  Returns { maxSpeed2 } of these rows.
 */
export function postRows(p, yLo, yHi) {
  const { half, values, gy, gz, gx, gridX0, gridY0, theta, nx, ny, layerMap, predicted, firstLayer, compactTable } = p;
  const at = compactTableLayout(gy);
  const t = halfToFloatTable();
  const row = gz * gx * CELL_VALUES;
  // the compact read-back: zeros, then the predicted layers' x range of each row (everywhere else the field is zero)
  const xLo = new Int32Array(yHi - yLo), xHi = new Int32Array(yHi - yLo);      // per row of the band
  const nd = predicted.length;
  for (let y = yLo; y < yHi; y++) {
    values.fill(0, y * row, (y + 1) * row);
    const len = compactTable[at.rowLen + y], x0 = compactTable[at.rowX0 + y], off = compactTable[y];
    xLo[y - yLo] = x0; xHi[y - yLo] = x0 + len;
    for (let zi = 0; zi < nd; zi++) {
      const src = (off + zi * len) * CELL_VALUES, dst = ((y * gz + predicted[zi]) * gx + x0) * CELL_VALUES;
      for (let i = 0; i < CELL_VALUES * len; i++) values[dst + i] = t[half[src + i]];
    }
  }
  if (p.interpolate) {
    // Layers between predicted ones: linear interpolation in height.  Exception: where the lower predicted layer is
    // inside a building and the interpolated layer is not, the upper (fluid) layer is copied: the model's output inside
    // buildings was never trained (the training loss covers fluid cells only), so it must not be mixed in.  Solid is
    // decided as in the solid input channel: the roof layer of the nearest city cell (round half to even, as WGSL
    // round) greater than the layer index.
    const cY = (ny - 1) / 2, cX = (nx - 1) / 2, cos = Math.cos(theta), sin = Math.sin(theta);
    const roof = new Float32Array(gx);
    for (let y = yLo; y < yHi; y++) {
      for (let x = 0; x < gx; x++) {
        const dx = gridX0 + x - cX, dy = gridY0 + y - cY;
        const ix = cX + dx * cos - dy * sin, iy = cY + dx * sin + dy * cos;     // windFrame toCity
        const hx = roundHalfEven(ix), hy = roundHalfEven(iy);
        roof[x] = hx >= 0 && hx < nx && hy >= 0 && hy < ny ? layerMap[hy * nx + hx] : 0;
      }
      const xa = xLo[y - yLo], xb = xHi[y - yLo];     // zeros elsewhere stay zeros
      for (let d = 0; d < predicted.length - 1; d++) {
        const z0 = predicted[d], z1 = predicted[d + 1];
        for (let z = z0 + 1; z < z1; z++) {
          const f = (z - z0) / (z1 - z0);
          const layer0 = firstLayer + z0, layer = firstLayer + z;
          const r0 = (y * gz + z0) * gx * CELL_VALUES, r1 = (y * gz + z1) * gx * CELL_VALUES;
          const r = (y * gz + z) * gx * CELL_VALUES;
          for (let x = xa; x < xb; x++) {
            const h = roof[x];
            const lowerSolid = h > layer0, here = h > layer;
            for (let c = 0; c < CELL_VALUES; c++) {
              const i = CELL_VALUES * x + c;
              values[r + i] = lowerSolid && !here ? values[r1 + i]
                : values[r0 + i] + f * (values[r1 + i] - values[r0 + i]);
            }
          }
        }
      }
    }
  }
  let maxSpeed2 = 0;
  for (let j = yLo * row; j < yHi * row; j += CELL_VALUES) {
    if (values[j + 3] > MIN_BLEND_WEIGHT) {
      const q = values[j] * values[j] + values[j + 1] * values[j + 1] + values[j + 2] * values[j + 2];
      if (q > maxSpeed2) maxSpeed2 = q;
    }
  }
  return { maxSpeed2 };
}
