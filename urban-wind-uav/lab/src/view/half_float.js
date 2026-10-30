// Values as the half-float bits of a THREE.HalfFloatType texel (three.js's own conversion): the scene's slice
// textures, made on the page (wind_layers.js) and in the sweep worker (sweep_worker.js).  The wind model's own
// float16 conversion (shared/numeric.js toHalf) rounds differently: the two are kept separate (the model's stored
// inputs and the slice textures each keep their own).
import { DataUtils } from "../../vendor/three.module.js";

/**
 * Cell `cell` of a slice texture (RGBA half-float bits, `texels` zero elsewhere) given a speed: R the speed, rounded
 * to float32 first (as a Float32 slice), G the mask 1.
 */
export function setSliceTexel(texels, cell, speed) {
  texels[cell * 4] = DataUtils.toHalfFloat(Math.fround(speed));
  texels[cell * 4 + 1] = 0x3c00;            // 1 as half-float bits
}
