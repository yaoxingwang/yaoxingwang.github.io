// The wind model's input manifest (metadata.inputs): which geometry inputs the model reads, checked against what the
// page computes (gpu/mddf.js).
import { modelMisfit } from "../shared/errors.js";

/**
 * What the manifest needs: stored Fourier coefficients (each with the scale of the first input that reads it, which
 * sets its packing step: mddf.js) and along-wind channels.
 */
export function inputPlan(metadata) {
  const inputs = metadata.inputs;
  if (!inputs) throw modelMisfit("the model metadata has no input manifest (metadata.inputs)");
  if (inputs.length !== metadata.model.input_channels) {
    throw modelMisfit(`manifest has ${inputs.length} inputs, model expects ${metadata.model.input_channels}`);
  }
  const stored = [];
  const storedIndex = (signal, part, mode) => stored.findIndex((s) => s.key === `${signal}:${part}:${mode}`);
  // a stored channel the input e reads, added if it is new
  const need = (e, part, mode) => {
    if (storedIndex(e.signal, part, mode) >= 0) return;
    stored.push({ key: `${e.signal}:${part}:${mode}`, signal: e.signal, part, mode, scale: e.scale });
  };
  const up = [];
  for (const e of inputs) {
    if (e.source === "r1") {
      if (e.part === "dc") {
        need(e, "dc", 0);
      } else {
        need(e, "re", e.mode);
        need(e, "im", e.mode);
      }
    } else if (e.source === "up") {
      if (!up.includes(e.index)) up.push(e.index);
    } else if (!["geo", "solid", "coord"].includes(e.source)) {
      throw modelMisfit(`unsupported input source ${e.source}`);
    }
    if (e.source === "geo" && ![0, 1].includes(e.index)) throw modelMisfit(`unsupported geo channel ${e.index}`);
    if (e.source === "up" && !(e.index >= 0 && e.index < 4)) throw modelMisfit(`unsupported up channel ${e.index}`);
  }
  for (const s of stored) if (s.mode > 2) throw modelMisfit("Fourier modes above 2 are not computed");
  const signals = [...new Set(stored.map((s) => s.signal))].sort();
  if (signals.length !== 1 || signals[0] !== 0) {
    throw modelMisfit("the stored channels must be those of the upper distance (signal 0) only");
  }
  // the assemble pass reads the roof heights only for these inputs
  const usesHeight = inputs.some((e) => e.source === "solid" || e.source === "geo");
  return { inputs, stored, storedIndex, up, usesHeight };
}
