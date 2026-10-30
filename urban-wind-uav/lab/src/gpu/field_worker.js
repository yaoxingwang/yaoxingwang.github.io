// A band of rows of the predicted field after the prediction (field_post.js postRows), off the main thread.
import { errorText } from "../shared/errors.js";
import { postRows } from "./field_post.js";

// Messages: { id, p (postRows' input), yLo, yHi } -> { id, maxSpeed2 } or { id, error }.
self.onmessage = ({ data: { id, p, yLo, yHi } }) => {
  try {
    // the shared buffers may be larger than this field (they are reused): views of exactly its length
    const half = new Uint16Array(p.half, 0, p.halfLength), values = new Float32Array(p.values, 0, p.valuesLength);
    const r = postRows({ ...p, half, values }, yLo, yHi);
    self.postMessage({ id, ...r });
  } catch (error) {
    self.postMessage({ id, error: errorText(error) });
  }
};
