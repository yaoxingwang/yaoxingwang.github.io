// The CFD ground-truth wind files (z2 q8): uint8 [y][layer][x][3] (stored u, w, v), then float32 lo[layer][3] and
// hi[layer][3]; value = lo + q (hi - lo) / 255, for the layers the manifest's cfd_encoding names.  Downloaded and
// decoded in a worker (cfd_worker.js), so neither takes time from the page's main thread while a mission is computed.
import { DemoError, ErrorCode } from "../shared/errors.js";
import { sharedMemory } from "../shared/platform.js";

/**
 * A Float32Array in shared memory when the page is cross-origin isolated (serve.py sends the headers), so the
 * wind fields reach the workers without a copy; an ordinary array otherwise (postMessage then copies it).
 */
function sharedFloat32(count) {
  return sharedMemory() ? new Float32Array(new SharedArrayBuffer(count * 4)) : new Float32Array(count);
}

/** Decode a z2 q8 file of `shape` [ny, nz, nx]; `onRows(share)` reports the decoded share after every row y. */
export function decodeCfd(buffer, shape, onRows) {
  const [ny, nz, nx] = shape;
  const count = ny * nz * nx * 3;
  const q = new Uint8Array(buffer, 0, count);
  const lo = new Float32Array(buffer.slice(count, count + nz * 3 * 4));
  const hi = new Float32Array(buffer.slice(count + nz * 3 * 4, count + 2 * nz * 3 * 4));
  const values = sharedFloat32(count);
  const step = new Float32Array(nz * 3);
  for (let i = 0; i < nz * 3; i++) step[i] = (hi[i] - lo[i]) / 255;
  for (let y = 0; y < ny; y++) {
    for (let k = 0; k < nz; k++) {
      const row = (y * nz + k) * nx * 3;
      for (let x = 0; x < nx; x++) for (let c = 0; c < 3; c++) {
        const i = row + x * 3 + c;
        values[i] = lo[k * 3 + c] + q[i] * step[k * 3 + c];
      }
    }
    onRows((y + 1) / ny);
  }
  return values;
}

/** The file size of a CFD entry in bytes (its download progress is measured against it). */
export const cfdBytes = ({ shape: [ny, nz, nx] }) => ny * nz * nx * 3 + 2 * nz * 3 * 4;

/** The error of a CFD download cancelled by another city (DemoError CANCELLED). */
export const downloadCancelled = (file) => new DemoError(ErrorCode.CANCELLED, `${file}: download cancelled`);

/**
 * Download and decode the CFD file of manifest entry `entry` in a worker.  Resolves to the grid
 * { values, ny, nz, nx, layerStride, firstLayer }.  `signal` cancels it (DemoError CANCELLED); `onBytes(n)` and
 * `onDecode(share)` report progress.
 */
export function loadCfd(entry, encoding, { signal, onBytes, onDecode }) {
  return new Promise((resolve, reject) => {
    const worker = new Worker(new URL("./cfd_worker.js", import.meta.url), { type: "module" });
    const cancel = () => {
      worker.terminate();
      reject(downloadCancelled(entry.file));
    };
    if (signal.aborted) { cancel(); return; }
    signal.addEventListener("abort", cancel, { once: true });
    const end = () => { signal.removeEventListener("abort", cancel); worker.terminate(); };
    worker.onmessage = ({ data: m }) => {
      if (m.bytes !== undefined) { onBytes(m.bytes); return; }
      if (m.decoded !== undefined) { onDecode(m.decoded); return; }
      end();
      if ("error" in m) {
        reject(new DemoError(m.code || ErrorCode.DOWNLOAD, m.error || `${entry.file} did not load`));
        return;
      }
      const [ny, nz, nx] = entry.shape;
      resolve({ values: m.values, ny, nz, nx, layerStride: encoding.layer_stride, firstLayer: encoding.first_layer });
    };
    worker.onerror = (event) => {
      end();
      reject(new DemoError(ErrorCode.DOWNLOAD, `${entry.file}: ${event.message || "the loading thread failed"}`));
    };
    worker.postMessage({ url: new URL(`assets/cfd/${entry.file}`, document.baseURI).href, shape: entry.shape });
  });
}
