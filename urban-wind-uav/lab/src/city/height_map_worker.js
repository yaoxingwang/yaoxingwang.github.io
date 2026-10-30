// A city's height map work off the page's main thread (city.js): { url } -> { map: { layers, nx, ny, maximumLayers } }
// (downloaded and decoded), { buildings: { nx, ny, layers } } -> { surfaces: { position, normal, sphere } } (the
// buildings' surfaces), { previews: { maps, sizes, topLayers, ramp, ground } } -> { pixels } (preview.js, per map);
// the arrays transferred (the layers are shared memory on a cross-origin isolated page).  A failure answers
// { error, code }.
import { download } from "../shared/download.js";
import { sharedMemory } from "../shared/platform.js";
import { buildingSurfaces } from "./buildings.js";
import { heightMapPixels } from "./preview.js";

/** The height map in PNG `bytes` (Uint8Array): { layers (Float32Array, row-major), nx, ny, maximumLayers }. */
async function decodeHeightMap(bytes) {
  const bitmap = await createImageBitmap(new Blob([bytes], { type: "image/png" }));
  const nx = bitmap.width, ny = bitmap.height;
  const context = new OffscreenCanvas(nx, ny).getContext("2d", { willReadFrequently: true });
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, nx, ny);
  bitmap.close();
  const layers = new Float32Array(sharedMemory() ? new SharedArrayBuffer(nx * ny * 4) : nx * ny);
  let maximumLayers = 0;
  for (let i = 0; i < nx * ny; i++) {
    layers[i] = data[i * 4];
    maximumLayers = Math.max(maximumLayers, data[i * 4]);
  }
  return { layers, nx, ny, maximumLayers };
}

self.onmessage = async ({ data: { url, buildings, previews } }) => {
  try {
    if (previews) {
      const { maps, sizes, topLayers, ramp, ground } = previews;
      const pixels = maps.map((map, k) => map && heightMapPixels(map, ...sizes[k], topLayers, ramp, ground));
      self.postMessage({ pixels }, pixels.filter(Boolean).map((p) => p.buffer));
    } else if (buildings) {
      const surfaces = buildingSurfaces(buildings);
      self.postMessage({ surfaces }, [surfaces.position.buffer, surfaces.normal.buffer]);
    } else {
      const map = await decodeHeightMap(await download(url));
      self.postMessage({ map }, map.layers.buffer instanceof ArrayBuffer ? [map.layers.buffer] : []);
    }
  } catch (error) {
    // only the page's own error codes (shared/errors.js) are passed on, not a browser error's numeric code
    self.postMessage({ error: error.message, code: typeof error.code === "string" ? error.code : undefined });
  }
};
