// Urban blocks: the manifest of the shipped cities and their height maps (PNG-8 grayscale, one pixel per 4 m cell,
// value = roof height in 1.5 m layers).
import { downloadJson } from "../shared/download.js";
import { DemoError } from "../shared/errors.js";

/** The city manifest (assets/cities/manifest.json). */
export const loadManifest = () => downloadJson("assets/cities/manifest.json");

// city id -> the promise of its height map (the picker's previews and the opened city share it)
const loaded = new Map();

/**
 * The height map of manifest entry `entry`: { id, layers (Float32Array, row-major), nx, ny, maximumLayers }, loaded
 * once (a failed load is tried again next time).  Nothing writes into it after it is made, so on a cross-origin
 * isolated page its layers are shared memory: the workers that get the city share them instead of a copy each.
 */
export function loadCity(entry) {
  if (!loaded.has(entry.id)) {
    loaded.set(entry.id, decodeCity(entry).catch((error) => {
      loaded.delete(entry.id);
      throw error;
    }));
  }
  return loaded.get(entry.id);
}

// height-map workers running at once (the picker asks for every city's preview together)
const WORKERS_AT_ONCE = 2;
let running = 0;
const queued = [];                     // requests waiting for a worker: their start functions

/**
 * One request to a worker of its own (height_map_worker.js, at most WORKERS_AT_ONCE at a time), ended when it answers:
 * resolves to its answer's `key`; rejects with the worker's error (a download's DemoError with its own code) or, when
 * the worker itself fails, an Error naming `what` (the file or the city).
 */
function inWorker(message, key, what) {
  return new Promise((resolve, reject) => {
    const start = () => {
      running += 1;
      const worker = new Worker(new URL("./height_map_worker.js", import.meta.url), { type: "module" });
      const end = () => {
        worker.terminate();
        running -= 1;
        queued.shift()?.();
      };
      worker.onmessage = ({ data: m }) => {
        end();
        if (!("error" in m)) resolve(m[key]);
        else reject(m.code ? new DemoError(m.code, m.error) : new Error(m.error || `${what} did not load`));
      };
      worker.onerror = (event) => {
        event.preventDefault();          // reported here, as the rejection
        end();
        reject(new Error(`${what}: ${event.message || "the loading thread failed"}`));
      };
      worker.postMessage(message);
    };
    if (running < WORKERS_AT_ONCE) start();
    else queued.push(start);
  });
}

/** Download and decode a city's PNG in a worker, so a large city does not hold up the page. */
async function decodeCity(entry) {
  const url = new URL(`assets/cities/${entry.file}`, document.baseURI).href;
  return { id: entry.id, ...await inWorker({ url }, "map", entry.file) };
}

/**
 * Preview images of height maps `maps` (city maps or null) in a worker (preview.js heightMapPixels): sizes [[W, H]],
 * topLayers, ramp and ground as there.  Resolves to the RGBA pixels per map (null for a null map).
 */
export function cityPreviews(maps, sizes, topLayers, ramp, ground) {
  const plain = maps.map((m) => m && { nx: m.nx, ny: m.ny, layers: m.layers });
  return inWorker({ previews: { maps: plain, sizes, topLayers, ramp, ground } }, "pixels", "the city previews");
}

/** The buildings' surfaces of `city` (city/buildings.js buildingSurfaces), built in a worker. */
export function cityBuildings(city) {
  const { nx, ny, layers } = city;
  return inWorker({ buildings: { nx, ny, layers } }, "surfaces", `city ${city.id}`);
}
