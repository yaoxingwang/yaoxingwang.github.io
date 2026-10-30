// The 3-D wind in a mission corridor from the 2-D wind model, layer by layer.
//
// Every layer of the flight band (shared/units.js) is a separate input to the 2-D operator (fno.js).  Every
// layerStride-th layer (and the last one) is predicted, the layers between them are interpolated in height, and the
// result is stacked into one [y][layer][x][u, w, v, weight] grid in the wind frame, which wind/samplers.js
// predictedWind and the page's displays read.  Tiling: tiling.js.  Blending: separable Hann weights per patch, summed
// per output cell and divided once at the end; the sums are kept only where a patch can reach (compactRows: the
// predicted layers, per row the x range its patches cover), as the read-back is.
//
// Inputs (mddf.js).  The stored upper-distance Fourier channels of the whole padded city are computed once when the
// city opens (setCity, city.ready); a new mission or wind direction does not compute them again (phase rotation in
// assemble).  The along-wind channels are computed for the current wind direction at the wind-frame cells of the
// patches, in tiles kept per direction (a new mission in the same wind adds only its new tiles).
import { FIRST_FLIGHT_LAYER, LAST_FLIGHT_LAYER, FLIGHT_LAYERS, CELL_VALUES } from "../shared/units.js";
import { yieldTurn, TURN_AFTER_MS } from "../shared/turn.js";
import { sharedMemory, cpuCores } from "../shared/platform.js";
import { DemoError, ErrorCode } from "../shared/errors.js";
import { Mddf3d } from "./mddf.js";
import { inputPlan } from "./input_plan.js";
import { Fno } from "./fno.js";
import { blendShader, FINALIZE_SHADER } from "./predictor_shaders.js";
import { corridorTiling, defaultCorridor } from "./tiling.js";
import { postRows, compactTableLayout } from "./field_post.js";
import { SharedRing } from "./shared_ring.js";

// Open ground added around the city, in cells (scanned and sampled grid), for patches of P cells.  A kept patch's
// cell-centre square [x, x + P - 1]^2 touches the city square [-0.5, n - 0.5]^2, so each of its cells lies within one
// cell-centre diagonal, (P - 1) sqrt(2) cells, of that square, i.e. at most half a cell more beyond the outermost city
// cell; the bilinear sample reads one more cell.  ceil(P sqrt(2)) covers that.
const expandedMargin = (P) => Math.ceil(P * Math.SQRT2);
// The largest stored field (padded city x predicted layers x stored channels x 2 B) a city may have.  With the shipped
// model (5 stored channels) and 31 predicted layers a 300-cell city needs 68.7 MiB and the 750-cell city 256.8 MiB.
const WHOLE_CITY_BUDGET_BYTES = 1024 << 20;
const BUFFER_LIMIT_BYTES = 128 << 20;  // the largest single GPU buffer (some adapters bind at most 128 MiB)
const RUN_AHEAD = 2;                   // network batches the CPU may submit ahead of the GPU
const FIELD_WORKER_LIMIT_MS = 60000;   // a field worker must answer within this
const MAX_FIELD_WORKERS = 4;           // post-processing threads of a cross-origin isolated page
const MAIN_THREAD_ROWS = 1;            // rows (each with all its layers) per step of the main-thread post-processing

/**
 * The grid cells a batch of [sy, layer, sx] patch origins can reach: { ry0, ry1, rx0, rx1, rz0, rz1 } (half-open,
 * clamped to the gx x gy predicted grid of `layout`; layers from the first flight layer).
 */
function batchReach(batch, layout, P) {
  let ry0 = Infinity, ry1 = -Infinity, rx0 = Infinity, rx1 = -Infinity, rz0 = Infinity, rz1 = -Infinity;
  for (const [sy, layer, sx] of batch) {
    ry0 = Math.min(ry0, sy - layout.gridY0); ry1 = Math.max(ry1, sy - layout.gridY0 + P);
    rx0 = Math.min(rx0, sx - layout.gridX0); rx1 = Math.max(rx1, sx - layout.gridX0 + P);
    rz0 = Math.min(rz0, layer - FIRST_FLIGHT_LAYER); rz1 = Math.max(rz1, layer - FIRST_FLIGHT_LAYER + 1);
  }
  return { ry0: Math.max(0, ry0), ry1: Math.min(layout.gy, ry1), rx0: Math.max(0, rx0), rx1: Math.min(layout.gx, rx1),
           rz0, rz1 };
}

/** The predicted layers: every layerStride-th layer from the first flight layer, and the last one. */
function predictedLayers(layerStride) {
  const list = [];
  for (let l = FIRST_FLIGHT_LAYER; l <= LAST_FLIGHT_LAYER; l++) {
    if ((l - FIRST_FLIGHT_LAYER) % layerStride === 0 || l === LAST_FLIGHT_LAYER) list.push(l);
  }
  return list;
}

/** The wind predictor of one device (see the file header): a city at a time, a corridor per prediction. */
export class Predictor {
  /**
   * The wind model on a device.  options.batchSize: patches per network batch; options.layerStride: every n-th
   * flight layer is predicted (plus the last), the rest interpolated.
   */
  constructor(context, metadata, weights, { batchSize, layerStride }) {
    this.context = context;
    this.metadata = metadata;
    this.patch = metadata.inference.patch[0];
    this.batchSize = batchSize;
    // the planner's wind grid needs evenly spaced predicted layers, the last flight layer one of them
    // (planner/wind_grid.js): a stride that does not divide the band is a broken page setting (app/config.js)
    if ((FLIGHT_LAYERS - 1) % layerStride !== 0) {
      throw new Error(`layer stride ${layerStride} does not divide the ${FLIGHT_LAYERS - 1} layer steps of the band`);
    }
    this.layerStride = layerStride;
    this.bufferLimit = Math.min(BUFFER_LIMIT_BYTES, context.device.limits.maxStorageBufferBindingSize);
    this.fno = new Fno(context, metadata, weights, batchSize);
    this.blendPipeline = context.pipeline(blendShader(this.fno.store));
    this.finalizePipeline = context.pipeline(FINALIZE_SHADER);
    this.inputPlan = inputPlan(metadata);
    this.city = null;
    this.generation = 0;               // a newer prediction or city cancels the running one
    this.fieldWorkers = null;          // post-processing threads (cross-origin isolated pages), made when first needed
    this.fieldJobs = new Map();        // "job id:band" -> { answer, cancel }, while a field worker works on it
    this.fieldJobCount = 0;
    this.valuesRing = new SharedRing();   // the post-processed fields (predict)
  }

  /** Stop the running prediction (it returns null at its next check). */
  cancel() { this.generation += 1; }

  /**
   * Let the predictor go: the running prediction, the city's GPU buffers and the field workers (the network's own
   * buffers go with the device, which is destroyed with it: app/gpu.js).
   */
  destroy() {
    this.cancel();
    this.city?.mddf.destroy();
    this.city = null;
    this.stopFieldWorkers((job) => job.cancel());     // the prediction was cancelled above: it returns null
  }

  /**
   * Open a city (city/city.js: its roof layers, ny, nx): the stored channels of the whole padded city are computed now,
   * for the predicted layers only.
   * Returns the city state: city.ready resolves (to the city) when they are done, city.progress is the share done.
   */
  setCity(city) {
    const { layers: layerMap, ny, nx } = city;
    this.cancel();                     // a prediction on the previous city stops
    this.city?.mddf.destroy();
    const mddf = new Mddf3d(this.context, city, expandedMargin(this.patch), this.fno.store, this.inputPlan);
    const layers = predictedLayers(this.layerStride);
    const storedBytes = mddf.gx * mddf.gy * layers.length * mddf.storedBytesPerCellLayer;
    if (storedBytes > WHOLE_CITY_BUDGET_BYTES) {
      mddf.destroy();
      this.city = null;
      const needed = (storedBytes / 2 ** 20).toFixed(0), budget = WHOLE_CITY_BUDGET_BYTES / 2 ** 20;
      throw new DemoError(ErrorCode.TOO_LARGE,
                          `the city's stored field needs ${needed} MiB, above the ${budget} MiB budget`);
    }
    // layerMap: the city's roof layers; predictedLayers: the layers the network predicts; groups: scanCity
    this.city = { mddf, ny, nx, layerMap, predictedLayers: layers, ready: null, progress: 0, groups: null };
    this.city.ready = this.scanCity(this.city);
    // whoever awaits it gets its failure; a scan of a city given up (another city, a destroyed device) has nobody
    this.city.ready.catch(() => {});
    return this.city;
  }

  /** The whole-city scan of the stored channels (once per city). */
  async scanCity(city) {
    const { mddf } = city;
    const groups = mddf.planGroups(city.predictedLayers, this.bufferLimit);
    mddf.allocate(groups);
    city.progress = 0;
    for (let gi = 0; gi < groups.length; gi++) {
      if (mddf.destroyed) return city;     // another city was opened meanwhile
      await mddf.scan(groups[gi], (f) => { city.progress = (gi + f) / groups.length; });
    }
    city.progress = 1;
    if (mddf.destroyed) return city;
    city.groups = groups;
    return city;
  }

  /**
   * Predict the flight layers (FIRST_FLIGHT_LAYER..LAST_FLIGHT_LAYER) over a corridor (options.polygon in city cells;
   * without it the rectangle along start -> goal).  Returns { maxSpeed2, layout, gx, gy, gz, firstLayer, layerStride,
   * values, valuesPerCell (CELL_VALUES), milliseconds }, or null when cancelled.  values: per cell of [gy][gz][gx],
   * the wind-frame u, w, v in reference units (multiply by speed / reference speed) and the blend weight (0 = not
   * predicted).
   * options: overlap (patch overlap, required), polygon, sweepEvery, onLayer(layer, sums, layout, target) for every
   * sweepEvery-th predicted layer (sums: its [gy][gx][4] weighted sums), onStage("geometry" | "network").
   */
  async predict(start, goal, theta, options) {
    const generation = ++this.generation;
    const cancelled = () => generation !== this.generation;
    const { context } = this;
    const city = this.city;
    const polygon = options.polygon ?? defaultCorridor(start, goal);
    const { overlap } = options;
    const patchStride = Math.max(1, Math.round(this.patch * (1 - overlap)));     // patch origins this far apart
    const layout = corridorTiling(polygon, theta, city.nx, city.ny, this.patch, patchStride);
    const started = performance.now();
    const { mddf } = city;
    await city.ready;
    if (cancelled() || mddf.destroyed) return null;     // another city or prediction meanwhile
    if (mddf.upWind) {
      // the along-wind channels at this prediction's wind-frame cells: the tiles not computed for this direction yet
      const tiles = mddf.ensureUpTiles(city.groups, theta, layout.patches, this.patch,
                                       () => options.onStage?.("geometry"));
      if (tiles) {
        await context.device.queue.onSubmittedWorkDone();
        context.releaseScratch();
      }
    }
    if (cancelled()) return null;
    const compact = compactRows(layout, this.patch, city.predictedLayers.map((l) => l - FIRST_FLIGHT_LAYER));
    const sums = await this.runNetwork(city, layout, compact, theta, options, cancelled);
    if (!sums) return null;
    const result = await this.finish(city, layout, compact, sums, theta, cancelled);
    if (!result) return null;
    return { ...result, milliseconds: performance.now() - started };
  }

  /** Every (layer, patch) pair through the network, blended into the compact weighted sums; null when cancelled. */
  async runNetwork(city, layout, compact, theta, options, cancelled) {
    const { context, patch: P } = this;
    const sums = context.storage(compact.count * CELL_VALUES * 4, "predictor.sums");    // float32
    const tableBuffer = context.upload(compact.table, "predictor.rows");
    const inFlight = [];
    const layers = city.predictedLayers;
    const layerLeft = new Map(layers.map((l) => [l, layout.patches.length]));
    // layers read for the sweep: every options.sweepEvery-th predicted layer and the top one
    const sweepEvery = Math.max(1, options.sweepEvery || 1);
    const sweepSet = new Set(layers.filter((l, k) => k % sweepEvery === 0 || k === layers.length - 1));
    const abort = () => { context.free(sums); context.free(tableBuffer); return null; };
    for (const group of city.groups) {
      if (cancelled()) return abort();
      options.onStage?.("network");
      const work = [];               // the group's (layer, patch) pairs, layer-major
      for (const layer of group.layers) for (const [sy, sx] of layout.patches) work.push([sy, layer, sx]);
      for (let offset = 0; offset < work.length; offset += this.batchSize) {
        const batch = work.slice(offset, offset + this.batchSize);
        const origins = new Int32Array(batch.flat());
        const encoder = context.device.createCommandEncoder();
        const originBuffer = city.mddf.assemble(encoder, origins, group, theta, P, this.fno.inputBuffer);
        const output = this.fno.encode(encoder, batch.length);
        this.encodeBlend(encoder, output, originBuffer, sums, tableBuffer, batch.length, batchReach(batch, layout, P),
                         layout);
        context.device.queue.submit([encoder.finish()]);
        // the CPU leads the GPU by at most RUN_AHEAD batches, so cancellation and the sweep follow the GPU closely
        // without a stall per batch
        inFlight.push(context.device.queue.onSubmittedWorkDone());
        if (inFlight.length > RUN_AHEAD) await inFlight.shift();
        context.free(originBuffer);
        context.releaseScratch();
        // a swept layer whose patches are all in: its weighted sums for the page's sweep, copied out after this batch
        // in queue order and read without waiting, so the prediction does not stall
        if (options.onLayer) {
          for (const [, layer] of batch) {
            layerLeft.set(layer, layerLeft.get(layer) - 1);
            if (layerLeft.get(layer) === 0 && sweepSet.has(layer)) {
              this.readSweptLayer(sums, layer, compact, layout, options, cancelled);
            }
          }
        }
        if (cancelled()) return abort();
      }
    }
    await Promise.all(inFlight);
    context.free(tableBuffer);
    return sums;
  }

  /**
   * The blend of one batch's network outputs into the compact weighted sums (`tableBuffer`: the compact table on the
   * GPU): over the cells it can reach (batchReach) - none when the batch lies outside the predicted grid.
   */
  encodeBlend(encoder, output, originBuffer, sums, tableBuffer, count, reach, layout) {
    const { context, patch: P } = this, { gy } = layout;
    const { ry0, ry1, rx0, rx1, rz0, rz1 } = reach;
    if (!(ry1 > ry0 && rx1 > rx0)) return;
    const pass = encoder.beginComputePass();
    context.dispatch(pass, this.blendPipeline, [output, originBuffer, sums, tableBuffer, context.uniform([
      ["u", count], ["u", P], ["u", this.metadata.model.output_channels], ["u", P],
      ["u", gy], ["u", FIRST_FLIGHT_LAYER], ["u", 0], ["u", 0],
      ["u", ry0], ["u", rz0], ["u", rx0], ["u", ry1 - ry0],
      ["u", rz1 - rz0], ["u", rx1 - rx0], ["i", layout.gridY0], ["i", layout.gridX0],
      ["f", this.metadata.inference.blend_window_minimum], ["f", 0], ["f", 0], ["f", 0],
    ])], (ry1 - ry0) * (rz1 - rz0) * (rx1 - rx0));
    pass.end();
  }

  /** A swept layer whose patches are all in: its weighted sums read (without waiting) and given to options.onLayer. */
  readSweptLayer(sums, layer, compact, layout, options, cancelled) {
    const at = compactTableLayout(layout.gy);
    this.readLayer(sums, compact.table[at.slots + layer - FIRST_FLIGHT_LAYER], compact.table, layout)
      .then((sumsLayer) => {
        if (!cancelled()) options.onLayer(layer, sumsLayer, layout, this.metadata.target);
      })
      .catch(() => {});      // a sweep slice that cannot be read (lost device, freed buffer) is skipped
  }

  /**
   * Copy the predicted layer in `slot` of the compact sums (`table`: compactRows) to a staging buffer, each row's x
   * range in its place, read: [gy][gx][4] (zero where no patch reaches: a new buffer is zero).
   */
  async readLayer(sums, slot, table, { gx, gy }) {
    const at = compactTableLayout(gy), cell = CELL_VALUES * 4;     // bytes per cell (float32)
    const copy = (encoder, to) => {
      for (let y = 0; y < gy; y++) {
        const len = table[at.rowLen + y];
        if (len) {
          encoder.copyBufferToBuffer(sums, (table[y] + slot * len) * cell, to, (y * gx + table[at.rowX0 + y]) * cell,
                                     len * cell);
        }
      }
    };
    return this.context.readBack(gy * gx * cell, copy, "layer staging", (range) => new Float32Array(range.slice(0)));
  }

  /**
   * The sums finished into the field: read back compactly (per row the x range of its patches, the predicted layers
   * only) as float16, then converted, interpolated and its largest speed found (field_post.js) - in worker threads on
   * a cross-origin isolated page (shared memory), otherwise on the main thread in slices.  Both give the same values.
   */
  async finish(city, layout, compact, sums, theta, cancelled) {
    const { context } = this;
    const { gx, gy } = layout;
    const gz = FLIGHT_LAYERS, cells = gx * gy * gz;
    const { predicted } = compact;
    const target = this.metadata.target;
    const packed = context.storage(Math.max(8, compact.count * CELL_VALUES * 2), "predictor.packed");   // float16
    const encoder = context.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    context.dispatch(pass, this.finalizePipeline, [sums, packed, context.uniform([
      ["u", compact.count], ["f", target.center[0]], ["f", target.center[1]], ["f", target.center[2]],
      ["f", target.scale[0]], ["f", target.scale[1]], ["f", target.scale[2]], ["f", 0],
    ])], compact.count);
    pass.end();
    context.device.queue.submit([encoder.finish()]);
    await context.device.queue.onSubmittedWorkDone();
    context.free(sums);
    const shared = sharedMemory();
    const half = await context.readHalf(packed, compact.count * CELL_VALUES, shared);
    context.free(packed);
    context.releaseScratch();
    if (cancelled()) return null;
    let values;
    if (shared) {
      // two shared buffers used in turn: the current prediction is in one, the next is written into the other, so
      // the one before (possibly still read by a superseded computation) is not touched; postRows writes every value
      // of every row, so nothing of an earlier field is left in the view
      values = new Float32Array(this.valuesRing.next(cells * CELL_VALUES * 4), 0, cells * CELL_VALUES);
    } else {
      values = new Float32Array(cells * CELL_VALUES);
    }
    const p = { half, values, gy, gz, gx, gridX0: layout.gridX0, gridY0: layout.gridY0, theta, nx: city.nx, ny: city.ny,
                layerMap: city.layerMap, predicted, firstLayer: FIRST_FLIGHT_LAYER, interpolate: this.layerStride > 1,
                compactTable: compact.table };
    const parts = shared ? await this.postInWorkers(p, gy) : await postOnMainThread(p, gy);
    if (cancelled()) return null;
    let maxSpeed2 = 0;
    for (const r of parts) if (r.maxSpeed2 > maxSpeed2) maxSpeed2 = r.maxSpeed2;
    return { maxSpeed2, layout, gx, gy, gz, firstLayer: FIRST_FLIGHT_LAYER, layerStride: this.layerStride, values,
             valuesPerCell: CELL_VALUES };
  }

  /** The field workers end; every waiting job is ended by endJob(job); the next prediction makes new workers. */
  stopFieldWorkers(endJob) {
    for (const worker of this.fieldWorkers ?? []) worker.terminate();
    this.fieldWorkers = null;
    for (const job of [...this.fieldJobs.values()]) endJob(job);
  }

  /** The field workers failed or stopped answering: every waiting job fails with `message`. */
  failFieldWorkers(message) {
    this.stopFieldWorkers((job) => job.answer({ error: message }));
  }

  /**
   * The field workers: a third of the logical cores, 1 to MAX_FIELD_WORKERS.  Made in a method of their own: their
   * handlers live as long as they do, so they must not close over a job's data (its fields would stay in memory).
   */
  startFieldWorkers() {
    const count = Math.max(1, Math.min(MAX_FIELD_WORKERS, Math.floor(cpuCores() / 3)));
    return Array.from({ length: count }, (_, band) => {
      const worker = new Worker(new URL("./field_worker.js", import.meta.url), { type: "module" });
      // each reply settles its own job, also an older job's reply while a newer job runs
      worker.onmessage = ({ data }) => this.fieldJobs.get(`${data.id}:${band}`)?.answer(data);
      // a worker that failed (it did not start, or ran out of memory): every waiting job fails now, and the next
      // prediction makes new workers
      worker.onerror = (event) => {
        event.preventDefault();
        this.failFieldWorkers(`field worker failed: ${event.message || "it did not load"}`);
      };
      return worker;
    });
  }

  /** postRows in worker threads (field_worker.js), the rows split in bands. */
  postInWorkers(p, gy) {
    this.fieldWorkers ??= this.startFieldWorkers();
    const id = ++this.fieldJobCount;
    const workers = this.fieldWorkers, K = workers.length;
    // the city's layer map goes with every job (shared memory on such a page, city.js loadCity), so an idle worker
    // holds no city
    const message = { ...p, half: p.half.buffer, halfLength: p.half.length, values: p.values.buffer,
                      valuesLength: p.values.length };
    return Promise.all(workers.map((worker, b) => new Promise((resolve, reject) => {
      const key = `${id}:${b}`;
      const end = () => { clearTimeout(timer); this.fieldJobs.delete(key); };
      const timer = setTimeout(() => {
        this.failFieldWorkers(`field worker did not answer in ${FIELD_WORKER_LIMIT_MS / 1000} s`);
      }, FIELD_WORKER_LIMIT_MS);
      this.fieldJobs.set(key, {
        answer: (data) => {
          end();
          if (data.error) reject(new Error(data.error));
          else resolve(data);
        },
        cancel: () => { end(); resolve(null); },
      });
      worker.postMessage({ id, p: message, yLo: Math.floor(b * gy / K), yHi: Math.floor((b + 1) * gy / K) });
    })));
  }
}

/**
 * The compact layout of the sums and the read-back: the predicted layers (offsets from the first flight layer,
 * ascending) and per row y the x range its patches cover; compact cell g: row y with rowOff[y] <= g < rowOff[y + 1],
 * then layer slot and x.  table (Uint32Array): the sections of compactTableLayout, the slots 0xffffffff for a flight
 * layer not predicted.  Returns { table, count (cells), predicted }.
 */
function compactRows(layout, patch, predicted) {
  const { gx, gy } = layout;
  const rowX0 = new Int32Array(gy).fill(gx), rowX1 = new Int32Array(gy).fill(0);
  for (const [sy, sx] of layout.patches) {
    const y0 = Math.max(0, sy - layout.gridY0), y1 = Math.min(gy, sy - layout.gridY0 + patch);
    const x0 = Math.max(0, sx - layout.gridX0), x1 = Math.min(gx, sx - layout.gridX0 + patch);
    for (let y = y0; y < y1; y++) {
      if (x0 < rowX0[y]) rowX0[y] = x0;
      if (x1 > rowX1[y]) rowX1[y] = x1;
    }
  }
  const at = compactTableLayout(gy);
  const table = new Uint32Array(at.slots + FLIGHT_LAYERS).fill(0xffffffff, at.slots);
  let off = 0;
  for (let y = 0; y < gy; y++) {
    const len = Math.max(0, rowX1[y] - rowX0[y]);
    table[y] = off;
    table[at.rowX0 + y] = len ? rowX0[y] : 0;
    table[at.rowLen + y] = len;
    off += len * predicted.length;
  }
  table[gy] = off;
  predicted.forEach((z, slot) => { table[at.slots + z] = slot; });
  return { table, count: off, predicted };
}

/** postRows on the main thread in bands of rows, with a turn for the page every TURN_AFTER_MS. */
async function postOnMainThread(p, gy) {
  const parts = [];
  let slice = performance.now();
  for (let y = 0; y < gy; y += MAIN_THREAD_ROWS) {
    parts.push(postRows(p, y, Math.min(gy, y + MAIN_THREAD_ROWS)));
    if (performance.now() - slice > TURN_AFTER_MS) {
      await yieldTurn();
      slice = performance.now();
    }
  }
  return parts;
}
