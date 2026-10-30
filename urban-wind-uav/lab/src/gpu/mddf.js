// The geometry inputs of the wind model (multi-directional distance features, MDDF) for a stack of heights, built
// from the model's input manifest (metadata.inputs), and the assemble pass that samples them into model input patches
// for any wind direction.  The shaders are in mddf_shaders.js.
//
// Stored channels ("r1" inputs, signal 0).  The upper distance of a cell at query height z in azimuth a is the
// distance along that direction to the first roof at or above z: 0 inside such a roof, the 1600 m cutoff when there
// is none.  Over 180 azimuths its Fourier coefficients (DC, modes 1-2) are the stored channels.  They are wind
// independent: a direction change rotates C_m by exp(i m theta) inside assemble.  The whole padded city is computed
// once (scan), for an explicit list of layers (slot k of a group holds layer group.layers[k]), and kept compressed
// (packed field).
//
// Along-wind channels ("up" inputs 0-3).  Recomputed for the current wind direction along the UPWIND ray (azimuth
// theta + 180 deg in the city grid), cell by cell through the padded city up to the 1600 m cutoff:
//   up 0, 1  the distance at which the ray enters the first roof at or above the query height z, and that roof's
//            height minus z (cutoff and 0 when there is none);
//   up 2, 3  where no such roof is reached: the distance at which the ray leaves the roof below z with the largest
//            slope (roof - z) / distance, and that roof's height minus z (cutoff and -z when there is none);
//            otherwise up 0 and 0.
// The network reads every input at the wind-frame cells of its patches (the city rotated by theta about its
// centre, integer cells).  The stored channels are sampled on the city grid; the along-wind channels are computed
// AT the wind-frame cells: the upwind ray starts at the cell's own position (in general not a city cell centre), and
// assemble reads them without interpolation.  Storage: UP_TILE x UP_TILE wind-frame tiles, computed when a
// prediction first needs them and kept for that direction (a new mission in the same wind computes only the tiles it
// adds).  At theta = 0 (and multiples of 90 degrees) the wind-frame cells are the city cells.  Stored as float16.
//
// Geo inputs: hloc = h / 100 and hrel = asinh((h - z) / 10) from the roof height of the cell the solid mask uses,
// rounded through float16 as the training files were.

import { LAYER_M, CELL_M } from "../shared/units.js";
import { DemoError, ErrorCode } from "../shared/errors.js";
import { windFrame } from "../shared/geometry.js";
import { ANGLE_COUNT, LAYER_GROUP, UP_TILE, PACK_TILE, scanShader, fourierAccumulateShader, packShader, packTileShader,
         alongWindShader, assembleShader, MDDF_WORKGROUP, PACK_BLOCKS } from "./mddf_shaders.js";

const SCAN_LAYERS = 4;                 // heights per scan sub-group (float32 sums of every stored channel per cell)
const SCAN_RAW_BYTES = 48 << 20;       // raw distances of one batch of azimuths
const SCAN_IN_FLIGHT = 3;              // scan batches the CPU may submit ahead of the GPU
const PACK_TOLERANCE = 0.016;          // maximum error of the packed field, share of the channel's input scale
const SLOT_TABLE = 256;                // slot of every layer index (-1: not in the group)

/** The geometry inputs of one city on the GPU (see the file header): its stored channels and the assemble pass. */
export class Mddf3d {
  /**
   * @param {import("./context.js").GpuContext} context
   * @param {object} city city/city.js: layers (the roof layer count per city cell [ny, nx]), ny rows, nx columns
   * @param {number} margin open ground added on every side, in cells
   * @param {string} store the stored channels' number format ("f16" or "f32", fno.js store)
   * @param {object} inputPlan the model's input plan (input_plan.js inputPlan)
   */
  constructor(context, { layers: layerMap, ny, nx }, margin, store, inputPlan) {
    this.context = context;
    this.inputPlan = inputPlan;
    this.ny = ny;
    this.nx = nx;
    this.margin = margin;
    this.gy = ny + 2 * margin;
    this.gx = nx + 2 * margin;
    const heights = new Float32Array(this.gy * this.gx);
    for (let y = 0; y < ny; y++) {
      for (let x = 0; x < nx; x++) {
        heights[(y + margin) * this.gx + x + margin] = layerMap[y * nx + x] * LAYER_M;
      }
    }
    this.heightBuffer = context.upload(heights, "mddf.height");
    const cosT = new Float32Array(ANGLE_COUNT);
    const sinT = new Float32Array(ANGLE_COUNT);
    for (let a = 0; a < ANGLE_COUNT; a++) {
      cosT[a] = Math.cos(2 * Math.PI * a / ANGLE_COUNT);
      sinT[a] = Math.sin(2 * Math.PI * a / ANGLE_COUNT);
    }
    this.cosT = cosT;
    this.sinT = sinT;
    this.scanPipeline = context.pipeline(scanShader());
    this.accumulatePipeline = context.pipeline(fourierAccumulateShader(inputPlan.stored));
    // along-wind channels on the wind-frame grid (header), computed by ensureUpTiles
    this.upWind = inputPlan.up.length > 0;
    this.upGrid = null;
    this.upTableBuffer = null;
    if (this.upWind) this.alongPipeline = context.pipeline(alongWindShader(inputPlan.up));
    // PACKED FIELD of the whole-city scan
    const steps = inputPlan.stored.map((st) => 2 * PACK_TOLERANCE * st.scale);
    const tx = Math.ceil(this.gx / PACK_TILE), ty = Math.ceil(this.gy / PACK_TILE);
    this.packed = { steps, tx, ty };
    this.packHeaderPipeline = context.pipeline(packShader(inputPlan.stored, steps, tx, ty, false));
    this.packPayloadPipeline = context.pipeline(packShader(inputPlan.stored, steps, tx, ty, true));
    this.packTilePipeline = context.pipeline(packTileShader());
    this.assemblePipeline = context.pipeline(assembleShader(store, inputPlan, this.packed));
    this.groups = null;                // the stored channels' groups (planGroups, allocate)
    // the scan's tables (made on the first scan) and its transient buffers (while a city is scanned)
    this.offsetBuffer = null; this.tableBuffer = null; this.trigBuffer = null; this.scanInfo = null;
    this.scanRaw = null; this.sumsBuffer = null;
    this.destroyed = false;            // destroy() was called: a scan still running stops at its next step
  }

  /** Bytes of the stored channels per cell and layer before packing (float16). */
  get storedBytesPerCellLayer() { return 2 * this.inputPlan.stored.length; }

  /**
   * Split `layers` into groups of at most SCAN_LAYERS layers (the scan packs each sub-group's sums straight into its
   * group's buffer) whose unpacked stored field of the whole padded grid stays within `limit` bytes.  Every group
   * gets a slot table (layer -> slot).
   */
  planGroups(layers, limit) {
    const pixels = this.gx * this.gy;
    const perLayer = this.storedBytesPerCellLayer * pixels;
    let groupLayers = Math.max(1, Math.min(layers.length, Math.floor(limit / perLayer)));
    groupLayers = Math.min(groupLayers, SCAN_LAYERS);
    if (perLayer > limit) {
      throw new DemoError(ErrorCode.TOO_LARGE,
                          `one layer needs ${perLayer} bytes, above the ${limit}-byte buffer limit`);
    }
    const groups = [];
    for (let i = 0; i < layers.length; i += groupLayers) {
      const list = layers.slice(i, i + groupLayers);
      const slots = new Int32Array(SLOT_TABLE).fill(-1);
      list.forEach((layer, k) => { slots[layer] = k; });
      // packTiles: tiles of its packed field (pack); upTiles ... upGrid: its along-wind tiles (ensureUpTiles)
      groups.push({ layers: list, slots, field: null, slotBuffer: null, layerBuffer: null, packTiles: 0,
                    upTiles: null, upCapacity: 0, upCount: 0, upGrid: null });
    }
    return groups;
  }

  /** The per-group layer lists and slot tables on the GPU (scan allocates the packed field once its size is known). */
  allocate(groups) {
    const { context } = this;
    for (const group of groups) {
      group.slotBuffer = context.upload(group.slots, "mddf.slots");
      group.layerBuffer = context.upload(new Uint32Array(group.layers), "mddf.layers");
    }
    this.groups = groups;
  }

  /** Free the buffers of the groups (context.free lets each go once). */
  releaseGroups() {
    for (const g of this.groups ?? []) {
      for (const b of [g.field, g.upTiles, g.slotBuffer, g.layerBuffer]) this.context.free(b);
    }
    this.groups = null;
  }

  /** Digital lines of every azimuth on the padded grid (scan): offsets, per-azimuth table, Fourier weights. */
  scanTables() {
    if (this.scanInfo) return this.scanInfo;
    const { gx, gy } = this, maxMajor = Math.max(gx, gy);
    const offsets = new Int32Array(ANGLE_COUNT * maxMajor);
    const table = new ArrayBuffer(ANGLE_COUNT * 32);
    const tu = new Uint32Array(table), ti = new Int32Array(table), tf = new Float32Array(table);
    const trig = new Float32Array(ANGLE_COUNT * 4);
    let lineMax = 0;
    for (let a = 0; a < ANGLE_COUNT; a++) {
      const c = this.cosT[a], s = this.sinT[a];
      const majorX = Math.abs(c) >= Math.abs(s);
      const major = majorX ? c : s, minor = majorX ? s : c;
      const slope = Math.abs(minor / major);
      const majorSize = majorX ? gx : gy, minorSize = majorX ? gy : gx;
      for (let i = 0; i < majorSize; i++) offsets[a * maxMajor + i] = Math.round(slope * i);
      const maxShift = offsets[a * maxMajor + majorSize - 1];
      const lineCount = minorSize + maxShift;
      lineMax = Math.max(lineMax, lineCount);
      tu[8 * a] = majorX ? 1 : 0; ti[8 * a + 1] = major >= 0 ? 1 : -1; ti[8 * a + 2] = minor >= 0 ? 1 : -1;
      ti[8 * a + 3] = maxShift; tu[8 * a + 4] = lineCount; tu[8 * a + 5] = a * maxMajor;
      tf[8 * a + 6] = CELL_M * Math.sqrt(1 + slope * slope);
      trig.set([c, s, Math.cos(4 * Math.PI * a / ANGLE_COUNT), Math.sin(4 * Math.PI * a / ANGLE_COUNT)], 4 * a);
    }
    this.offsetBuffer = this.context.upload(offsets, "mddf.offsets");
    this.tableBuffer = this.context.upload(tu, "mddf.scanTable");
    this.trigBuffer = this.context.upload(trig, "mddf.trig");
    this.scanInfo = { lineMax };
    return this.scanInfo;
  }

  /**
   * Stored channels of one group by the whole-field scan (whole padded grid): all its heights at once (a packed group
   * is packed from one scan's sums, so it holds at most SCAN_LAYERS heights); azimuths in batches whose raw distances
   * stay within SCAN_RAW_BYTES.
   */
  async scan(group, onProgress) {
    if (this.destroyed) return;
    const { context, gx, gy } = this;
    const { lineMax } = this.scanTables();
    const pixels = gx * gy, S = this.inputPlan.stored.length;
    const count = group.layers.length, pairs = Math.ceil(count / 2);
    const batch = Math.max(1, Math.min(ANGLE_COUNT, Math.floor(SCAN_RAW_BYTES / (pairs * pixels * 4))));
    const rawBytes = batch * pairs * pixels * 4, sumsBytes = count * pixels * S * 4;
    if (!this.scanRaw || this.scanRaw.size < rawBytes) {
      context.free(this.scanRaw);
      this.scanRaw = context.storage(rawBytes, "mddf.scanRaw");
    }
    if (!this.sumsBuffer || this.sumsBuffer.size < sumsBytes) {
      context.free(this.sumsBuffer);
      this.sumsBuffer = context.storage(sumsBytes, "mddf.sums");
    }
    const clear = context.device.createCommandEncoder();
    clear.clearBuffer(this.sumsBuffer, 0, count * pixels * S * 4);
    context.device.queue.submit([clear.finish()]);
    const inFlight = [];
    for (let a0 = 0; a0 < ANGLE_COUNT; a0 += batch) {
      if (this.destroyed) return;            // the city was closed while it was being scanned
      const n = Math.min(batch, ANGLE_COUNT - a0);
      const encoder = context.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      // the scan shader's sub-group starts at the group's first layer (0)
      const su = context.uniform([["u", gx], ["u", gy], ["u", pixels], ["u", 0], ["u", count], ["u", pairs],
                                  ["u", a0], ["u", n], ["u", lineMax], ["u", 0], ["u", 0], ["u", 0]]);
      context.dispatch(pass, this.scanPipeline,
        [this.heightBuffer, group.layerBuffer, this.offsetBuffer, this.tableBuffer, this.scanRaw, su],
        lineMax * pairs * n, MDDF_WORKGROUP);
      const au = context.uniform([["u", pixels], ["u", count], ["u", pairs], ["u", a0], ["u", n],
                                  ["u", 0], ["u", 0], ["u", 0]]);
      context.dispatch(pass, this.accumulatePipeline, [this.scanRaw, this.trigBuffer, this.sumsBuffer, au],
                       pixels * count, MDDF_WORKGROUP);
      pass.end();
      context.device.queue.submit([encoder.finish()]);
      inFlight.push(context.device.queue.onSubmittedWorkDone());
      if (inFlight.length >= SCAN_IN_FLIGHT) await inFlight.shift();
      onProgress((a0 + n) / ANGLE_COUNT);
    }
    if (this.destroyed) return;
    await this.packGroup(group, count, pixels);
    if (this.destroyed) return;              // destroy() has let the scan's buffers go
    await context.device.queue.onSubmittedWorkDone();
    if (this.destroyed) return;
    // transient: only while a city is scanned
    context.free(this.scanRaw);
    this.scanRaw = null;
    context.free(this.sumsBuffer);
    this.sumsBuffer = null;
    context.releaseScratch();
  }

  /**
   * PACKED FIELD of one group from the scan sums: block headers and tile sizes, offsets, payload.  The payload pass
   * has its own uniform, made after the read-back: a scratch buffer made before an await may be released meanwhile by
   * another task (releaseScratch).  A city closed meanwhile stops it (its own buffers let go).
   */
  async packGroup(group, count, pixels) {
    const { context, gx, gy } = this;
    const { tx, ty } = this.packed;
    const nTiles = count * this.inputPlan.stored.length * tx * ty;
    const headers = context.storage(nTiles * PACK_BLOCKS * 4, "mddf.packHeaders");
    const tileWords = context.storage(nTiles * 4, "mddf.packTileWords");
    const eu = context.uniform([["u", gx], ["u", gy], ["u", pixels], ["u", count], ["u", nTiles],
                                ["u", 0], ["u", 0], ["u", 0]]);
    const tu = context.uniform([["u", nTiles], ["u", 0], ["u", 0], ["u", 0]]);
    const freeStaging = () => { context.free(headers); context.free(tileWords); };
    const encode = (encoder, to) => {
      const pass = encoder.beginComputePass();
      // one thread per block of every tile
      context.dispatch(pass, this.packHeaderPipeline, [this.sumsBuffer, headers, eu], nTiles * PACK_BLOCKS,
                       MDDF_WORKGROUP);
      context.dispatch(pass, this.packTilePipeline, [headers, tileWords, tu], nTiles, MDDF_WORKGROUP);
      pass.end();
      encoder.copyBufferToBuffer(tileWords, 0, to, 0, nTiles * 4);
    };
    let words;
    try {
      // null: the city was closed meanwhile (nothing more is done for it)
      words = await context.readBack(nTiles * 4, encode, "mddf.packTileWords read",
                                     (range) => (this.destroyed ? null : new Uint32Array(range.slice(0))));
    } catch (error) {
      freeStaging();                   // a lost device
      throw error;
    }
    if (!words) { freeStaging(); return; }
    const offsets = new Uint32Array(nTiles);
    let at = nTiles + nTiles * PACK_BLOCKS;
    for (let t = 0; t < nTiles; t++) { offsets[t] = at; at += words[t]; }
    context.free(group.field);
    group.field = context.storage(Math.max(4, at * 4), "mddf.field");
    group.packTiles = nTiles;
    context.write(group.field, offsets);
    const encoder = context.device.createCommandEncoder();
    encoder.copyBufferToBuffer(headers, 0, group.field, nTiles * 4, nTiles * PACK_BLOCKS * 4);
    const pass = encoder.beginComputePass();
    const pu = context.uniform([["u", gx], ["u", gy], ["u", pixels], ["u", count], ["u", nTiles],
                                ["u", 0], ["u", 0], ["u", 0]]);
    context.dispatch(pass, this.packPayloadPipeline, [this.sumsBuffer, group.field, pu], nTiles * PACK_BLOCKS,
                     MDDF_WORKGROUP);
    pass.end();
    context.device.queue.submit([encoder.finish()]);
    await context.device.queue.onSubmittedWorkDone();
    freeStaging();
  }

  /** Upwind azimuth (city grid) of the wind direction theta, as float32 cos / sin. */
  upwindDirection(theta) {
    // Wind toward theta comes from theta + pi.  On the 2-degree azimuth grid the azimuth table's values are used
    // (the same float32 numbers as the 180-azimuth stored channels), elsewhere float32 cos / sin of the angle.
    let up = (theta + Math.PI) % (2 * Math.PI);
    if (up < 0) up += 2 * Math.PI;
    const index = up / (2 * Math.PI) * ANGLE_COUNT;
    const nearest = Math.round(index) % ANGLE_COUNT;
    if (Math.abs(index - Math.round(index)) < 1e-6) return [this.cosT[nearest], this.sinT[nearest]];
    return [Math.fround(Math.cos(up)), Math.fround(Math.sin(up))];
  }

  /**
   * Wind-frame tile grid of direction theta: UP_TILE x UP_TILE tiles of wind-frame cells covering every cell a patch
   * can have - a kept patch overlaps the city square (tiling.js corridorTiling), so its cells lie within the square's
   * wind-frame bounding box grown by the patch size.  table: tile -> slot in the tile buffers (-1: not computed).
   */
  upTileGrid(theta, patch) {
    const frame = windFrame(theta, this.nx, this.ny);
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (const ix of [-0.5, this.nx - 0.5]) for (const iy of [-0.5, this.ny - 0.5]) {
      const [sx, sy] = frame.toWind([ix, iy]);
      x0 = Math.min(x0, sx); x1 = Math.max(x1, sx); y0 = Math.min(y0, sy); y1 = Math.max(y1, sy);
    }
    const ox = Math.floor((x0 - patch) / UP_TILE) * UP_TILE, oy = Math.floor((y0 - patch) / UP_TILE) * UP_TILE;
    const tw = Math.ceil((x1 + patch + 1 - ox) / UP_TILE), th = Math.ceil((y1 + patch + 1 - oy) / UP_TILE);
    return { theta, x0: ox, y0: oy, tw, th, table: new Int32Array(tw * th).fill(-1), count: 0 };
  }

  /**
   * Along-wind channels at the wind-frame cells of `patches` ([sy, sx] origins, patch x patch cells) for every group:
   * the tiles not computed yet for this direction are computed now (all of them after a direction change).  Each
   * group keeps its tiles in one buffer, tile after tile ([tile][slot][cell][channel], float16), grown when needed
   * with the computed tiles copied over.  onCompute is called before any tile is computed.  Returns the number of
   * tiles computed.
   */
  ensureUpTiles(groups, theta, patches, patch, onCompute) {
    const { context } = this;
    let g = this.upGrid;
    // a new direction, or groups that do not hold this grid's tiles: start over
    if (!g || g.theta !== theta || (g.count > 0 && groups.some((gr) => gr.upGrid !== g))) {
      g = this.upGrid = this.upTileGrid(theta, patch);
    }
    const fresh = [];
    for (const [sy, sx] of patches) {
      const tx0 = Math.floor((sx - g.x0) / UP_TILE), tx1 = Math.floor((sx + patch - 1 - g.x0) / UP_TILE);
      const ty0 = Math.floor((sy - g.y0) / UP_TILE), ty1 = Math.floor((sy + patch - 1 - g.y0) / UP_TILE);
      if (tx0 < 0 || ty0 < 0 || tx1 >= g.tw || ty1 >= g.th) {
        throw new Error("a patch lies outside the wind-frame tile grid");
      }
      for (let ty = ty0; ty <= ty1; ty++) for (let tx = tx0; tx <= tx1; tx++) {
        const k = ty * g.tw + tx;
        if (g.table[k] < 0) {
          g.table[k] = g.count + fresh.length / 3;
          fresh.push(tx, ty, g.table[k]);
        }
      }
    }
    const n = fresh.length / 3;
    if (!n) return 0;
    onCompute();
    const before = g.count, after = before + n, U = this.inputPlan.up.length, cells = UP_TILE * UP_TILE;
    const encoder = context.device.createCommandEncoder();
    const old = [];
    for (const group of groups) {
      const perTile = group.layers.length * cells * U * 2;
      if (group.upGrid !== g) group.upCount = 0;           // tiles of another direction: none kept
      if (!group.upTiles || group.upCapacity < after) {
        const capacity = Math.min(g.tw * g.th, Math.max(after, Math.ceil(group.upCapacity * 1.5)));
        const buffer = context.storage(capacity * perTile, "mddf.upTiles");
        if (group.upTiles && group.upCount) {
          encoder.copyBufferToBuffer(group.upTiles, 0, buffer, 0, group.upCount * perTile);
        }
        if (group.upTiles) old.push(group.upTiles);
        group.upTiles = buffer; group.upCapacity = capacity;
      }
      group.upGrid = g;
      encoder.clearBuffer(group.upTiles, before * perTile, n * perTile);
      group.upCount = after;
    }
    const tileBuffer = context.upload(new Int32Array(fresh), "mddf.upTileList");
    const [cosU, sinU] = this.upwindDirection(theta);
    const pass = encoder.beginComputePass();
    for (const group of groups) {
      const buffers = [this.heightBuffer, group.layerBuffer, group.upTiles, tileBuffer];
      context.dispatch(pass, this.alongPipeline, [...buffers, context.uniform([
        ["u", this.gx], ["u", this.gy], ["u", n], ["u", group.layers.length],
        ["u", this.margin], ["u", this.nx], ["u", this.ny], ["u", 0],
        ["i", g.x0], ["i", g.y0], ["u", 0], ["u", 0],
        ["f", cosU], ["f", sinU], ["f", Math.cos(theta)], ["f", Math.sin(theta)],
        ["f", (this.nx - 1) / 2], ["f", (this.ny - 1) / 2], ["f", 0], ["f", 0],
      ])], n * cells * Math.ceil(group.layers.length / LAYER_GROUP), MDDF_WORKGROUP);
    }
    pass.end();
    if (!this.upTableBuffer || this.upTableBuffer.size !== g.table.byteLength) {
      context.free(this.upTableBuffer);
      this.upTableBuffer = context.storage(g.table.byteLength, "mddf.upTable");
    }
    context.write(this.upTableBuffer, g.table);
    context.device.queue.submit([encoder.finish()]);
    for (const b of old) context.free(b);                // destroyed once the copy above has run
    context.free(tileBuffer);
    g.count = after;
    return n;
  }

  /**
   * Encode the input tensor for patch x patch-cell patches with wind-frame origins [sy, layer, sx] of one group.
   * Returns the uploaded origins' buffer (the blend pass reads it too; the caller frees it).
   */
  assemble(encoder, origins, group, theta, patch, outputBuffer) {
    const { context, inputPlan, gx, gy } = this;
    const batch = origins.length / 3;
    const originBuffer = context.upload(origins, "assemble.origins");
    const uniform = context.uniform([
      ["u", batch], ["u", inputPlan.inputs.length], ["u", patch], ["u", patch],
      // the stored field's region: the whole padded grid
      ["u", gx], ["u", gy], ["u", 0], ["u", 0], ["u", gx], ["u", gy], ["u", gx * gy], ["u", this.margin],
      ["f", Math.cos(theta)], ["f", Math.sin(theta)], ["f", theta], ["f", (this.ny - 1) / 2],
      ["f", (this.nx - 1) / 2], ["u", group.packTiles], ["f", 0], ["f", 0],
      ["i", this.upGrid?.x0 ?? 0], ["i", this.upGrid?.y0 ?? 0],
      ["u", this.upGrid?.tw ?? 0], ["u", this.upGrid?.th ?? 0],
      ["u", group.layers.length], ["u", 0], ["u", 0], ["u", 0],
    ]);
    const buffers = [group.field];
    if (inputPlan.usesHeight) buffers.push(this.heightBuffer);
    buffers.push(originBuffer, group.slotBuffer);
    if (inputPlan.up.length) {
      if (group.upGrid !== this.upGrid) throw new Error("along-wind tiles of another direction (ensureUpTiles first)");
      buffers.push(group.upTiles, this.upTableBuffer);
    }
    buffers.push(outputBuffer, uniform);
    const pass = encoder.beginComputePass();
    context.dispatch(pass, this.assemblePipeline, buffers, batch * inputPlan.inputs.length * patch * patch);
    pass.end();
    return originBuffer;
  }

  /** Let the city's GPU buffers go; a scan still running stops at its next step. */
  destroy() {
    this.destroyed = true;
    this.releaseGroups();
    for (const buffer of [this.heightBuffer, this.sumsBuffer, this.offsetBuffer, this.tableBuffer, this.trigBuffer,
                          this.scanRaw, this.upTableBuffer]) this.context.free(buffer);
  }
}
