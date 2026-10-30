// The wind model's 2-D Fourier neural operator on WebGPU, driven entirely by the exported metadata
// (assets/model2d/metadata.json and weights.bin).
//
// The spectral convolution keeps only a few low modes, so the transform is evaluated as small dense DFT products
// rather than an FFT: exact, plain matrix work, and free of any browser FFT dependency.  The staged formulation agrees
// with the PyTorch model to 2.6e-6 (largest output difference).  The spectral weights arrive as a Tucker core plus
// factor matrices and are contracted into the dense stack once, on the GPU, when the model loads.
import { toHalf, fromHalf } from "../shared/numeric.js";
import { modelMisfit } from "../shared/errors.js";
import { WORKGROUP } from "./context.js";
import { shaders, CONTRACT, CONV_BLOCK } from "./fno_shaders.js";

/** The truncated DFT tables of a patch: forward and inverse, along columns (cos, sin) and rows (re, im). */
function dftTables(patch, modes1, modes2) {
  const rows = 2 * modes1;
  const ky = [];
  for (let s = 0; s < modes1; s++) ky.push(s);
  for (let s = patch - modes1; s < patch; s++) ky.push(s);

  const forwardCos = new Float32Array(patch * modes2);
  const forwardSin = new Float32Array(patch * modes2);
  for (let x = 0; x < patch; x++) {
    for (let k = 0; k < modes2; k++) {
      const t = 2 * Math.PI * x * k / patch;
      forwardCos[x * modes2 + k] = Math.cos(t);
      forwardSin[x * modes2 + k] = -Math.sin(t);
    }
  }
  const rowRe = new Float32Array(rows * patch);
  const rowIm = new Float32Array(rows * patch);
  for (let s = 0; s < rows; s++) {
    for (let y = 0; y < patch; y++) {
      const t = 2 * Math.PI * ky[s] * y / patch;
      rowRe[s * patch + y] = Math.cos(t);
      rowIm[s * patch + y] = -Math.sin(t);
    }
  }
  const inverseRowRe = new Float32Array(patch * rows);
  const inverseRowIm = new Float32Array(patch * rows);
  for (let y = 0; y < patch; y++) {
    for (let s = 0; s < rows; s++) {
      const t = 2 * Math.PI * ky[s] * y / patch;
      inverseRowRe[y * rows + s] = Math.cos(t) / patch;
      inverseRowIm[y * rows + s] = Math.sin(t) / patch;
    }
  }
  // Real reconstruction from the retained half-spectrum; the Nyquist bin is never inside the retained range, so every
  // non-zero mode doubles.
  const inverseColCos = new Float32Array(modes2 * patch);
  const inverseColSin = new Float32Array(modes2 * patch);
  for (let k = 0; k < modes2; k++) {
    const amplitude = (k === 0 ? 1 : 2) / patch;
    for (let x = 0; x < patch; x++) {
      const t = 2 * Math.PI * k * x / patch;
      inverseColCos[k * patch + x] = amplitude * Math.cos(t);
      inverseColSin[k * patch + x] = -amplitude * Math.sin(t);
    }
  }
  return { forwardCos, forwardSin, rowRe, rowIm, inverseRowRe, inverseRowIm,
           inverseColCos, inverseColSin };
}

/** Tensor `name` of the model (metadata.tensors) from weights.bin, as float32 values. */
function tensorOf(metadata, weights, name) {
  const entry = metadata.tensors[name];
  if (!entry) throw modelMisfit(`missing tensor ${name}`);
  return fromHalf(new Uint16Array(weights, entry.offset, entry.count));
}

/** The operator: its weights on the GPU, its activation buffers for up to maxBatch patches, and encode(). */
export class Fno {
  /**
   * @param {import("./context.js").GpuContext} context
   * @param {object} metadata the exported model metadata
   * @param {ArrayBuffer} weights weights.bin
   * @param {number} maxBatch patches per submission
   */
  constructor(context, metadata, weights, maxBatch) {
    const model = metadata.model;
    this.context = context;
    this.metadata = metadata;
    this.width = model.width;
    this.depth = model.layers;               // the operator's layers (blocks)
    this.inputChannels = model.input_channels;
    this.outputChannels = model.output_channels;
    this.patch = metadata.inference.patch[0];
    this.modes1 = model.modes[0];
    this.modes2 = model.modes[1];
    this.store = context.useF16 ? "f16" : "f32";
    this.elementBytes = this.store === "f16" ? 2 : 4;
    if (metadata.inference.patch[0] !== metadata.inference.patch[1]) {
      throw modelMisfit("only square patches are supported");
    }
    if ((this.patch * this.patch) % WORKGROUP) {          // the 1x1 convolution shares a workgroup's work per area
      throw modelMisfit(`the patch area must be a multiple of ${WORKGROUP}`);
    }

    const spec = { store: this.store, patch: this.patch, modes1: this.modes1, modes2: this.modes2,
                   cinMax: Math.max(this.inputChannels, 2 * this.width) };
    this.pipelines = {};
    for (const [name, code] of Object.entries(shaders(spec))) {
      this.pipelines[name] = context.pipeline(code);
    }
    this.contractPipeline = context.pipeline(CONTRACT);

    const read = (name) => tensorOf(metadata, weights, name);

    this.tables = {};
    for (const [name, data] of Object.entries(dftTables(this.patch, this.modes1, this.modes2))) {
      this.tables[name] = context.upload(data, `fno.table.${name}`);
    }

    this.weights = {};
    const upload = (key, name) => {
      const values = read(name);
      this.weights[key] = this.store === "f16"
        ? context.upload(toHalf(values), `fno.${key}`)
        : context.upload(values, `fno.${key}`);
    };
    upload("p1w", "p.mlp1.weight");
    upload("p1b", "p.mlp1.bias");
    upload("p2w", "p.mlp2.weight");
    upload("p2b", "p.mlp2.bias");
    upload("q1w", "q.mlp1.weight");
    upload("q1b", "q.mlp1.bias");
    upload("q2w", "q.mlp2.weight");
    upload("q2b", "q.mlp2.bias");
    for (let l = 0; l < this.depth; l++) {
      upload(`m1w${l}`, `mlp_layers.${l}.mlp1.weight`);
      upload(`m1b${l}`, `mlp_layers.${l}.mlp1.bias`);
      upload(`m2w${l}`, `mlp_layers.${l}.mlp2.weight`);
      upload(`m2b${l}`, `mlp_layers.${l}.mlp2.bias`);
      upload(`lw${l}`, `w_layers.${l}.weight`);
      upload(`lb${l}`, `w_layers.${l}.bias`);
    }

    this.buildSpectralWeights(read);
    this.allocate(maxBatch);
    this.passes = new Map();           // batch size -> its prepared dispatches (encode)
  }

  /** Contract the Tucker core with its factors (read(name): a tensor's values) into the dense weight stack. */
  buildSpectralWeights(read) {
    const { context, metadata } = this;
    const tucker = metadata.tucker;
    let dims = tucker.core_shape.slice();
    let re = context.upload(read(`${tucker.core}.re`), "tucker.core.re");
    let im = context.upload(read(`${tucker.core}.im`), "tucker.core.im");

    for (const factor of tucker.factors) {
      const [outer, rank] = factor.shape;
      if (dims[factor.mode] !== rank) {
        throw modelMisfit(`factor ${factor.name} does not match the core`);
      }
      const next = dims.slice();
      next[factor.mode] = outer;
      const total = next.reduce((a, b) => a * b, 1);
      const strideMode = next.slice(factor.mode + 1).reduce((a, b) => a * b, 1);
      const outRe = context.storage(total * 4, "tucker.re");
      const outIm = context.storage(total * 4, "tucker.im");
      const facRe = context.upload(read(`${factor.name}.re`), "tucker.f");
      const facIm = context.upload(read(`${factor.name}.im`), "tucker.f");
      const uniform = context.uniform([
        ["u", next[0]], ["u", next[1]], ["u", next[2]], ["u", next[3]],
        ["u", next[4]], ["u", factor.mode], ["u", outer], ["u", rank],
        ["u", strideMode], ["u", 0], ["u", total], ["u", 0],
      ]);
      const encoder = context.device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      context.dispatch(pass, this.contractPipeline, [re, im, facRe, facIm, outRe, outIm, uniform], total);
      pass.end();
      context.device.queue.submit([encoder.finish()]);
      context.free(re); context.free(im);
      context.free(facRe); context.free(facIm);
      re = outRe; im = outIm;
      dims = next;
    }
    context.releaseScratch();
    const expected = [2 * this.depth, this.width, this.width, this.modes1, this.modes2];
    if (dims.join() !== expected.join()) {
      throw modelMisfit(`spectral stack is ${dims} but ${expected} was expected`);
    }
    this.spectralRe = re;
    this.spectralIm = im;
  }

  /**
   * The activation buffers for `batch` patches.  A forward pass (encode) has at most three activations in use at a
   * time, so they share three buffers, each holding one of its activations at a time and none bound twice in one
   * dispatch: x (the state through the layers); a large one (4C channels): the input, the inverse transform's output
   * t2, the gates, the spectral branch, the output; a medium one (2C): the lifted input `wide`, the normalised state
   * t1, the hidden gate, the local branch.  Of the spectra, x and w share: x is spent once y is made, w is made from z.
   */
  allocate(batch) {
    const { context } = this;
    const area = this.patch * this.patch;
    const C = this.width;
    const rows = 2 * this.modes1;
    const activation = (channels, label) => context.storage(batch * channels * area * this.elementBytes, label);
    const complex = (count, label) => context.storage(batch * C * count * 4, label);
    const large = activation(Math.max(4 * C, this.inputChannels, this.outputChannels), "fno.large");
    const medium = activation(2 * C, "fno.medium");
    const xwRe = complex(this.patch * this.modes2, "fno.xwRe"), xwIm = complex(this.patch * this.modes2, "fno.xwIm");
    this.buffers = {
      x: activation(C, "fno.x"),
      input: large, t2: large, gates: large, spectral: large, output: large,
      wide: medium, t1: medium, hidden: medium, local: medium,
      xRe: xwRe, xIm: xwIm,
      yRe: complex(rows * this.modes2, "fno.yRe"),
      yIm: complex(rows * this.modes2, "fno.yIm"),
      zRe: complex(rows * this.modes2, "fno.zRe"),
      zIm: complex(rows * this.modes2, "fno.zIm"),
      wRe: xwRe, wIm: xwIm,
    };
  }

  /** The buffer the next batch's input patches are written to. */
  get inputBuffer() { return this.buffers.input; }

  /** Threads to dispatch for a 1x1 convolution with `cout` outputs (CONV_BLOCK outputs per thread). */
  convThreads(batch, cout) {
    return batch * Math.ceil(cout / CONV_BLOCK) * this.patch * this.patch;
  }

  /**
   * Encode a forward pass over `batch` patches already present in `input`.  Its dispatches (pipelines, bind groups
   * with their uniforms, workgroups) depend on the batch size only: prepared once per size and kept.
   */
  encode(encoder, batch) {
    if (!this.passes.has(batch)) this.passes.set(batch, this.dispatches(batch));
    const pass = encoder.beginComputePass();
    for (const dispatch of this.passes.get(batch)) this.context.replay(pass, dispatch);
    pass.end();
    return this.buffers.output;
  }

  /** The dispatches of a forward pass over `batch` patches (encode), in order. */
  dispatches(batch) {
    const { context, buffers: b, pipelines: p, tables: t } = this;
    const C = this.width;
    const area = this.patch * this.patch;
    const rows = 2 * this.modes1;
    const list = [];
    const run = (pipeline, resources, values, threads) => {
      list.push(context.prepared(pipeline, [...resources, context.keptUniform(values)], context.workgroups(threads)));
    };
    // one workgroup per (patch, channel)
    const norm = (source, target, pipeline = p.instanceNorm) => {
      list.push(context.prepared(pipeline, [source, target], [batch * C]));
    };

    run(p.conv1x1, [b.input, this.weights.p1w, this.weights.p1b, b.wide],
        [["u", batch], ["u", this.inputChannels], ["u", 2 * C], ["u", 1]],
        this.convThreads(batch, 2 * C));
    run(p.conv1x1, [b.wide, this.weights.p2w, this.weights.p2b, b.x],
        [["u", batch], ["u", 2 * C], ["u", C], ["u", 0]], this.convThreads(batch, C));

    for (let l = 0; l < this.depth; l++) {
      norm(b.x, b.t1);
      run(p.forwardColumns, [b.t1, t.forwardCos, t.forwardSin, b.xRe, b.xIm],
          [["u", batch], ["u", C], ["u", 0], ["u", 0]], batch * C * this.patch * this.modes2);
      run(p.rowMix, [b.xRe, b.xIm, t.rowRe, t.rowIm, b.yRe, b.yIm],
          [["u", batch], ["u", C], ["u", rows], ["u", this.patch]], batch * C * rows * this.modes2);
      run(p.spectralMix, [b.yRe, b.yIm, this.spectralRe, this.spectralIm, b.zRe, b.zIm],
          [["u", batch], ["u", C], ["u", 2 * l], ["u", 0]], batch * C * rows * this.modes2);
      run(p.rowMix, [b.zRe, b.zIm, t.inverseRowRe, t.inverseRowIm, b.wRe, b.wIm],
          [["u", batch], ["u", C], ["u", this.patch], ["u", rows]], batch * C * this.patch * this.modes2);
      run(p.inverseColumns, [b.wRe, b.wIm, t.inverseColCos, t.inverseColSin, b.t2],
          [["u", batch], ["u", C], ["u", 0], ["u", 0]], batch * C * area);
      norm(b.t2, b.t1, p.instanceNormScaled);
      run(p.conv1x1, [b.t1, this.weights[`m1w${l}`], this.weights[`m1b${l}`], b.gates],
          [["u", batch], ["u", C], ["u", 4 * C], ["u", 0]], this.convThreads(batch, 4 * C));
      run(p.gate, [b.gates, b.hidden], [["u", batch], ["u", 2 * C], ["u", 0], ["u", 0]], batch * 2 * C * area);
      run(p.conv1x1, [b.hidden, this.weights[`m2w${l}`], this.weights[`m2b${l}`], b.spectral],
          [["u", batch], ["u", 2 * C], ["u", C], ["u", 0]], this.convThreads(batch, C));
      run(p.conv1x1, [b.x, this.weights[`lw${l}`], this.weights[`lb${l}`], b.local],
          [["u", batch], ["u", C], ["u", C], ["u", 0]], this.convThreads(batch, C));
      run(p.residual, [b.spectral, b.local, b.x],
          [["u", batch * C * area], ["u", l < this.depth - 1 ? 1 : 0], ["u", 0], ["u", 0]], batch * C * area);
    }

    run(p.conv1x1, [b.x, this.weights.q1w, this.weights.q1b, b.wide],
        [["u", batch], ["u", C], ["u", 2 * C], ["u", 1]], this.convThreads(batch, 2 * C));
    run(p.conv1x1, [b.wide, this.weights.q2w, this.weights.q2b, b.output],
        [["u", batch], ["u", 2 * C], ["u", this.outputChannels], ["u", 0]],
        this.convThreads(batch, this.outputChannels));
    return list;
  }
}
