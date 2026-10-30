// The WebGPU device of the wind model: buffers, uniforms, the compute pipeline cache, dispatch and read-back.
import { DemoError, ErrorCode } from "../shared/errors.js";
import { yieldTurn, TURN_AFTER_MS } from "../shared/turn.js";
import { SharedRing } from "./shared_ring.js";

/** Threads per workgroup of the compute shaders that do not set their own (written into their text). */
export const WORKGROUP = 256;

const READ_SLICE = 1 << 20;            // values copied per slice of a read-back

/** A WebGPU device of the wind model (GpuContext.create) with its buffers, pipelines and read-back. */
export class GpuContext {
  constructor(device, useF16, adapterInfo) {
    this.adapterInfo = adapterInfo;   // { vendor, architecture, ... } of the adapter the device came from, or null
    this.device = device;
    this.useF16 = useF16;
    this.pipelines = new Map();
    this.scratch = [];
    this.halfRing = new SharedRing();  // the read-backs into shared memory (readHalf)
  }

  /**
   * A high-performance device, with float16 shaders when the adapter has them and allowF16.  Throws DemoError GPU
   * when the browser has no WebGPU or no adapter.
   */
  static async create({ allowF16 = true } = {}) {
    if (!navigator.gpu) {
      throw new DemoError(ErrorCode.GPU, "WebGPU unavailable. Use desktop Chrome or Edge over https:// or " +
                                         "http://localhost (a secure context is required).");
    }
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new DemoError(ErrorCode.GPU, "No WebGPU adapter available.");
    const f16 = allowF16 && adapter.features.has("shader-f16");
    const device = await adapter.requestDevice({
      requiredFeatures: f16 ? ["shader-f16"] : [],
      requiredLimits: {
        maxStorageBufferBindingSize: Math.min(adapter.limits.maxStorageBufferBindingSize, 1 << 30),
        maxBufferSize: Math.min(adapter.limits.maxBufferSize, 1 << 30),
      },
    });
    return new GpuContext(device, f16, adapter.info ?? null);
  }

  /** Let the device go (its buffers with it). */
  destroy() {
    try {
      this.device.destroy();
    } catch {
      // already gone
    }
  }

  /** A buffer of at least `bytes` (rounded up to 4) with `usage`. */
  buffer(bytes, usage, label) {
    const size = Math.max(4, Math.ceil(bytes / 4) * 4);
    const buffer = this.device.createBuffer({ size, usage, label });
    buffer.tracked = true;             // made here: free() lets it go, once
    return buffer;
  }

  /** A storage buffer that can be copied to and from. */
  storage(bytes, label) {
    const usage = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    return this.buffer(bytes, usage, label);
  }

  /** Destroy a buffer made here (once; null and other buffers are ignored). */
  free(buffer) {
    if (buffer && buffer.tracked) {
      buffer.tracked = false;
      buffer.destroy();
    }
  }

  /** A storage buffer holding `array`. */
  upload(array, label) {
    const buffer = this.storage(array.byteLength, label);
    this.write(buffer, array);
    return buffer;
  }

  /** Upload `array` to the start of `buffer`. */
  write(buffer, array) {
    // writeBuffer needs a multiple of four bytes: odd-length 16-bit arrays are padded up to the allocated size
    if (array.byteLength % 4) {
      const padded = new Uint8Array(Math.ceil(array.byteLength / 4) * 4);
      padded.set(new Uint8Array(array.buffer, array.byteOffset, array.byteLength));
      this.device.queue.writeBuffer(buffer, 0, padded);
    } else {
      this.device.queue.writeBuffer(buffer, 0, array);
    }
  }

  /**
   * A small per-dispatch uniform block, freed by releaseScratch after the submission.  `fields`: [kind, value] with
   * kind "u", "i" or "f" - explicit, because a float that happens to be integral (cos 0 = 1.0) would otherwise be
   * written with integer bits.
   */
  uniform(fields) {
    const buffer = this.keptUniform(fields);
    this.scratch.push(buffer);
    return buffer;
  }

  /** A uniform block of `fields` (as uniform) that lives as long as the device: for prepared dispatches. */
  keptUniform(fields) {
    const size = Math.ceil(fields.length / 4) * 16;
    const data = new ArrayBuffer(size);
    const asU32 = new Uint32Array(data), asI32 = new Int32Array(data), asF32 = new Float32Array(data);
    fields.forEach(([kind, value], index) => {
      if (kind === "u") asU32[index] = value;
      else if (kind === "i") asI32[index] = value;
      else asF32[index] = value;
    });
    const buffer = this.device.createBuffer({ size, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.device.queue.writeBuffer(buffer, 0, data);
    return buffer;
  }

  /** Destroy the uniform blocks made since the last call (their dispatches have been submitted). */
  releaseScratch() {
    for (const buffer of this.scratch) buffer.destroy();
    this.scratch = [];
  }

  /**
   * The compute pipeline of WGSL `code` (entry point main, layout "auto"), compiled once.  Every uniform a shader
   * declares must be read: an unread binding is stripped from an "auto" layout and the bind group then fails.
   */
  pipeline(code) {
    if (!this.pipelines.has(code)) {
      const module = this.device.createShaderModule({ code });
      const pipeline = this.device.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } });
      this.pipelines.set(code, pipeline);
    }
    return this.pipelines.get(code);
  }

  /**
   * Dispatch `threads` invocations of `pipeline` with `buffers` bound in order.  One dispatch dimension is capped
   * (65535 on common hardware), so the excess spills into the second dimension; the shaders index linearly through
   * num_workgroups.
   */
  dispatch(pass, pipeline, buffers, threads, workgroup = WORKGROUP) {
    this.dispatchGroups(pass, pipeline, buffers, ...this.workgroups(threads, workgroup));
  }

  /** The workgroups [x, y] of `threads` invocations (dispatch): x capped at the device's limit, the excess in y. */
  workgroups(threads, workgroup = WORKGROUP) {
    const groups = Math.max(1, Math.ceil(threads / workgroup));
    const limit = this.device.limits.maxComputeWorkgroupsPerDimension;
    return [Math.min(groups, limit), Math.ceil(groups / limit)];
  }

  /** Dispatch `pipeline` with `buffers` bound in order, as `groups` workgroups (x[, y], as dispatchWorkgroups). */
  dispatchGroups(pass, pipeline, buffers, ...groups) {
    this.replay(pass, this.prepared(pipeline, buffers, groups));
  }

  /** A dispatch ready to be encoded again and again (replay): `pipeline` with `buffers` bound in order, `groups`. */
  prepared(pipeline, buffers, groups) {
    const bindGroup = this.device.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    });
    return { pipeline, bindGroup, groups };
  }

  /** Encode a prepared dispatch into `pass`. */
  replay(pass, { pipeline, bindGroup, groups }) {
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(...groups);
  }

  /**
   * GPU data read back: a staging buffer of `bytes` is filled by encode(encoder, staging) (copies, and any passes
   * before them), mapped, and handed to read(mapped ArrayBuffer); returns what read returns (read may be async).  The
   * staging buffer is let go in every case (a lost device fails the map).
   */
  async readBack(bytes, encode, label, read) {
    const staging = this.buffer(bytes, GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, label);
    const encoder = this.device.createCommandEncoder();
    encode(encoder, staging);
    this.device.queue.submit([encoder.finish()]);
    try {
      await staging.mapAsync(GPUMapMode.READ);
      const result = await read(staging.getMappedRange());
      staging.unmap();
      return result;
    } finally {
      this.free(staging);
    }
  }

  /**
   * Read `count` float16 values of `buffer` as they are (Uint16Array).  shared: into shared memory (shared_ring.js;
   * all `count` values are copied over, so nothing of an earlier read is left in the view).  The values are copied in
   * slices, with a turn for the page after TURN_AFTER_MS, so a large field does not block the page.
   */
  async readHalf(buffer, count, shared) {
    const copy = (encoder, to) => encoder.copyBufferToBuffer(buffer, 0, to, 0, to.size);
    return this.readBack(count * 2, copy, "staging", (range) => this.copyHalves(range, count, shared));
  }

  /** The `count` float16 values of a mapped read-back range copied out (readHalf), in slices. */
  async copyHalves(range, count, shared) {
    const out = shared ? new Uint16Array(this.halfRing.next(count * 2), 0, count) : new Uint16Array(count);
    const mapped = new Uint16Array(range, 0, count);
    let slice = performance.now();
    for (let start = 0; start < count; start += READ_SLICE) {
      out.set(mapped.subarray(start, Math.min(count, start + READ_SLICE)), start);
      if (start + READ_SLICE < count && performance.now() - slice > TURN_AFTER_MS) {
        await yieldTurn();
        slice = performance.now();
      }
    }
    return out;
  }
}
