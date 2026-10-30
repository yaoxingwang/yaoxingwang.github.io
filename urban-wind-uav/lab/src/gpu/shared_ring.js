// Two shared-memory buffers used in turn for the field of each prediction (the read-back and the post-processed
// field): a new shared buffer per prediction would be freed only when every worker thread has collected its
// garbage, and the field of the previous prediction stays readable while the next one is made.

/** Two shared buffers used in turn (see the file header). */
export class SharedRing {
  constructor() {
    this.buffers = [null, null];
    this.slot = 1;                     // the first buffer used is buffers[0]
  }

  /** The next buffer of at least `bytes` (grown when it is too small; its old contents are not cleared). */
  next(bytes) {
    this.slot ^= 1;
    if (!(this.buffers[this.slot]?.byteLength >= bytes)) this.buffers[this.slot] = new SharedArrayBuffer(bytes);
    return this.buffers[this.slot];
  }
}
