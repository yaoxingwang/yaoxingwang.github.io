// Binary min-heaps of the lattice searches (lattice.js).

const INITIAL_CAPACITY = 1 << 16;            // entries of a heap before it first grows

/**
 * Binary min-heap of (key, node) pairs (keys, nodes; size of them in use); a node may be pushed several times
 * (stale entries are skipped).
 */
export class Heap {
  constructor() {
    this.keys = new Float64Array(INITIAL_CAPACITY);
    this.nodes = new Int32Array(INITIAL_CAPACITY);
    this.size = 0;
  }
  /** Add node `val` with priority `key`. */
  push(key, val) {
    if (this.size === this.keys.length) this.grow();
    let i = this.size++;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (this.keys[p] <= key) break;
      this.keys[i] = this.keys[p]; this.nodes[i] = this.nodes[p];
      i = p;
    }
    this.keys[i] = key; this.nodes[i] = val;
  }
  /** Twice the capacity. */
  grow() {
    const keys = new Float64Array(this.size * 2), nodes = new Int32Array(this.size * 2);
    keys.set(this.keys);
    nodes.set(this.nodes);
    this.keys = keys;
    this.nodes = nodes;
  }
  /** Remove and return the node with the smallest key. */
  pop() {
    const val = this.nodes[0], lk = this.keys[--this.size], lv = this.nodes[this.size];
    let i = 0;
    for (;;) {
      let c = 2 * i + 1;
      if (c >= this.size) break;
      if (c + 1 < this.size && this.keys[c + 1] < this.keys[c]) c++;
      if (this.keys[c] >= lk) break;
      this.keys[i] = this.keys[c]; this.nodes[i] = this.nodes[c];
      i = c;
    }
    this.keys[i] = lk; this.nodes[i] = lv;
    return val;
  }
}

/** Binary min-heap with decrease-key: every node at most once (pos[node]: its slot, -1 none). */
export class IndexedHeap extends Heap {
  constructor(nodes) {
    super();
    this.pos = new Int32Array(nodes).fill(-1);
  }
  /** Move entry i (key, val) up to its place. */
  up(i, key, val) {
    const k = this.keys, v = this.nodes, pos = this.pos;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (k[p] <= key) break;
      k[i] = k[p]; v[i] = v[p]; pos[v[i]] = i;
      i = p;
    }
    k[i] = key; v[i] = val; pos[val] = i;
  }
  /** Insert val or lower its key. */
  set(key, val) {
    const at = this.pos[val];
    if (at >= 0) { this.up(at, key, val); return; }
    if (this.size === this.keys.length) this.grow();
    this.up(this.size++, key, val);
  }
  /** Remove and return the node with the smallest key. */
  pop() {
    const k = this.keys, v = this.nodes, pos = this.pos;
    const val = v[0];
    pos[val] = -1;
    const n = --this.size;
    if (n > 0) {
      const lk = k[n], lv = v[n];
      let i = 0;
      for (;;) {
        let c = 2 * i + 1;
        if (c >= n) break;
        if (c + 1 < n && k[c + 1] < k[c]) c++;
        if (k[c] >= lk) break;
        k[i] = k[c]; v[i] = v[c]; pos[v[i]] = i;
        i = c;
      }
      k[i] = lk; v[i] = lv; pos[lv] = i;
    }
    return val;
  }
}
