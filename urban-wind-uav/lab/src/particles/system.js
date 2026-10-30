// Wind particles: every particle is born at a random point of the corridor and the flight band, moves with the wind
// at its own position every frame, and is reborn at a new random point when its life ends, when it hits a building
// or when it leaves the corridor or the band.  So there are no fixed tracks.
//
// Each particle drags a trail of TRAIL points: the head is its current position, the others are snapshots taken
// every TRAIL_DT seconds of screen time.  Every trail point keeps the wind speed at its own position (colour is
// locked to space); the opacity only depends on the place along the trail (head solid, tail transparent) and on
// birth and death fades - the two are independent.
//
// Motion: a fixed time lapse, TIME_LAPSE x real time, the same at every zoom and in every city, so zooming feels
// natural.  The particles are objects of the scene: zooming in, they move faster on screen and their trails grow,
// together with the buildings; a stronger wind moves them faster.  Trails are TRAIL_DT x (TRAIL - 1) seconds of
// motion, so their length is fixed in metres.
import { CELL_M } from "../shared/units.js";
import { insidePolygon } from "../shared/geometry.js";
import { solidAt } from "../city/cells.js";

export const TRAIL = 16;              // points per trail (the head and TRAIL - 1 snapshots)
// the viewer reads the trail points from textures this wide (texels per row; trail point j at (j % W, j / W))
export const POINT_TEXTURE_WIDTH = 1024;
const TIME_LAPSE = 14;                // particles move this many times faster than the air (the route arrows have
                                      // their own time lapse, view/routes.js)
// trail and life are set in seconds of the air's travel (TRAIL_AIR_S between snapshots, a life LIFE_AIR_S) and shown
// TIME_LAPSE times faster, so the particles look the same in space at any time lapse.  Both also scale with 1 / the
// background wind speed (option timeScale = TRAIL_SPEED / background speed: app/particle_feed.js), so the trails keep
// their length when the wind speed changes; only the motion is faster or slower.
const TRAIL_AIR_S = 6, LIFE_AIR_S = [90, 180];
const TRAIL_DT = TRAIL_AIR_S / TIME_LAPSE;              // s of screen time between snapshots
const LIFE = LIFE_AIR_S.map((s) => s / TIME_LAPSE);    // s of screen time
const FADE_IN = 0.4, FADE_OUT = 0.6;  // s

// A particle leaves the allowed space this far (m) outside the corridor (grown about its centroid) or above and below
// the band; it then runs out (see the constructor).
const CORRIDOR_SLACK_M = 30;
const BAND_SLACK_M = 6;
const MIN_BIRTH_SPEED = 0.05;         // m/s: no particle is born in still air
const BIRTH_TRIES = 200;

/** Outward offset of every corner by d metres away from the centroid (an approximate grown polygon). */
function pushFromCentroid(poly, d) {
  const cx = poly.reduce((s, p) => s + p[0], 0) / poly.length;
  const cy = poly.reduce((s, p) => s + p[1], 0) / poly.length;
  return poly.map(([x, y]) => {
    const vx = x - cx, vy = y - cy, l = Math.hypot(vx, vy) || 1;
    return [x + vx / l * d, y + vy / l * d];
  });
}

/** A seeded random generator (mulberry32): () -> [0, 1). */
function mulberry32(seed) {
  return () => {
    seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A set of wind particles with their trails (see the file header). */
export class ParticleSystem {
  /**
   * @param windAt  sampler (x, y, z) metres -> [wx, wy, wz] m/s
   * @param city    {layers, nx, ny}
   * @param polygon corridor polygon in metres (births inside it)
   * @param band    [low, high] metres
   * @param count   number of particles
   * @param seed    seed of the particles' random births (the same seed: the same particles)
   * @param options plane (null: 3-D particles, else their birth plane, see below), timeScale (trail and life times
   *                are multiplied by it), slowSpeed (m/s: births where the wind is slower are always taken, faster
   *                ones less often, spawn)
   */
  constructor(windAt, city, polygon, band, count, seed, { plane, timeScale, slowSpeed }) {
    // plane: 2-D particles are born on the plane, then move in the 3-D wind like all particles -
    //   { kind: "h", z }                        horizontal plane at height z
    //   { kind: "v", origin, u, s0, s1 }        vertical plane through origin along the inflow direction u, s0..s1 m
    // null: 3-D particles, born anywhere in the band
    Object.assign(this, { windAt, city, polygon, band, count, plane, slowSpeed });
    this.trailDt = TRAIL_DT * timeScale;
    this.lifeRange = [LIFE[0] * timeScale, LIFE[1] * timeScale];
    this.random = mulberry32(seed);
    this.grown = pushFromCentroid(polygon, CORRIDOR_SLACK_M);
    const xs = polygon.map((p) => p[0]), ys = polygon.map((p) => p[1]);
    this.box = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    // per particle: trail positions (TRAIL x 3, index 0 = head), trail speeds, age, life, snapshot clock
    this.pos = new Float32Array(count * TRAIL * 3);
    this.speed = new Float32Array(count * TRAIL);
    this.age = new Float32Array(count);
    this.life = new Float32Array(count);
    this.clock = new Float32Array(count);
    this.filled = new Uint8Array(count);      // trail points recorded since birth (<= TRAIL)
    // Leaving the corridor or the band, or entering a building: the particle does not vanish.
    // It keeps going with its last velocity; trail points outside the allowed space are hidden (vis = 0) and the
    // rest of the trail runs out normally; it is reborn once its whole trail has passed out.
    this.vis = new Uint8Array(count * TRAIL);
    this.dying = new Uint8Array(count);
    this.dieT = new Float32Array(count);
    this.vel = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      this.spawn(i);
      this.age[i] = this.random() * this.life[i];     // start at random ages so births are spread in time
    }
  }

  /** Whether a particle may be at (x, y, z): in the grown corridor and band, in the city, not in a building. */
  usable(x, y, z) {
    const [lo, hi] = this.band;
    return z >= lo - BAND_SLACK_M && z <= hi + BAND_SLACK_M
      && x >= 0 && y >= 0 && x <= this.city.nx * CELL_M && y <= this.city.ny * CELL_M
      && insidePolygon(this.grown, x, y) && !solidAt(this.city, x, y, z);
  }

  /**
   * A new random birth point inside the corridor and band, not in a building and not in still air; a point where the
   * wind is faster than slowSpeed is taken with chance sqrt(slowSpeed / its speed) (else another is drawn), so a slow
   * region's short trails get more particles (the square root: the fast inflow keeps enough of them).
   */
  spawn(i) {
    const [x0, x1, y0, y1] = this.box, [lo, hi] = this.band;
    let x = 0, y = 0, z = 0, s = 0;
    for (let tries = 0; tries < BIRTH_TRIES; tries++) {
      x = x0 + this.random() * (x1 - x0); y = y0 + this.random() * (y1 - y0); z = lo + this.random() * (hi - lo);
      const pl = this.plane;
      if (pl?.kind === "h") z = pl.z;
      if (pl?.kind === "v") {
        const along = pl.s0 + this.random() * (pl.s1 - pl.s0);
        x = pl.origin[0] + pl.u[0] * along; y = pl.origin[1] + pl.u[1] * along;
      }
      if (!insidePolygon(this.polygon, x, y) || solidAt(this.city, x, y, z)) continue;
      const w = this.windAt(x, y, z);
      s = Math.hypot(w[0], w[1], w[2]);
      if (s >= MIN_BIRTH_SPEED && (s <= this.slowSpeed || this.random() ** 2 * s < this.slowSpeed)) break;
    }
    const p = this.pos, b = i * TRAIL;
    for (let k = 0; k < TRAIL; k++) {
      p[(b + k) * 3] = x; p[(b + k) * 3 + 1] = y; p[(b + k) * 3 + 2] = z;
      this.speed[b + k] = s;
      this.vis[b + k] = 1;
    }
    this.dying[i] = 0; this.dieT[i] = 0;
    this.age[i] = 0;
    this.life[i] = this.lifeRange[0] + this.random() * (this.lifeRange[1] - this.lifeRange[0]);
    this.clock[i] = 0;
    this.filled[i] = 1;
  }

  /** Advance every particle by `delta` seconds of screen time. */
  step(delta) {
    if (!(delta > 0)) return;
    const h = delta * TIME_LAPSE;
    const p = this.pos, sp = this.speed;
    const vel = this.vel, vis = this.vis;
    for (let i = 0; i < this.count; i++) {
      const b = i * TRAIL, o = b * 3;
      const x = p[o], y = p[o + 1], z = p[o + 2];
      let xNext, yNext, zNext, inside;
      if (!this.dying[i]) {
        this.age[i] += delta;
        if (this.age[i] >= this.life[i]) { this.spawn(i); continue; }   // faded out already (lifeAlpha)
        // midpoint rule; 2-D particles are only born on the plane, every particle moves in the full 3-D wind
        const w = this.windAt(x, y, z);
        const m = this.windAt(x + 0.5 * h * w[0], y + 0.5 * h * w[1], z + 0.5 * h * w[2]);
        vel[i * 3] = m[0]; vel[i * 3 + 1] = m[1]; vel[i * 3 + 2] = m[2];
        xNext = x + h * m[0]; yNext = y + h * m[1]; zNext = z + h * m[2];
        inside = this.usable(xNext, yNext, zNext);
        if (!inside) { this.dying[i] = 1; this.dieT[i] = 0; }
      } else {
        // out of the allowed space: straight on with the last velocity until the whole trail has passed out
        this.dieT[i] += delta;
        if (this.dieT[i] > TRAIL * this.trailDt) { this.spawn(i); continue; }
        xNext = x + h * vel[i * 3]; yNext = y + h * vel[i * 3 + 1]; zNext = z + h * vel[i * 3 + 2];
        inside = false;
      }
      // snapshot: shift the trail back one place every trailDt
      this.clock[i] += delta;
      if (this.clock[i] >= this.trailDt) {
        this.clock[i] -= this.trailDt;
        p.copyWithin(o + 3, o, o + (TRAIL - 1) * 3);
        sp.copyWithin(b + 1, b, b + TRAIL - 1);
        vis.copyWithin(b + 1, b, b + TRAIL - 1);
        if (this.filled[i] < TRAIL) this.filled[i]++;
      }
      p[o] = xNext; p[o + 1] = yNext; p[o + 2] = zNext;
      vis[b] = inside ? 1 : 0;
      if (inside) {
        const v = this.windAt(xNext, yNext, zNext);
        sp[b] = Math.hypot(v[0], v[1], v[2]);
      }
    }
  }

  /** Opacity of particle i from its birth and death fades (0..1). */
  lifeAlpha(i) {
    const a = this.age[i], l = this.life[i];
    return Math.min(1, a / FADE_IN, Math.max(0, (l - a) / FADE_OUT));
  }
}
