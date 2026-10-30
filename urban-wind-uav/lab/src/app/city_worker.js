// A module worker of the page that holds the current city (sent once per city as { city: { nx, ny, layers } }) and
// is replaced after a failure: the sweep worker (sweep.js) and the particle worker (particle_feed.js).

/**
 * After a failure a worker restarts on its own (sweep.js: the next slice; particle_feed.js: the particles come back)
 * at most this many times in a row (an answer resets the count); the particle worker also starts anew for a request
 * the user makes (the sweep then stays blank: its results do not depend on it).
 */
const MAX_RESTARTS = 2;

/** One such worker: `url` its module, wire(worker) sets its message and error handlers when it starts. */
export class CityWorker {
  constructor(url, wire) {
    this.url = url;
    this.wire = wire;
    this.worker = null;
    this.city = null;              // the city the worker holds
    this.failures = 0;             // failures since the last answer
    this.hasAnswered = false;      // the current worker has answered (it loaded and runs)
  }

  /** The worker holding `city`: started if there is none, given the city if it holds another one. */
  ensure(city) {
    if (!this.worker) {
      this.hasAnswered = false;
      this.worker = new Worker(this.url, { type: "module" });
      this.wire(this.worker);
    }
    if (this.city !== city) {
      this.worker.postMessage({ city: { nx: city.nx, ny: city.ny, layers: city.layers } });
      this.city = city;
    }
    return this.worker;
  }

  /** Whether a new worker may start on its own after the failures so far. */
  get mayRestart() { return this.failures <= MAX_RESTARTS; }

  /** It answered (any reply): the failures in a row start again from 0. */
  answered() {
    this.failures = 0;
    this.hasAnswered = true;
  }

  /** It failed or stopped answering: ended (the next ensure starts a new one). */
  endAfterFailure() {
    this.worker?.terminate();
    this.worker = null;
    this.city = null;
    this.failures += 1;
  }

  /** The city it holds is let go with its other data (a new city): the next ensure sends the city again. */
  forgetCity() { this.city = null; }
}
