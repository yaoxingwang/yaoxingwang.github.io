/**
 * Independent, bounded geometry check; no planner/clearance helper imports.
 *
 * validateRoute(points, city, { stepM = 0.25, minClearanceM = 10,
 *   mission, endpointToleranceM = 1e-6 } = {})
 *
 * points: nonempty array of [x,y,z] metre triples (numeric typed triples also
 * accepted). city: {nx,ny,layers}, row-major nonnegative integer layer counts.
 * Cell (ix,iy) is a CLOSED solid AABB [4ix,4(ix+1)] x [4iy,4(iy+1)]
 * x [0,1.5*layers[iy*nx+ix]]; zero layers means no building. These are the
 * physical cell footprints/roofs, NOT upstream's voxel-centre distance model.
 * Domain: 0 <= x < 4nx, 0 <= y < 4ny; z in [30.75,120.75] inclusive,
 * matching the centres of upstream flight layers 20..80. Optional mission
 * {start,goal} uses the same triples and domain/band; endpoints must match
 * within endpointToleranceM (Euclidean distance). Unknown options are rejected.
 *
 * Every straight segment is divided into ceil(length/stepM) intervals, including
 * BOTH endpoints even for very short segments. Singleton routes are checked.
 * Samples do not certify unsampled interiors, rendered splines, flight dynamics,
 * wind, terrain, other obstacles, or real-world operational safety.
 *
 * Each sample examines cells in a conservative horizontal clearance-radius
 * neighborhood. Any omitted column has horizontal distance >= minClearanceM. Thus
 * minClearanceM in the result is min(actual sampled minimum, requested radius):
 * 'exact-sampled' below the radius, otherwise 'lower-bound-sampled' (even if
 * there are no buildings). It is NOT an exact nearest-building distance outside
 * the cutoff, nor a continuous-route minimum. Equality at the threshold passes;
 * no clearance epsilon is subtracted. Floating-point arithmetic still applies.
 *
 * Result: {pass,minClearanceM,samples,violations,metadata}. Invalid inputs or
 * resource limits fail closed, with minClearanceM:null and complete:false.
 * violations contains at most 100 records; metadata.violationCount is untruncated.
 * Limits bound validation work/memory; exceeding them is failure, not a pass.
 * No input mutation, IO, global caches, or upstream validation calls.
 */
const CELL_M = 4;
const LAYER_M = 1.5;
const BAND = [30.75, 120.75];
const LIMITS = Object.freeze({
  cityCells: 16_000_000, routePoints: 1_000_000, samples: 1_000_000,
  cellChecks: 50_000_000, reportedViolations: 100,
});
const OPTION_KEYS = new Set(['stepM', 'minClearanceM', 'mission', 'endpointToleranceM']);
const isRecord = (v) => v !== null && typeof v === 'object' && !Array.isArray(v) && !ArrayBuffer.isView(v);
const isVector = (v) => (Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView))) &&
  v.length === 3 && [v[0], v[1], v[2]].every(Number.isFinite);
const isLayers = (v) => Array.isArray(v) || (ArrayBuffer.isView(v) && !(v instanceof DataView));

export function validateRoute(points, city, options = {}) {
  const result = {
    pass: false, minClearanceM: null, samples: 0, violations: [],
    metadata: {
      method: 'sampled-point-to-building-column-aabb', complete: false,
      continuousProof: false, flightDynamicsCertified: false,
      minClearanceKind: 'unavailable', cellM: CELL_M, layerM: LAYER_M,
      flightBandM: [...BAND], domainXY: 'half-open', buildingBounds: 'closed',
      limits: { ...LIMITS }, violationCount: 0, violationsTruncated: false,
      cellChecks: 0,
    },
  };
  const meta = result.metadata;
  const violate = (code, message, detail = {}) => {
    meta.violationCount++;
    if (result.violations.length < LIMITS.reportedViolations) {
      result.violations.push({ code, message, ...detail });
    } else meta.violationsTruncated = true;
  };
  const reject = (code, message, detail) => {
    violate(code, message, detail);
    return result;
  };
  if (!isRecord(options)) return reject('INVALID_OPTIONS', 'options must be an object');
  for (const key of Object.keys(options)) {
    if (!OPTION_KEYS.has(key)) return reject('INVALID_OPTIONS', `Unknown option: ${key}`);
  }
  const stepM = options.stepM === undefined ? 0.25 : options.stepM;
  const radius = options.minClearanceM === undefined ? 10 : options.minClearanceM;
  const tolerance = options.endpointToleranceM === undefined ? 1e-6 : options.endpointToleranceM;
  if (!Number.isFinite(stepM) || stepM <= 0 || !Number.isFinite(radius) || radius <= 0 ||
      !Number.isFinite(tolerance) || tolerance < 0) {
    return reject('INVALID_OPTIONS', 'stepM and minClearanceM must be finite and positive; endpointToleranceM finite and nonnegative');
  }
  Object.assign(meta, { stepM, requiredClearanceM: radius, searchRadiusM: radius, endpointToleranceM: tolerance });
  if (!isRecord(city) || !Number.isSafeInteger(city.nx) || city.nx <= 0 ||
      !Number.isSafeInteger(city.ny) || city.ny <= 0) {
    return reject('INVALID_CITY', 'city must have positive safe-integer nx and ny');
  }
  const cells = city.nx * city.ny;
  if (!Number.isSafeInteger(cells) || cells > LIMITS.cityCells) {
    return reject('RESOURCE_LIMIT', 'City exceeds the cell validation budget');
  }
  if (!isLayers(city.layers) || city.layers.length !== cells) {
    return reject('INVALID_LAYERS', 'layers must be a numeric array or typed array of exactly nx*ny counts');
  }
  let buildingCount = 0;
  for (let i = 0; i < cells; i++) {
    const n = city.layers[i];
    if (!Number.isSafeInteger(n) || n < 0 || !Number.isFinite(n * LAYER_M)) {
      return reject('INVALID_LAYERS', 'Every layer count must be a nonnegative safe integer', { cellIndex: i });
    }
    if (n > 0) buildingCount++;
  }
  meta.buildingColumns = buildingCount;
  const width = city.nx * CELL_M, height = city.ny * CELL_M;
  meta.domainM = { x: [0, width], y: [0, height] };
  const checkPosition = (p, detail) => {
    if (p[0] < 0 || p[0] >= width || p[1] < 0 || p[1] >= height) {
      violate('OUT_OF_DOMAIN', 'Position outside the half-open city XY domain', detail);
    }
    if (p[2] < BAND[0] || p[2] > BAND[1]) {
      violate('OUT_OF_BAND', 'Position outside the inclusive flight band', detail);
    }
  };
  if (!Array.isArray(points) || points.length === 0) {
    return reject('INVALID_ROUTE', 'points must be a nonempty array of metre triples');
  }
  if (points.length > LIMITS.routePoints) return reject('RESOURCE_LIMIT', 'Too many route points');
  for (let i = 0; i < points.length; i++) {
    if (!isVector(points[i])) return reject('INVALID_POINT', 'Each route point must be a finite numeric triple', { pointIndex: i });
    checkPosition(points[i], { pointIndex: i });
  }
  if (options.mission !== undefined) {
    const m = options.mission;
    if (!isRecord(m) || !isVector(m.start) || !isVector(m.goal)) {
      return reject('INVALID_MISSION', 'mission must contain finite start and goal triples');
    }
    for (const [name, p, actual] of [['start', m.start, points[0]], ['goal', m.goal, points[points.length - 1]]]) {
      checkPosition(p, { endpoint: name });
      const errorM = Math.hypot(p[0] - actual[0], p[1] - actual[1], p[2] - actual[2]);
      if (errorM > tolerance) violate('ENDPOINT_MISMATCH', `Route ${name} does not match mission`, { endpoint: name, errorM });
    }
  }
  if (meta.violationCount) return result;

  // Preflight the sample count before interpolation (tiny steps cannot hang us).
  let plannedSamples = 1, maxSampleSpacingM = 0;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    const length = Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
    const intervals = Math.max(1, Math.ceil(length / stepM));
    plannedSamples += intervals;
    if (!Number.isSafeInteger(plannedSamples) || plannedSamples > LIMITS.samples) {
      return reject('RESOURCE_LIMIT', 'Route exceeds the dense-sample budget');
    }
    maxSampleSpacingM = Math.max(maxSampleSpacingM, length / intervals);
  }
  Object.assign(meta, { plannedSamples, maxSampleSpacingM });
  let minimum = radius;
  const sample = (p, segmentIndex, t) => {
    if (buildingCount === 0) { result.samples++; return true; }
    const [x, y, z] = p;
    // Include touching footprint cells, not just cells whose CENTRES are nearby.
    // One extra lower cell also protects exact grid-boundary contacts.
    const x0 = Math.max(0, Math.floor((x - radius) / CELL_M) - 1);
    const x1 = Math.min(city.nx - 1, Math.floor((x + radius) / CELL_M));
    const y0 = Math.max(0, Math.floor((y - radius) / CELL_M) - 1);
    const y1 = Math.min(city.ny - 1, Math.floor((y + radius) / CELL_M));
    const checks = (x1 - x0 + 1) * (y1 - y0 + 1);
    if (meta.cellChecks + checks > LIMITS.cellChecks) {
      meta.incompleteReason = 'RESOURCE_LIMIT';
      violate('RESOURCE_LIMIT', 'Spatial-neighborhood cell-check budget exceeded');
      return false;
    }
    meta.cellChecks += checks;
    let nearest = radius, nearestCell = null;
    for (let iy = y0; iy <= y1; iy++) {
      const dy = Math.max(iy * CELL_M - y, 0, y - (iy + 1) * CELL_M);
      for (let ix = x0; ix <= x1; ix++) {
        const layers = city.layers[iy * city.nx + ix];
        if (layers === 0) continue;
        const dx = Math.max(ix * CELL_M - x, 0, x - (ix + 1) * CELL_M);
        const dz = Math.max(-z, 0, z - layers * LAYER_M);
        const distance = Math.hypot(dx, dy, dz);
        if (distance < nearest) { nearest = distance; nearestCell = [ix, iy]; }
      }
    }
    const sampleIndex = result.samples++;
    minimum = Math.min(minimum, nearest);
    if (nearest < radius) {
      violate('CLEARANCE', 'Sample is too close to a solid building column', {
        sampleIndex, segmentIndex, t, point: [...p], clearanceM: nearest, cell: nearestCell,
      });
    }
    return true;
  };
  if (!sample(points[0], 0, 0)) return result;
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    const intervals = Math.max(1, Math.ceil(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) / stepM));
    for (let j = 1; j <= intervals; j++) {
      const t = j / intervals;
      // Use the supplied endpoint verbatim, avoiding interpolation roundoff.
      const p = j === intervals ? b : [
        a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t,
      ];
      if (!sample(p, i - 1, t)) return result;
    }
  }
  result.minClearanceM = minimum;
  meta.minClearanceKind = minimum < radius ? 'exact-sampled' : 'lower-bound-sampled';
  meta.complete = true;
  result.pass = meta.violationCount === 0;
  return result;
}
