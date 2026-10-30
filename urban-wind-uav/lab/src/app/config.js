// Constants of the page: the four planning methods, the wind and display settings, the prediction and planning
// limits, the chart tabs, the city picker and the first mission of every urban block.

/**
 * The four routes, each planned in its own wind, in the order of the paper's Table 1.  With a CFD wind for this city,
 * speed and direction, every route is scored in it (the route planned in the CFD wind itself is a reference, not a
 * proven optimum); without one only the geometry of the routes is shown.  Colours: Ours a blue, Wind-agnostic an
 * orange, Profile wind a rose, and Ground truth a graphite grey that stands out of the cool grey scene.
 */
export const METHODS = [
  { id: "cfd", name: "Ground truth", colour: "#46515d",
    sub: "Planned with the offline high-fidelity CFD wind field" },
  { id: "agnostic", name: "Wind-agnostic", colour: "#ee943b",
    sub: "Planned with zero wind" },
  { id: "profile", name: "Profile wind", colour: "#dd758b",
    sub: "Planned with a height-dependent vertical wind profile (no horizontal structure)" },
  { id: "ours", name: "Ours", colour: "#2989da",
    sub: "Planned with the corridor-localized 3D wind field predicted by GeoWind2Plan in this page" },
];

/** The method of id `id`. */
export const methodById = (id) => METHODS.find((m) => m.id === id);

/** Inflow speed (m/s) the model predicts for: a predicted wind is scaled by speed / REFERENCE_SPEED. */
export const REFERENCE_SPEED = 4.0;
export const SPEED_MIN = 2, SPEED_MAX = 8;          // background wind speed range (m/s), the range of the CFD cases
export const SPEED_STEP = 0.1;                      // a free background speed is a multiple of this (m/s)

// ---------- wind prediction ----------
export const MODEL_DIR = "assets/model2d";         // the model's metadata.json and weights.bin
// Every second flight layer is predicted (the layers between are interpolated) and patches overlap by 0.2: about half
// the prediction time of predicting every layer at overlap 0.4, with the predicted-wind route's energy changed by about
// 0.1 % on average.
export const PREDICT_OVERLAP = 0.2;
const PREDICT_LAYER_STRIDE = 2;
// batchSize: patches per network batch
export const PREDICTOR_OPTIONS = Object.freeze({ batchSize: 16, layerStride: PREDICT_LAYER_STRIDE });
// The prediction sweep shows every third predicted layer as a slice while the wind is predicted.  Its colour range is
// fixed before the prediction: 0 to SWEEP_RANGE_RATIO x the background speed, just above the corridor maximum of the
// predicted wind (typically 1.6-1.8 x the background speed).
export const SWEEP_EVERY = 3;
export const SWEEP_RANGE_RATIO = 1.8;

// ---------- limits: a step that does not finish ends with a failure message instead of an endless wait ----------
export const DEVICE_LIMIT_MS = 30000;          // a new graphics device (the browser's adapter and device requests)
export const PREDICT_LIMIT_MS = 180000;        // a wind prediction; the graphics device is then made anew
export const CITY_GEOMETRY_LIMIT_MS = 120000;  // the whole-city geometry features of a city
export const GPU_WARM_UP_LIMIT_MS = 60000;     // the throw-away prediction that compiles the GPU pipelines
export const PLAN_LIMIT_MS = 120000;           // one route in a planner worker
export const PLANNER_START_LIMIT_MS = 120000;  // the planner workers' energy tables at start-up
export const PARTICLE_LIMIT_MS = 30000;        // a particle request (the particle worker is then replaced)

// ---------- wind display ----------
// Particle density: m^2 of corridor per particle for the full field, m^2 of the plane per particle for particles born
// on a horizontal or vertical plane; between PARTICLE_MIN and PARTICLE_MAX in every mode and city.
export const PARTICLE_M2_FULL = 200;
export const PARTICLE_M2_HORIZONTAL = 200, PARTICLE_M2_VERTICAL = 160;
export const PARTICLE_MIN = 200, PARTICLE_MAX = 5000;
// The Density slider: the particle count above (in every particle mode) times its value; 0: no particles.
export const DENSITY_MIN = 0, DENSITY_MAX = 2, DENSITY_STEP = 0.05;
// Height slider: whole metres, 1 m per step.  The value 30 m stands for 30.75 m, the centre of the lowest predicted
// layer and the lowest height the wind and the planner work at (state.js atLeastBand, heightLabel).
export const HEIGHT_MIN = 30, HEIGHT_MAX = 120, HEIGHT_STEP = 1;

// ---------- results ----------
/** The line charts along the route (tabs in index.html): the unit; `wind`: shown only with Ground truth on. */
export const CHART_TABS = {
  height: { unit: "m" },
  ground: { unit: "m/s" },
  air: { unit: "m/s", wind: true },
};

// ---------- city picker ----------
// Height-map previews at one scale for every city (true relative size), PICKER_PX_PER_M pixels per metre, so the open
// picker fits a 1280 x 720 screen with urban block E (3 km) at full scale.
export const PICKER_PX_PER_M = 0.07;
// Building height colours from low to tall, on the base-plate colour (one height scale for every preview).
export const PICKER_RAMP = [[217, 225, 230], [127, 141, 153]];

// ---------- first missions ----------
// The first mission of each urban block, shown every time the block is opened (Random mission gives others).
// Missions drawn as in the paper's mission sampling with the direction in the headwind class, one random stream per
// block, kept when within 10 degrees of straight headwind and the start-goal line passes within 1/8 of the side from
// the city centre: 48 per block, each planned in the block's CFD wind (4 m/s, 0 degrees).  Chosen: the closest to the
// reference energy saving and recovery: the paper's headwind values at 4 m/s (Table 1 / Table 4); A instead flies from
// its bottom-right corner to its top-left one (the owner's choice among start x 980-1178, y 22-220 m and goal x 22-220,
// y 980-1178 m; 55 tried):
//   A          (1178, 22) to (22, 1178) m,    7.4 % / 83.0 %
//   B #43   4.2 deg,   40 m from the centre, 12.12 % / 81.7 % (paper 12.65 / 84.0)
//   C #41   0.4 deg,   54 m from the centre, 16.01 % / 90.1 % (paper 16.00 / 90.3)
//   D #0    5.3 deg,  139 m from the centre, 12.26 % / 85.3 % (paper 12.80 / 84.0)
//   E #15   2.3 deg,  314 m from the centre, 18.35 % / 82.4 % (paper 17.68 / 84.2)
export const FIRST_MISSIONS = {
  case_20: { start: [1178, 22], goal: [22, 1178] },
  case_0: { start: [1178, 682], goal: [22, 598] },
  case_8: { start: [1178, 658], goal: [22, 650] },
  case_27: { start: [1178, 514], goal: [22, 406] },
  case_302: { start: [2978, 1246], goal: [22, 1126] },
};
