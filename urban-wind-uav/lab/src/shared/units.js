// Grid units of the city data, shared by every part of the demo.  City frame: metres, x along the height map's
// columns, y along its rows, z up.  Cell (ix, iy) spans [4 ix, 4 ix + 4) x [4 iy, 4 iy + 4) with its centre at
// (i + 0.5) x 4 m; layer l spans [1.5 l, 1.5 l + 1.5) with its centre at (l + 0.5) x 1.5 m.

// the horizontal cell size and the layer height (m)
export const CELL_M = 4;
export const LAYER_M = 1.5;
/** The position (m) of cell index i along x or y (i fractional: a point between cell centres). */
export const cellCentreM = (i) => (i + 0.5) * CELL_M;
/** The fractional cell index of position m (m) along x or y: cellCentreM's inverse. */
export const cellIndexOf = (m) => m / CELL_M - 0.5;
/** The cell that position m (m) along x or y lies in. */
export const cellOf = (m) => Math.floor(m / CELL_M);
/** A length of m metres as kilometres with one decimal (text, without the unit). */
export const kmText = (m) => (m / 1000).toFixed(1);
/** The height (m) of layer l's centre. */
export const layerCentreM = (l) => (l + 0.5) * LAYER_M;
/** The fractional layer index of height z (m): layerCentreM's inverse. */
export const layerIndexOf = (z) => z / LAYER_M - 0.5;

// The flight band: layers 20 to 80, centres 30.75 m to 120.75 m (the paper's 30-120 m band).  Wind is predicted and
// routes are planned in it.
export const FIRST_FLIGHT_LAYER = 20;
export const LAST_FLIGHT_LAYER = 80;
export const FLIGHT_LAYERS = LAST_FLIGHT_LAYER - FIRST_FLIGHT_LAYER + 1;
export const BAND_LOW_M = layerCentreM(FIRST_FLIGHT_LAYER);
export const BAND_HIGH_M = layerCentreM(LAST_FLIGHT_LAYER);

// The planner's wind grid (planner/wind_grid.js), shared with the predicted region it needs (gpu/tiling.js): a sample
// every WIND_GRID_STEP cells, over the corridor's box reaching WIND_BOX_ACROSS_M across beyond its widest half width.
export const WIND_GRID_STEP = 2;
export const WIND_BOX_ACROSS_M = 32;

/** Values per cell of a predicted wind field and of its weighted sums: wind-frame u, w, v and the blend weight. */
export const CELL_VALUES = 4;
