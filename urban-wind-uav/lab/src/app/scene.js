// The page's 3-D view (view/viewer.js) and the mission corridor in the view's cell units.
import { cellCentreM, cellIndexOf } from "../shared/units.js";
import { clipToBox } from "../shared/geometry.js";
import { corridorShape } from "../planner/mission.js";
import { Viewer } from "../view/viewer.js";
import { state } from "./state.js";
import { $ } from "./dom.js";

/** The 3-D view of the city, made by createViewer when the page starts. */
export let viewer = null;
/** Make the page's 3-D view (once, when the page starts). */
export function createViewer() {
  viewer = new Viewer($("scene"));
}

/** A point in metres as fractional cell indices (cell i is centred at (i + 0.5) CELL_M). */
export const toCells = ([x, y]) => [cellIndexOf(x), cellIndexOf(y)];

/** The mission's corridor (planner/mission.js corridorShape) in cell units, not clipped. */
export const corridorCells = (mission) => corridorShape(mission.start, mission.goal).polygon.map((p) => toCells(p));

/** The mission corridor (cell units) clipped to the city, for everything drawn. */
export function shownCorridor(mission) {
  const { nx, ny } = state.city;
  return clipToBox(corridorCells(mission), -0.5, nx - 0.5, -0.5, ny - 0.5);
}

/** The shown corridor in metres (its cell centres' positions). */
export const shownCorridorMetres = (mission) =>
  shownCorridor(mission).map(([ix, iy]) => [cellCentreM(ix), cellCentreM(iy)]);
