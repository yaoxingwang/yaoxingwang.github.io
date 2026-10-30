// A city's height map as space (city/city.js loadCity): the cell under a point, whether a point is inside a
// building, and the cells of a wind-frame grid inside a corridor.  City metres; cells CELL_M square (shared/units.js).
import { LAYER_M, cellCentreM, cellOf } from "../shared/units.js";
import { polygonRuns } from "../shared/geometry.js";

/** The index of the city cell under (x, y) metres (row-major), or -1 outside the city. */
export function cellAt(city, x, y) {
  const ix = cellOf(x), iy = cellOf(y);
  return ix < 0 || iy < 0 || ix >= city.nx || iy >= city.ny ? -1 : iy * city.nx + ix;
}

/** Whether height z (m) is inside the building of city cell `cell` (its roof reaches z). */
const roofReaches = (city, cell, z) => city.layers[cell] * LAYER_M >= z;

/** Whether (x, y, z) metres is inside a building (outside the city: no). */
export function solidAt(city, x, y, z) {
  const cell = cellAt(city, x, y);
  return cell >= 0 && roofReaches(city, cell, z);
}

/** Whether (x, y, z) metres is open air of the city: inside it and not in a building. */
export function openAir(city, x, y, z) {
  const cell = cellAt(city, x, y);
  return cell >= 0 && !roofReaches(city, cell, z);
}

/**
 * Each cell of a wind-frame grid { gx, gy, gridX0, gridY0 } inside the polygon windPoly (wind-frame cells), rows in
 * ascending order: visit(o, x, y) with o the cell's index in the grid (row-major) and (x, y) its centre in city
 * metres; frame (shared/geometry.js windFrame) maps wind-frame cells to city cells (gpu/tiling.js), as its toCity
 * (written out here: no arrays per cell).
 */
export function forCellsInside(windPoly, grid, frame, visit) {
  const { gx, gy, gridX0, gridY0 } = grid;
  const { cX, cY, cos, sin } = frame;
  for (let y = 0; y < gy; y++) {
    const dy = y + gridY0 - cY;
    polygonRuns(windPoly, y + gridY0, gridX0, gridX0 + gx, (x0, x1) => {
      for (let sx = x0; sx < x1; sx++) {
        const dx = sx - cX;
        visit(y * gx + sx - gridX0, cellCentreM(cX + dx * cos - dy * sin), cellCentreM(cY + dx * sin + dy * cos));
      }
    });
  }
}
