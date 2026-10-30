// The meshes of the scene's wind layers (view/viewer.js puts them in the scene): the particle ribbons, the horizontal
// and vertical speed slices, the particles' birth plane and the inflow arrows.  City metres, z up.
import * as THREE from "../../vendor/three.module.js";
import { CELL_M, cellCentreM } from "../shared/units.js";
import { verticalPlaneCorners, lineInBox } from "../shared/geometry.js";
import { SLICE_OPACITY, PLANE_OPACITY, PARTICLE_OPACITY, ARROW_COLOUR, INFLOW_ARROW } from "../shared/style.js";
import { POINT_TEXTURE_WIDTH } from "../particles/system.js";
import { PARTICLE_VERTEX, PARTICLE_FRAGMENT, sliceMaterial, RENDER_ORDER } from "./materials.js";
import { setSliceTexel } from "./half_float.js";

/** The inflow arrows' size (m, the same in every city) and their gap to the city's edge. */
export const ARROW_LENGTH_M = 84, ARROW_GAP_M = 42;
// A fixed display lift, not a sampling height or a wind-speed encoding.
export const ARROW_HEIGHT_M = 52;
/** A particle ribbon's head width (CSS pixels, before the viewer's zoom factor); the tail's: particles/ribbons.js. */
export const PARTICLE_WIDTH_PX = 1.2;

/** [speed, mask] per cell (Float32Array; mask 1, or 0 with speed 0) as the half-float RGBA texels of a slice. */
export function sliceTexels(data, cells) {
  const texels = new Uint16Array(cells * 4);
  for (let i = 0; i < cells; i++) if (data[i * 2 + 1]) setSliceTexel(texels, i, data[i * 2]);
  return texels;
}

/** A quad of four corners [x, y, z] (in order around it) as two triangles. */
function quadGeometry(corners) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(corners.flat()), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  return geometry;
}

/** A three.js colour from sRGB bytes [r, g, b]. */
const srgbColour = (rgb) => new THREE.Color().setRGB(rgb[0] / 255, rgb[1] / 255, rgb[2] / 255, THREE.SRGBColorSpace);

/**
 * The particle ribbons of `system` (viewer.setParticles): one ribbon of TRAIL points per particle whose points are
 * read from two float textures (points: position and speed; alphas: opacity), coloured by `palette` over `range`;
 * system.ribbons: its vertices' sides and tapers and its index (particles/ribbons.js, from the worker).
 * resolution: the drawing buffer's half size; pixelRatio: the renderer's.  PARTICLE_OPACITY; widths PARTICLE_WIDTH_PX
 * at the head, narrower towards the tail (ribbons.js).
 */
export function particleMesh(system, range, palette, resolution, pixelRatio) {
  const { side, taper, index } = system.ribbons;
  const geometry = new THREE.BufferGeometry();
  const rows = system.arrays[1].length / POINT_TEXTURE_WIDTH;
  const pointTexture = (data, format) => {
    const t = new THREE.DataTexture(data, POINT_TEXTURE_WIDTH, rows, format, THREE.FloatType);
    t.magFilter = THREE.NearestFilter;
    t.minFilter = THREE.NearestFilter;
    t.generateMipmaps = false;
    t.needsUpdate = true;
    return t;
  };
  const pointsTexture = pointTexture(system.arrays[0], THREE.RGBAFormat);
  const alphaTexture = pointTexture(system.arrays[1], THREE.RedFormat);
  // no position attribute: the shader reads each vertex's point from the textures by its number (the draw is indexed)
  geometry.setAttribute("side", new THREE.BufferAttribute(side, 1));
  geometry.setAttribute("taper", new THREE.BufferAttribute(taper, 1));
  geometry.setIndex(new THREE.BufferAttribute(index, 1));
  const material = new THREE.ShaderMaterial({
    vertexShader: PARTICLE_VERTEX, fragmentShader: PARTICLE_FRAGMENT,
    uniforms: { points: { value: pointsTexture }, alphas: { value: alphaTexture }, palette: { value: palette },
                range: { value: new THREE.Vector2(...range) }, opacity: { value: PARTICLE_OPACITY },
                resolution: { value: resolution }, width: { value: PARTICLE_WIDTH_PX * pixelRatio } },
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;
  mesh.renderOrder = RENDER_ORDER.particles;
  return mesh;
}

/**
 * The particles' birth plane: a transparent plane in the inflow arrows' colour.  points = [[x, y, z], ...] in metres:
 * a horizontal polygon (one z) or the vertical plane's four corners.
 */
export function planeFrameMesh(points) {
  let geometry;
  if (points.every((p) => p[2] === points[0][2])) {
    geometry = new THREE.ShapeGeometry(new THREE.Shape(points.map(([x, y]) => new THREE.Vector2(x, y))));
    geometry.translate(0, 0, points[0][2]);
  } else {
    geometry = quadGeometry(points);
  }
  // one pass, as a slice (materials.js sliceMaterial): the plane is flat
  const mesh = new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
    color: srgbColour(ARROW_COLOUR),
    transparent: true, opacity: PLANE_OPACITY, depthWrite: false, side: THREE.DoubleSide, forceSinglePass: true }));
  mesh.renderOrder = RENDER_ORDER.planeFrame;
  mesh.frustumCulled = false;
  return mesh;
}

/**
 * A horizontal speed slice on the wind-frame grid of `layout` (gpu/tiling.js: grid origin and size gx by gy cells,
 * theta, toCity), at `height` metres, from its half-float RGBA texels (sliceTexels, or the sweep worker's).
 */
export function sliceGroup(texels, layout, height, range, palette) {
  const { gx, gy } = layout;
  const mesh = new THREE.Mesh(new THREE.PlaneGeometry(gx * CELL_M, gy * CELL_M),
                              sliceMaterial(texels, gx, gy, palette, range, SLICE_OPACITY));
  const [ix, iy] = layout.toCity([layout.gridX0 + gx / 2 - 0.5, layout.gridY0 + gy / 2 - 0.5]);   // its centre
  mesh.position.set(cellCentreM(ix), cellCentreM(iy), height);
  mesh.rotation.z = layout.theta;
  mesh.renderOrder = RENDER_ORDER.slice;
  const group = new THREE.Group();
  group.add(mesh);
  return group;
}

/**
 * A vertical slice: data = Float32Array of [speed, mask] per cell, ns along the inflow direction by nz up; the plane
 * { origin, u, s0, s1, z0, z1 } runs from origin + u s0 to origin + u s1 and from z0 to z1 (metres).
 */
export function verticalSliceGroup(data, ns, nz, { origin, u, s0, s1, z0, z1 }, range, palette) {
  const material = sliceMaterial(sliceTexels(data, ns * nz), ns, nz, palette, range, SLICE_OPACITY);
  const geometry = quadGeometry(verticalPlaneCorners(origin, u, s0, s1, z0, z1));
  geometry.setAttribute("uv", new THREE.BufferAttribute(new Float32Array([0, 0, 1, 0, 1, 1, 0, 1]), 2));
  const mesh = new THREE.Mesh(geometry, material);
  mesh.renderOrder = RENDER_ORDER.slice;
  mesh.frustumCulled = false;
  const group = new THREE.Group();
  group.add(mesh);
  return group;
}

/**
 * Inflow arrows of a W x H m city: lit, slender arrows just outside its upwind side, pointing along the background
 * wind. They are spread evenly across the city's width seen by the wind,
 * each placed where its line enters the city: one row along an edge for wind square to the city, around the corner
 * for oblique wind.  Count 3-6 from that width.
 */
export function inflowArrowGroup(W, H, theta) {
  const side = Math.max(W, H);
  const u = [Math.cos(theta), Math.sin(theta)], n = [-u[1], u[0]], c = [W / 2, H / 2];
  const across = [[0, 0], [W, 0], [W, H], [0, H]].map(([x, y]) => (x - c[0]) * n[0] + (y - c[1]) * n[1]);
  const nmin = Math.min(...across), nmax = Math.max(...across), span = nmax - nmin;
  const count = Math.min(6, Math.max(3, Math.round(span / (0.28 * side))));
  // arrow size in true metres, the same in every city (arrows scaled with the city would hide the large city's
  // size); their spacing follows the city
  const L = ARROW_LENGTH_M, gap = ARROW_GAP_M;
  const geometry = new THREE.LatheGeometry([
    new THREE.Vector2(0, 0), new THREE.Vector2(3.2, 0),
    new THREE.Vector2(3.2, L - 24), new THREE.Vector2(11, L - 24), new THREE.Vector2(0, L),
  ], 24);
  geometry.rotateZ(-Math.PI / 2);
  const material = new THREE.MeshStandardMaterial({ color: INFLOW_ARROW, roughness: 0.72, metalness: 0 });
  const group = new THREE.Group();
  for (let i = 0; i < count; i++) {
    const t = nmin + (i + 0.5) * span / count;
    const o = [c[0] + n[0] * t, c[1] + n[1] * t];
    // where the line o + s u enters the city rectangle
    const [s0, s1] = lineInBox(o, u, [0, 0], [W, H], 1e-9);
    if (!(s1 > s0)) continue;
    const tip = [o[0] + u[0] * (s0 - gap), o[1] + u[1] * (s0 - gap)];
    const arrow = new THREE.Mesh(geometry, material);
    arrow.position.set(tip[0] - u[0] * L, tip[1] - u[1] * L, ARROW_HEIGHT_M);
    arrow.rotation.z = theta;
    arrow.renderOrder = RENDER_ORDER.inflowArrows;
    group.add(arrow);
  }
  return group;
}
