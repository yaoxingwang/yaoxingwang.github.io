// Materials, shaders and textures of the scene's data layers: the wind palette, the particle ribbons, the speed slices,
// the route tubes with moving arrow marks, and the buildings.
import * as THREE from "../../vendor/three.module.js";
import { windColour } from "../shared/colormap.js";
import { BUILDING } from "../shared/style.js";
import { TRAIL, POINT_TEXTURE_WIDTH } from "../particles/system.js";

/**
 * The drawing order of the scene's transparent layers (three.js renderOrder): a larger one is drawn later, over the
 * smaller ones.
 */
export const RENDER_ORDER = Object.freeze({ inflowArrows: 1, slice: 2, particles: 3, corridorLine: 4, routeGhost: 6,
                                            planeFrame: 6, pinBead: 7, edgeLabels: 55, pinButton: 100 });
/** Heights (m) of the flat layers on the ground, apart so that they do not flicker through each other. */
export const GROUND_Z = Object.freeze({ plate: -0.3, plateEdge: -0.2, corridorGround: 0.3, shadow: 0.4,
                                        corridorLine: 0.6, pinFoot: 0.7, edgeLabel: 0.5 });

/** The wind colour map as a 256 x 1 texture (shared by every layer, never disposed with one of them). */
export function paletteTexture() {
  const bytes = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const [r, g, b] = windColour(i / 255);
    bytes.set([r, g, b, 255], i * 4);
  }
  const texture = new THREE.DataTexture(bytes, 256, 1, THREE.RGBAFormat);
  texture.userData.shared = true;
  texture.colorSpace = THREE.SRGBColorSpace;
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return texture;
}

// Wind particles (particles/system.js): one screen-space ribbon per particle trail.  Colour from the wind speed stored
// at each trail point (locked to space); opacity from the place along the trail and the birth/death fades.  The trail
// points come from two float textures written by the particle worker (texel j = trail point j: position and speed;
// its opacity) rather than per-vertex attributes, so each step uploads little data; vertex 2 j and 2 j + 1 are the
// two sides of point j.  ribbonAt pushes them out along the mitred screen normal (from the previous and the next
// point), so consecutive segments meet without gaps or overlaps.
export const PARTICLE_VERTEX = `
uniform highp sampler2D points;
uniform highp sampler2D alphas;
attribute float side;
attribute float taper;
uniform vec2 resolution;
uniform float width;
varying float vSpeed;
varying float vAlpha;
varying float vSide;
vec4 ribbonAt(vec3 pc, vec3 pp, vec3 pn, float w) {
  mat4 m = projectionMatrix * modelViewMatrix;
  vec4 c = m * vec4(pc, 1.0);
  vec4 p = m * vec4(pp, 1.0);
  vec4 n = m * vec4(pn, 1.0);
  vec2 sc = c.xy / c.w * resolution, sp = p.xy / p.w * resolution, sn = n.xy / n.w * resolution;
  vec2 d1 = sc - sp, d2 = sn - sc;
  float l1 = length(d1), l2 = length(d2);
  vec2 t1 = l1 > 1e-4 ? d1 / l1 : (l2 > 1e-4 ? d2 / l2 : vec2(1.0, 0.0));
  vec2 t2 = l2 > 1e-4 ? d2 / l2 : t1;
  vec2 tangent = t1 + t2;
  tangent = length(tangent) > 1e-4 ? normalize(tangent) : t1;
  vec2 miter = vec2(-tangent.y, tangent.x);
  float scale = w / max(dot(miter, vec2(-t1.y, t1.x)), 0.5);
  c.xy += miter * side * scale / resolution * c.w;
  return c;
}
ivec2 texel(int j) { return ivec2(j % ${POINT_TEXTURE_WIDTH}, j / ${POINT_TEXTURE_WIDTH}); }
void main() {
  int j = gl_VertexID / 2, k = j % ${TRAIL}, first = j - k;
  vec4 c = texelFetch(points, texel(j), 0);
  // prev = the older neighbour (k + 1), next = the newer one (k - 1), clamped at the trail's ends
  vec3 prev = texelFetch(points, texel(first + min(${TRAIL - 1}, k + 1)), 0).xyz;
  vec3 next = texelFetch(points, texel(first + max(0, k - 1)), 0).xyz;
  vSpeed = c.w;
  vAlpha = texelFetch(alphas, texel(j), 0).r;
  vSide = side;
  gl_Position = ribbonAt(c.xyz, prev, next, width * taper);
}`;
/** The ribbons' colour: the wind palette at the point's speed, its trail opacity, softened at the two sides. */
export const PARTICLE_FRAGMENT = `
uniform sampler2D palette;
uniform vec2 range;
uniform float opacity;
varying float vSpeed;
varying float vAlpha;
varying float vSide;
void main() {
  float t = clamp((vSpeed - range.x) / max(range.y - range.x, 1e-6), 0.0, 1.0);
  float edge = 1.0 - smoothstep(0.6, 1.0, abs(vSide));
  gl_FragColor = vec4(texture2D(palette, vec2(t, 0.5)).rgb, opacity * vAlpha * edge);
  #include <colorspace_fragment>
}`;

// Speed slices: a half-float texture of [speed, mask] per cell over a plane.
const SLICE_VERTEX = `
varying vec2 vUv;
void main() { vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const SLICE_FRAGMENT = `
uniform sampler2D field;
uniform sampler2D palette;
uniform vec2 range;
uniform float opacity;
varying vec2 vUv;
void main() {
  vec4 texel = texture2D(field, vUv);
  // soft mask edge over about 2 px (no jagged edge); the mask is bilinear between 0 and 1
  float fw = max(fwidth(texel.g), 1e-4);
  float m = smoothstep(0.5 - fw, 0.5 + fw, texel.g);
  if (m <= 0.0) discard;
  float t = clamp((texel.r - range.x) / max(range.y - range.x, 1e-6), 0.0, 1.0);
  gl_FragColor = vec4(texture2D(palette, vec2(t, 0.5)).rgb, opacity * m);
  #include <colorspace_fragment>
}`;

/** The material of a speed slice: half-float texels (Uint16Array RGBA) of width x height, colour range, opacity. */
export function sliceMaterial(half, width, height, palette, range, opacity) {
  // half-float: linear filtering of float32 textures needs an extension some GPUs lack (the layer then stays blank
  // without an error)
  const texture = new THREE.DataTexture(half, width, height, THREE.RGBAFormat, THREE.HalfFloatType);
  texture.magFilter = THREE.LinearFilter;
  texture.minFilter = THREE.LinearFilter;
  texture.needsUpdate = true;
  return new THREE.ShaderMaterial({
    vertexShader: SLICE_VERTEX, fragmentShader: SLICE_FRAGMENT,
    uniforms: { field: { value: texture }, palette: { value: palette },
                range: { value: new THREE.Vector2(...range) }, opacity: { value: opacity } },
    // no depth write: routes below the map are still drawn over it; one pass (a slice is flat, all its faces face one
    // way): what three.js's back-then-front passes would draw, without changing the material twice a frame
    transparent: true, depthWrite: false, side: THREE.DoubleSide, forceSinglePass: true,
  });
}

/**
 * The route tube material with moving arrow marks: chevrons travel along the route at the planned ground speed, sped
 * up by the routes' time lapse.  Attribute tAlong = flight time (s) from the start at each vertex; a mark at
 * tAlong = time + k * period moves forward.  The V of each chevron points along the flight.
 */
export function arrowMaterial(params, period, time) {
  const material = new THREE.MeshStandardMaterial(params);
  material.onBeforeCompile = (shader) => {
    shader.uniforms.uTime = time;
    shader.uniforms.uPeriod = { value: period };
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float tAlong;\nvarying float vT;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvT = tAlong;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>",
               "#include <common>\nuniform float uTime;\nuniform float uPeriod;\nvarying float vT;")
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
  // V shape: the middle of the tube as seen (view-space normal facing the camera) leads, its silhouette lags,
  // so the chevron's tip points along the flight whatever the tube's own orientation
  float g = fract((vT - uTime) / uPeriod + 0.3 * acos(clamp(abs(normalize(normal).z), 0.0, 1.0)) / 1.5708);
  float mark = smoothstep(0.0, 0.06, g) * (1.0 - smoothstep(0.26, 0.34, g));
  gl_FragColor.rgb = mix(gl_FragColor.rgb, vec3(1.0), 0.3 * mark);`);
  };
  return material;
}

/**
 * `material` (of a tube with the along attribute: metres along it, view/routes.js) drawn only behind the drone and
 * clear of it: cut = { at, gap, fade, show }, uniforms { value } of the drone's place along the tube (m), the stretch
 * behind it left out (m), the stretch over which the route then fades in (m), and the share (0..1) of the opacity the
 * whole route is shown with (it fades out before a flight).  Its own shader edits, if any, stay; its program is told
 * apart from the uncut one's.
 */
export function cutAhead(material, cut) {
  const own = material.onBeforeCompile;
  material.transparent = true;
  material.onBeforeCompile = (shader, renderer) => {
    own?.(shader, renderer);
    Object.assign(shader.uniforms, { uAt: cut.at, uGap: cut.gap, uFade: cut.fade, uShow: cut.show });
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float along;\nvarying float vS;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvS = along;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", `#include <common>
varying float vS;
uniform float uAt;
uniform float uGap;
uniform float uFade;
uniform float uShow;`)
      .replace("void main() {", "void main() {\n  if (vS > uAt - uGap) discard;")
      .replace("#include <dithering_fragment>",
               `#include <dithering_fragment>
  gl_FragColor.a *= uShow * clamp((uAt - uGap - vS) / uFade, 0.0, 1.0);`);
  };
  material.customProgramCacheKey = () => `cut ${own ?? ""}`;
  return material;
}

/** The buildings' material: white, walls (not roofs) darkening toward the ground (the shader's ramp below). */
export function buildingMaterial() {
  const material = new THREE.MeshStandardMaterial({ color: BUILDING, roughness: 0.9 });
  material.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vHeight;\nvarying float vUp;")
      .replace("#include <project_vertex>",
               "#include <project_vertex>\n  vHeight = transformed.z;\n  vUp = objectNormal.z;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vHeight;\nvarying float vUp;")
      .replace("#include <color_fragment>", `#include <color_fragment>
  // walls only (not roofs) darken toward the ground: 0.85 at 0 m to 1.0 at 30 m (true metres)
  if (vUp < 0.5) diffuseColor.rgb *= mix(0.85, 1.0, smoothstep(0.0, 30.0, vHeight));`);
  };
  return material;
}
