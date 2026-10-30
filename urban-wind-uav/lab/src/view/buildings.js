// The buildings as one mesh (city frame, metres) from their surfaces (city/buildings.js, built in a worker).
import * as THREE from "../../vendor/three.module.js";

/** The buildings' mesh geometry from surfaces { position, normal, sphere } (city/buildings.js buildingSurfaces). */
export function buildingGeometry({ position, normal, sphere }) {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(...sphere.center), sphere.radius);
  return geometry;
}
