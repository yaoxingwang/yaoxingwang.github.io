// The multirotor power model of GeoWind2Plan (the paper's corrected energy model), z up.
//
// Thrust T = m a + m g z_hat - D, drag D = -0.5 rho Cd A_f |v_a| v_a, with v_a the airspeed (ground velocity minus
// wind).  Power = T . v_a (useful) + |T| v_i (induced, momentum theory with climb) + P_hover (|T| / m g)^1.5.

/**
 * The vehicle: mass (kg), air density (kg/m^3), drag coefficient, frame and rotor areas (m^2), hover profile power
 * (W), g (m/s^2), and its limits.
 */
export const VEHICLE = Object.freeze({
  mass: 5.0, rho: 1.18, cd: 1.0, frameArea: 0.16, rotorArea: 0.5,
  profilePowerHover: 100.0, g: 9.81,
  maxAirspeed: 20.0, maxThrust: 100.0, maxAcceleration: 3.0,
});

const { mass: M, rho: RHO, cd: CD, frameArea: AF, rotorArea: AR, profilePowerHover: PP, g: G } = VEHICLE;
/** The drag factor 0.5 rho Cd A_f (|D| = DRAG |v_a|^2); EPS: added under the model's square roots and to a divisor. */
export const DRAG = 0.5 * RHO * CD * AF;
export const EPS = 1e-9;

/** Power (W) for airspeed (ax, ay, az) and acceleration (bx, by, bz). */
export function power(ax, ay, az, bx = 0, by = 0, bz = 0) {
  const speed = Math.sqrt(ax * ax + ay * ay + az * az + EPS);
  const tx = M * bx + DRAG * speed * ax;
  const ty = M * by + DRAG * speed * ay;
  const tz = M * bz + M * G + DRAG * speed * az;
  const thrust = Math.sqrt(tx * tx + ty * ty + tz * tz + EPS);
  const climb = (ax * tx + ay * ty + az * tz) / (thrust + EPS);
  const induced = -0.5 * climb + Math.sqrt(0.25 * climb * climb + thrust / (2 * RHO * AR) + EPS);
  return ax * tx + ay * ty + az * tz + thrust * induced
    + PP * Math.pow(thrust / (M * G), 1.5);
}

/** The thrust vector (N) [tx, ty, tz] at airspeed (ax, ay, az) and acceleration (bx, by, bz), as power() has it. */
export function thrust(ax, ay, az, bx, by, bz) {
  const speed = Math.sqrt(ax * ax + ay * ay + az * az + EPS);
  return [M * bx + DRAG * speed * ax, M * by + DRAG * speed * ay, M * bz + M * G + DRAG * speed * az];
}

/**
 * The paper's evaluation model of a route given by points [x, y, z] (m), ground speeds and ground velocities per
 * point (the planner's): the time steps dt (segment length over mean speed) and each point's acceleration [ax, ay, az]
 * (the first derivative of velocity; zero for a route of two points).
 */
export function pointAccelerations(points, speeds, vel) {
  const n = points.length;
  const dt = [];
  for (let i = 0; i < n - 1; i++) {
    const ds = Math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1],
                          points[i + 1][2] - points[i][2]);
    const v = 0.5 * (speeds[i] + speeds[i + 1]);
    dt.push(ds / Math.max(v, 1e-6));
  }
  const acc = [];
  for (let i = 0; i < n; i++) {
    const a = [0, 0, 0];
    if (n > 2) {
      const i0 = Math.min(Math.max(i, 1), n - 2);
      const d1 = dt[i0 - 1], d2 = dt[i0];
      for (let c = 0; c < 3; c++) a[c] = (vel[i0 + 1][c] - vel[i0 - 1][c]) / (d1 + d2);
    }
    acc.push(a);
  }
  return { dt, acc };
}

/**
 * Energy (J) of a route given by points [x, y, z] (m), ground speeds and ground velocities per point (the planner's),
 * in a wind sampler windAt(x, y, z) -> [wx, wy, wz], in the paper's evaluation model (pointAccelerations).
 */
export function trajectoryEnergy(points, speeds, windAt, vel) {
  const n = points.length;
  if (n < 2) return 0;
  const { dt, acc } = pointAccelerations(points, speeds, vel);
  const p = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const w = windAt(points[i][0], points[i][1], points[i][2]);
    const a = acc[i];
    p[i] = power(vel[i][0] - w[0], vel[i][1] - w[1], vel[i][2] - w[2], a[0], a[1], a[2]);
  }
  let energy = 0;
  for (let i = 0; i < n - 1; i++) energy += 0.5 * (p[i] + p[i + 1]) * dt[i];
  return energy;
}
