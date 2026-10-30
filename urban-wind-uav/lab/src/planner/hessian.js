// Hessian of the trajectory optimizer (trajopt.js) in the accelerations, assembled from its terms in structured form.
// Every objective term at node k (or at a segment point between nodes k and k + 1) adds, per (c, d),
//   H[3j+c][3l+d] += Bxx X[j] X[l] + Bvx V[j] X[l] + Bxv X[j] V[l] + Bvv V[j] V[l]
// with X = dt^2 (lambda^T Mm), V = dt (lambda^T Nm) (lambda: the term's row of the node, or the segment's mix of rows
// k and k + 1): n^2 per term, about 27 n^3 per Newton step if added term by term.  Mm = Y - c e^T and
// Nm = U - 1 f^T (kinematicMaps: Y = y, U = u, c_k = k / (n - 1), e = y[n - 1], f = e / (n - 1)), and U = Sigma A,
// Y = Sigma A Sigma A with Sigma the lower-triangular matrix of ones and A the averaging matrix
// (A[m][m] = A[m][m - 1] = 1/2 for m >= 1, row 0 zero).  So the sum over all terms is
//   Y^T T Y = A^T S(A^T S(T) A) A,  U^T D Y = (A^T S(D) A) Sigma A,  U^T D U = A^T S(D) A
// (T: tridiagonal sum of Bxx lambda lambda^T, D: diagonal; S(M)[p][q] = sum over r >= p, s >= q of M[r][s]) plus
// rank-one parts from c, e, f: O(n^2) per (c, d).  The same matrix as adding term by term, up to rounding.

const BLOCK_ACCUMULATORS = ["Pd", "Po", "aP", "Qd", "qc", "Rd", "rc", "Sd", "Bxa", "Bva", "Bax", "Bav", "Baa"];

/** Accumulators of the dense part for up to n nodes, per (c, d), and the work arrays of structAdd. */
export function structBuffers(n) {
  const perBlock = () => Array.from({ length: 9 }, () => new Float64Array(n));
  const vector = () => new Float64Array(n);
  return { n, ...Object.fromEntries(BLOCK_ACCUMULATORS.map((name) => [name, perBlock()])),
           sP: new Float64Array(9), sQ: new Float64Array(9), sR: new Float64Array(9), sS: new Float64Array(9),
           G: new Float64Array(n * n), W: new Float64Array(n * n),
           f: vector(), Le: vector(), Lf: vector(), R1: vector(), R2: vector(),
           t1: vector(), t2: vector(), t3: vector(), t4: vector() };
}

/** Clear the accumulators for a new Newton step. */
export function structReset(acc, n) {
  for (const key of BLOCK_ACCUMULATORS) for (let i = 0; i < 9; i++) acc[key][i].fill(0, 0, n);
  acc.sP.fill(0); acc.sQ.fill(0); acc.sR.fill(0); acc.sS.fill(0);
}

/** Record one term's Hy (its dense part, summed per node in O(1)). */
export function structRecord(acc, n, k, isMix, al, Hy) {
  const ck = k / (n - 1);
  for (let c = 0; c < 3; c++) for (let d = 0; d < 3; d++) {
    const i = c * 3 + d, B = Hy[c * 9 + d];
    if (isMix) {
      const g = (1 - al) * ck + al * (k + 1) / (n - 1);
      acc.Pd[i][k] += B * (1 - al) * (1 - al); acc.Pd[i][k + 1] += B * al * al; acc.Po[i][k] += B * al * (1 - al);
      acc.aP[i][k] += B * g * (1 - al); acc.aP[i][k + 1] += B * g * al; acc.sP[i] += B * g * g;
      continue;
    }
    const Q = Hy[(3 + c) * 9 + d], R = Hy[c * 9 + 3 + d], S = Hy[(3 + c) * 9 + 3 + d];
    acc.Pd[i][k] += B; acc.aP[i][k] += B * ck; acc.sP[i] += B * ck * ck;
    acc.Qd[i][k] += Q; acc.qc[i][k] += Q * ck; acc.sQ[i] += Q * ck;
    acc.Rd[i][k] += R; acc.rc[i][k] += R * ck; acc.sR[i] += R * ck;
    acc.Sd[i][k] += S; acc.sS[i] += S;
    // rows / columns of a_k: block (c, d) [j][k] and [k][j], and [k][k]
    acc.Bxa[i][k] += Hy[c * 9 + 6 + d]; acc.Bva[i][k] += Hy[(3 + c) * 9 + 6 + d];
    acc.Bax[i][k] += Hy[(6 + c) * 9 + d];
    acc.Bav[i][k] += Hy[(6 + c) * 9 + 3 + d];
    acc.Baa[i][k] += Hy[(6 + c) * 9 + 6 + d];
  }
}

/** M (n x n, row-major) replaced by its suffix sums in both indices: M[p][q] = sum over i >= p, j >= q. */
function suffix2(M, n) {
  for (let p = n - 1; p >= 0; p--) for (let q = n - 1; q >= 0; q--) {
    let v = M[p * n + q];
    if (p + 1 < n) v += M[(p + 1) * n + q];
    if (q + 1 < n) v += M[p * n + q + 1];
    if (p + 1 < n && q + 1 < n) v -= M[(p + 1) * n + q + 1];
    M[p * n + q] = v;
  }
}
/** out = A^T M A (n x n, A the averaging matrix above). */
function atma(M, out, n) {
  for (let p = 0; p < n; p++) {
    const ap = p >= 1 ? 0.5 : 0, bp = p + 1 < n ? 0.5 : 0;
    for (let q = 0; q < n; q++) {
      const aq = q >= 1 ? 0.5 : 0, bq = q + 1 < n ? 0.5 : 0;
      let v = ap * aq * M[p * n + q];
      if (bq) v += ap * bq * M[p * n + q + 1];
      if (bp) { v += bp * aq * M[(p + 1) * n + q]; if (bq) v += bp * bq * M[(p + 1) * n + q + 1]; }
      out[p * n + q] = v;
    }
  }
}
// (U^T v)[l] and (Y^T v)[l] from the shapes of u and y (kinematicMaps): u[k][l] = 1 (k > l >= 1), 1/2 (k = l >= 1, or
// l = 0 < k); y[k][l] = k - l (k > l >= 1), 1/4 (k = l >= 1), k/2 - 1/4 (l = 0 < k); 0 otherwise.  O(n).
function utv(v, out, n) {
  let s0 = 0;
  for (let l = n - 1; l >= 1; l--) { out[l] = 0.5 * v[l] + s0; s0 += v[l]; }
  out[0] = 0.5 * s0;
}
/** out = Y^T v (as utv for y). */
function ytv(v, out, n) {
  let s0 = 0, s1 = 0;
  for (let l = n - 1; l >= 1; l--) { out[l] = 0.25 * v[l] + s1 - l * s0; s0 += v[l]; s1 += l * v[l]; }
  out[0] = 0.5 * s1 - 0.25 * s0;
}

/**
 * Add the dense part of every recorded term to Hz (rows / columns of the accelerations).  With Y = Sigma A U:
 *   sum = U^T W U + rank-one parts,  W = d2^2 A^T S(T) A + dt d2 (diag(Qd) U + (diag(Rd) U)^T) + dt^2 diag(Sd),
 *   U^T W U = A^T S(W) A.  The Hessian is symmetric: blocks (c, d), c <= d, are computed, (d, c) mirrored.
 */
export function structAdd(ctx, acc, Y, U) {
  const { n, mVar, Hz, dtv, Mm, Nm } = ctx;
  const d2 = dtv * dtv, xx = d2 * d2, xv = dtv * d2, vv = d2;
  const { G, W, f, Le, Lf, R1, R2, t1, t2, t3, t4 } = acc;
  const e = Y.subarray((n - 1) * n, n * n);
  for (let j = 0; j < n; j++) f[j] = e[j] / (n - 1);
  for (let c = 0; c < 3; c++) for (let d = c; d < 3; d++) {
    const i = c * 3 + d;
    const Pd = acc.Pd[i], Po = acc.Po[i], Qd = acc.Qd[i], Rd = acc.Rd[i], Sd = acc.Sd[i];
    // W_T = A^T S(T) A (T tridiagonal)
    G.fill(0, 0, n * n);
    for (let k = 0; k < n; k++) {
      G[k * n + k] = Pd[k];
      if (k + 1 < n) { G[k * n + k + 1] = Po[k]; G[(k + 1) * n + k] = Po[k]; }
    }
    suffix2(G, n); atma(G, W, n);
    for (let p = 0; p < n; p++) {
      const qp = xv * Qd[p], rowP = p * n;
      for (let q = 0; q < n; q++) W[rowP + q] = xx * W[rowP + q] + qp * U[rowP + q] + xv * Rd[q] * U[q * n + p];
      W[rowP + p] += vv * Sd[p];
    }
    suffix2(W, n); atma(W, G, n);
    // rank-one parts: G + Le e^T + Lf f^T + e R1^T + f R2^T
    ytv(acc.aP[i], t1, n); utv(acc.qc[i], t2, n); ytv(Rd, t3, n); utv(Sd, t4, n);
    const sP = acc.sP[i], sQ = acc.sQ[i], sR = acc.sR[i], sS = acc.sS[i];
    for (let j = 0; j < n; j++) {
      Le[j] = -xx * t1[j] + xx * sP * e[j] - xv * t2[j] + xv * sQ * f[j];
      Lf[j] = -xv * t3[j] + xv * sR * e[j] - vv * t4[j] + vv * sS * f[j];
      R1[j] = -xx * t1[j];
      R2[j] = -vv * t4[j];
    }
    utv(acc.rc[i], t2, n); ytv(Qd, t3, n);
    for (let j = 0; j < n; j++) { R1[j] -= xv * t2[j]; R2[j] -= xv * t3[j]; }
    const Bxa = acc.Bxa[i], Bva = acc.Bva[i], Bax = acc.Bax[i], Bav = acc.Bav[i], Baa = acc.Baa[i];
    for (let j = 0; j < n; j++) {
      const a = Le[j], b = Lf[j], ej = e[j], fj = f[j], rowJ = j * n;
      const row = (3 * j + c) * mVar + d, xj = d2 * Bax[j], vj = dtv * Bav[j];
      G[rowJ + j] += Baa[j];
      for (let l = 0; l < n; l++) {
        const v = G[rowJ + l] + a * e[l] + b * f[l] + ej * R1[l] + fj * R2[l]
          + d2 * Bxa[l] * Mm[l * n + j] + xj * Mm[rowJ + l] + dtv * Bva[l] * Nm[l * n + j] + vj * Nm[rowJ + l];
        Hz[row + 3 * l] += v;
        if (d !== c) Hz[(3 * l + d) * mVar + 3 * j + c] += v;
      }
    }
  }
}

/** out[j] = (Mm^T v)[j] = (Y^T v)[j] - e[j] sum_k c_k v_k  (and Nm^T v = U^T v - f sum v): O(n). */
export function mmtv(v, e, out, n) {
  ytv(v, out, n);
  let s = 0;
  for (let k = 0; k < n; k++) s += k * v[k];
  s /= n - 1;
  for (let j = 0; j < n; j++) out[j] -= e[j] * s;
}

/** out[j] = (Nm^T v)[j] = (U^T v)[j] - e[j] sum_k v_k / (n - 1): O(n). */
export function nmtv(v, e, out, n) {
  utv(v, out, n);
  let s = 0;
  for (let k = 0; k < n; k++) s += v[k];
  s /= n - 1;
  for (let j = 0; j < n; j++) out[j] -= e[j] * s;
}
