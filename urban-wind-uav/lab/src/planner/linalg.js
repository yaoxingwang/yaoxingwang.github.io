// Dense linear algebra of the trajectory optimizer (trajopt.js): Cholesky factorizations and their solve, row-major.

/** Lower-triangular Cholesky of a dense SPD matrix (row-major, size m) into L; false if not SPD. */
export function cholesky(A, m, L) {
  for (let i = 0; i < m; i++) {
    for (let j = 0; j <= i; j++) {
      let sum = A[i * m + j];
      for (let k = 0; k < j; k++) sum -= L[i * m + k] * L[j * m + k];
      if (i === j) {
        if (!(sum > 0)) return false;
        L[i * m + i] = Math.sqrt(sum);
      } else {
        L[i * m + j] = sum / L[j * m + j];
      }
    }
  }
  return true;
}

/**
 * The factorization of the Newton steps, with register blocking: rows i..i+3 against rows j, j+1 share their loads.
 * In the blocked part a sum of products is accumulated first and then subtracted from A (cholesky subtracts term by
 * term), so its last bits can differ from cholesky's.
 */
export function choleskyBlocked(A, m, L) {
  const one = (r, ii, j) => {
    const rj = j * m;
    let sum = A[r + j];
    for (let k = 0; k < j; k++) sum -= L[r + k] * L[rj + k];
    if (ii === j) {
      if (!(sum > 0)) return false;
      L[r + j] = Math.sqrt(sum);
    } else {
      L[r + j] = sum / L[rj + j];
    }
    return true;
  };
  let i = 0;
  for (; i + 3 < m; i += 4) {
    const r0 = i * m, r1 = r0 + m, r2 = r1 + m, r3 = r2 + m;
    let j = 0;
    for (; j + 1 < i; j += 2) {
      const q0 = j * m, q1 = q0 + m;
      let a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0, c1 = 0, d0 = 0, d1 = 0;
      for (let k = 0; k < j; k++) {
        const y0 = L[q0 + k], y1 = L[q1 + k];
        const x0 = L[r0 + k], x1 = L[r1 + k], x2 = L[r2 + k], x3 = L[r3 + k];
        a0 += x0 * y0; a1 += x0 * y1; b0 += x1 * y0; b1 += x1 * y1;
        c0 += x2 * y0; c1 += x2 * y1; d0 += x3 * y0; d1 += x3 * y1;
      }
      const p0 = L[q0 + j], y1j = L[q1 + j], p1 = L[q1 + j + 1];
      const l0 = (A[r0 + j] - a0) / p0, l1 = (A[r1 + j] - b0) / p0;
      const l2 = (A[r2 + j] - c0) / p0, l3 = (A[r3 + j] - d0) / p0;
      L[r0 + j] = l0; L[r1 + j] = l1; L[r2 + j] = l2; L[r3 + j] = l3;
      L[r0 + j + 1] = (A[r0 + j + 1] - a1 - l0 * y1j) / p1; L[r1 + j + 1] = (A[r1 + j + 1] - b1 - l1 * y1j) / p1;
      L[r2 + j + 1] = (A[r2 + j + 1] - c1 - l2 * y1j) / p1; L[r3 + j + 1] = (A[r3 + j + 1] - d1 - l3 * y1j) / p1;
    }
    for (; j <= i + 3; j++) {
      for (let rr = Math.max(0, j - i); rr < 4; rr++) if (!one(r0 + rr * m, i + rr, j)) return false;
    }
  }
  for (; i < m; i++) for (let j = 0; j <= i; j++) if (!one(i * m, i, j)) return false;
  return true;
}

/** Solve L L^T x = b. */
export function choleskySolve(L, m, b, x) {
  for (let i = 0; i < m; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * m + k] * x[k];
    x[i] = s / L[i * m + i];
  }
  for (let i = m - 1; i >= 0; i--) {
    let s = x[i];
    for (let k = i + 1; k < m; k++) s -= L[k * m + i] * x[k];
    x[i] = s / L[i * m + i];
  }
  return x;
}
