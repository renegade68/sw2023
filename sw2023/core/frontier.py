"""
SW (2023) Steps 4-6: Nonparametric frontier estimation
Local Linear Least Squares (LLLS) based estimation

Estimation order:
  r̂_1(z) = E[U|Z=z]        → conditional mean (local linear regression)
  ε̂_i = U_i - r̂_1(Z_i)    → residuals
  r̂_2(z) = E[ε̂²|Z=z]      → 2nd conditional moment
  r̂_3(z) = E[ε̂³|Z=z]      → 3rd conditional moment

Performance optimization strategy:
  1. Compute K_full (n×n) once → reuse for r1, r2, r3
  2. Construct XtWX (n, d+1, d+1) once → reuse 3 times (depends only on K, Z)
  3. XtWy is computed 3 times since only y changes (K @ y BLAS operation)
  4. All operations at numpy matrix level → no Python loops
"""

import numpy as np


def _bandwidth_silverman(Z):
    """
    Bandwidth based on Silverman's rule.
    Computed independently per dimension for use with product kernel.

    h_j = 1.06 * std(Z_j) * n^(-1/(d+4))
    """
    n, d = Z.shape
    h = 1.06 * np.std(Z, axis=0) * n ** (-1.0 / (d + 4))
    h = np.where(h == 0, 1e-6, h)
    return h


def _kernel_weights(Z, z0, h):
    """
    Gaussian product kernel weights (single evaluation point, kept for compatibility).

    Parameters
    ----------
    Z  : (n, d)
    z0 : (d,)
    h  : (d,) bandwidth

    Returns
    -------
    w : (n,) kernel weights
    """
    diff  = (Z - z0) / h
    log_w = -0.5 * np.sum(diff ** 2, axis=1)
    log_w -= log_w.max()
    return np.exp(log_w)


def _compute_K_full(Z, h, chunk_size=512):
    """
    Compute the full kernel matrix K (n×n).

    K[i,j] = exp(-0.5 × Σ_d ((Z[i,d]-Z[j,d])/h[d])²)

    Memory usage is controlled by chunk-wise computation:
      Peak memory: chunk × n × d × 8 bytes

    Parameters
    ----------
    Z          : (n, d)
    h          : (d,)
    chunk_size : int

    Returns
    -------
    K : (n, n)  K[i,j] = kernel weight of data point j at eval point i
    """
    n, d = Z.shape
    K    = np.empty((n, n), dtype=float)
    Zh   = Z / h                          # (n, d) scaled

    for start in range(0, n, chunk_size):
        end   = min(start + chunk_size, n)
        Z0h   = Zh[start:end]             # (bs, d)

        diff  = Z0h[:, None, :] - Zh[None, :, :]  # (bs, n, d)
        log_K = -0.5 * np.sum(diff ** 2, axis=-1) # (bs, n)
        log_K -= log_K.max(axis=1, keepdims=True)
        K[start:end] = np.exp(log_K)

    return K


def _compute_XtWX_batch(K, Z, ZZT, reg):
    """
    Construct XtWX for all evaluation points at once from the K matrix.

    XtWX[i] = (X_loc[i]')·diag(K[i])·X_loc[i]
    X_loc[i][j] = [1, Z[j] - Z[i]]

    Depends only on K, Z → independent of y → compute once and reuse.

    Formula (decomposed using K matrix + statistics):
      S00[i]           = Σ_j K[i,j]
      cross[i,k]       = Σ_j K[i,j](Z[j,k]-Z[i,k])  = (KZ)[i,k] - S00[i]·Z[i,k]
      block[i,k,l]     = Σ_j K[i,j](Z[j,k]-Z[i,k])(Z[j,l]-Z[i,l])
                       = KZZ[i,k,l] - (KZ[i,k]·Z[i,l]+Z[i,k]·KZ[i,l])
                         + S00[i]·Z[i,k]·Z[i,l]

    Parameters
    ----------
    K   : (n, n)
    Z   : (n, d)
    ZZT : (n, d, d)  Z[j]⊗Z[j] (precomputed)
    reg : (d+1, d+1) regularization matrix

    Returns
    -------
    XtWX : (n, d+1, d+1)
    """
    n, d = Z.shape

    S00 = K.sum(axis=1)                          # (n,)
    KZ  = K @ Z                                  # (n, d)  — BLAS
    KZZ = np.einsum('ij,jkl->ikl', K, ZZT)      # (n, d, d)

    cross    = KZ - S00[:, None] * Z             # (n, d)
    Z_outer  = np.einsum('ik,il->ikl', Z, Z)     # (n, d, d)
    KZ_ZT    = np.einsum('ik,il->ikl', KZ, Z)    # (n, d, d)
    Z_KZT    = np.einsum('ik,il->ikl', Z, KZ)    # (n, d, d)

    XtWX = np.empty((n, d+1, d+1))
    XtWX[:, 0,  0]  = S00
    XtWX[:, 0,  1:] = cross
    XtWX[:, 1:, 0]  = cross
    XtWX[:, 1:, 1:] = KZZ - KZ_ZT - Z_KZT + S00[:, None, None] * Z_outer
    XtWX += reg

    return XtWX


def _compute_XtWy_batch(K, Z, y):
    """
    Construct XtWy for all evaluation points at once from K matrix and y vector.

    XtWy[i, 0]   = Σ_j K[i,j] y[j]                = (K @ y)[i]
    XtWy[i, k+1] = Σ_j K[i,j] (Z[j,k]-Z[i,k]) y[j]
                 = (K @ (Z[:,k]*y))[i] - Z[i,k] * (K @ y)[i]

    Parameters
    ----------
    K : (n, n)
    Z : (n, d)
    y : (n,)

    Returns
    -------
    XtWy : (n, d+1)
    """
    Ky  = K @ y                           # (n,)   — BLAS
    KZy = K @ (Z * y[:, None])           # (n, d) — BLAS

    XtWy = np.empty((K.shape[0], Z.shape[1] + 1))
    XtWy[:, 0]  = Ky
    XtWy[:, 1:] = KZy - Z * Ky[:, None]
    return XtWy


def _llls_from_normal_equations(XtWX, XtWy, y_fallback):
    """
    Solve batch linear equations XtWX @ beta = XtWy.

    Returns β₀ (intercept) for each eval point.
    """
    try:
        beta = np.linalg.solve(XtWX, XtWy[:, :, np.newaxis])[:, :, 0]
        return beta[:, 0]
    except np.linalg.LinAlgError:
        # Per-point fallback
        n = XtWX.shape[0]
        result = np.full(n, np.nanmean(y_fallback))
        for i in range(n):
            try:
                result[i] = np.linalg.solve(XtWX[i], XtWy[i])[0]
            except np.linalg.LinAlgError:
                pass
        return result


def local_linear(Z, y, h=None, eval_points=None, chunk_size=512,
                 _K_precomputed=None, _XtWX_precomputed=None):
    """
    Local Linear Least Squares regression.

    At each evaluation point z0:
      min_{β} Σ_j K(Z_j, z0; h) × (y_j - β0 - β1'(Z_j - z0))²

    Estimated value: m̂(z0) = β̂_0

    Parameters
    ----------
    Z                  : (n, d) independent variables
    y                  : (n,)   dependent variable
    h                  : (d,) bandwidth (Silverman if None)
    eval_points        : (m, d) evaluation points (Z itself if None)
    chunk_size         : int    chunk size for K computation
    _K_precomputed     : (n, n) precomputed K (for reuse)
    _XtWX_precomputed  : (n, d+1, d+1) precomputed XtWX (for reuse)

    Returns
    -------
    m_hat : (m,) estimated conditional mean
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = Z.shape

    if h is None:
        h = _bandwidth_silverman(Z)
    h = np.broadcast_to(np.atleast_1d(h), (d,)).copy()

    # Optimized path only for eval_points == Z (self-evaluation)
    if eval_points is not None and not np.array_equal(eval_points, Z):
        # eval_points differs from Z: simple chunk-wise fallback
        eval_points = np.asarray(eval_points, dtype=float)
        return _local_linear_external(Z, y, h, eval_points, chunk_size)

    reg = 1e-10 * np.eye(d + 1)

    # ── Compute K (reuse if precomputed) ──────────────────────
    if _K_precomputed is not None:
        K = _K_precomputed
    else:
        K = _compute_K_full(Z, h, chunk_size)

    # ── Compute XtWX (reuse if precomputed) ───────────────────
    if _XtWX_precomputed is not None:
        XtWX = _XtWX_precomputed
    else:
        ZZT  = np.einsum('ij,ik->ijk', Z, Z)    # (n, d, d) — computed once
        XtWX = _compute_XtWX_batch(K, Z, ZZT, reg)

    # ── XtWy (recomputed each time since y changes) ───────────
    XtWy  = _compute_XtWy_batch(K, Z, y)

    return _llls_from_normal_equations(XtWX, XtWy, y)


def _local_linear_external(Z, y, h, eval_points, chunk_size):
    """Chunk-wise fallback for the case eval_points ≠ Z."""
    n, d  = Z.shape
    m_pts = len(eval_points)
    m_hat = np.full(m_pts, np.nan)
    reg   = 1e-10 * np.eye(d + 1)
    Zh    = Z / h
    ZZT   = np.einsum('ij,ik->ijk', Z, Z)
    y_mean = np.nanmean(y)

    for start in range(0, m_pts, chunk_size):
        end = min(start + chunk_size, m_pts)
        Z0h = eval_points[start:end] / h
        diff  = Z0h[:, None, :] - Zh[None, :, :]
        log_K = -0.5 * np.sum(diff ** 2, axis=-1)
        log_K -= log_K.max(axis=1, keepdims=True)
        K_bs  = np.exp(log_K)

        XtWX, XtWy = _build_normal_eq_external(K_bs, Z, eval_points[start:end], y, ZZT, reg)
        try:
            beta = np.linalg.solve(XtWX, XtWy[:, :, np.newaxis])[:, :, 0]
            m_hat[start:end] = beta[:, 0]
        except np.linalg.LinAlgError:
            for i in range(end - start):
                try:
                    m_hat[start+i] = np.linalg.solve(XtWX[i], XtWy[i])[0]
                except np.linalg.LinAlgError:
                    m_hat[start+i] = y_mean
    return m_hat


def _build_normal_eq_external(K, Z_data, Z_eval, y, ZZT, reg):
    """Construct XtWX, XtWy for the case eval_points ≠ Z_data."""
    d  = Z_data.shape[1]
    S00 = K.sum(axis=1)
    KZ  = K @ Z_data
    KZZ = np.einsum('ij,jkl->ikl', K, ZZT)
    cross    = KZ - S00[:, None] * Z_eval
    Z0_outer = np.einsum('ik,il->ikl', Z_eval, Z_eval)
    KZ_Z0T   = np.einsum('ik,il->ikl', KZ, Z_eval)
    Z0_KZT   = np.einsum('ik,il->ikl', Z_eval, KZ)
    XtWX = np.empty((K.shape[0], d+1, d+1))
    XtWX[:, 0, 0]   = S00
    XtWX[:, 0, 1:]  = cross
    XtWX[:, 1:, 0]  = cross
    XtWX[:, 1:, 1:] = KZZ - KZ_Z0T - Z0_KZT + S00[:, None, None] * Z0_outer
    XtWX += reg
    Ky  = K @ y
    KZy = K @ (Z_data * y[:, None])
    XtWy = np.empty((K.shape[0], d+1))
    XtWy[:, 0]  = Ky
    XtWy[:, 1:] = KZy - Z_eval * Ky[:, None]
    return XtWX, XtWy


def estimate_moments(Z, U, h=None, bandwidth_method='silverman', chunk_size=512):
    """
    Estimate conditional moments required for SVKZ/HMS estimation.

    bandwidth_method='silverman' (default):
      Compute K, XtWX once and reuse for r1, r2, r3.

    bandwidth_method='loocv' (SW 2023 paper approach):
      Select optimal bandwidth independently for each of r1, r2, r3 via LOO-CV.
      → K, XtWX are computed 3 times (different bandwidths)

    Parameters
    ----------
    Z                : (n, d) independent variables
    U                : (n,)   dependent variable
    h                : bandwidth (used in silverman mode only; auto if None)
    bandwidth_method : 'silverman' | 'loocv'
    chunk_size       : int

    Returns
    -------
    dict: r1, r2, r3, eps, h, h_r2, h_r3
    """
    Z = np.asarray(Z, dtype=float)
    U = np.asarray(U, dtype=float)
    n, d = Z.shape

    reg = 1e-10 * np.eye(d + 1)
    ZZT = np.einsum('ij,ik->ijk', Z, Z)   # (n, d, d)

    if bandwidth_method == 'loocv':
        # ── LOO-CV: select optimal bandwidth independently per regression ──
        from .bandwidth import bandwidth_loocv

        # Step 1: r1 — U ~ Z
        h_r1  = bandwidth_loocv(Z, U)
        K1    = _compute_K_full(Z, h_r1, chunk_size)
        XtWX1 = _compute_XtWX_batch(K1, Z, ZZT, reg)
        XtWy1 = _compute_XtWy_batch(K1, Z, U)
        r1    = _llls_from_normal_equations(XtWX1, XtWy1, U)
        eps   = U - r1

        # Step 2: r2 — ε̂² ~ Z
        h_r2  = bandwidth_loocv(Z, eps ** 2)
        K2    = _compute_K_full(Z, h_r2, chunk_size)
        XtWX2 = _compute_XtWX_batch(K2, Z, ZZT, reg)
        XtWy2 = _compute_XtWy_batch(K2, Z, eps ** 2)
        r2    = _llls_from_normal_equations(XtWX2, XtWy2, eps ** 2)

        # Step 3: r3 — ε̂³ ~ Z
        h_r3  = bandwidth_loocv(Z, eps ** 3)
        K3    = _compute_K_full(Z, h_r3, chunk_size)
        XtWX3 = _compute_XtWX_batch(K3, Z, ZZT, reg)
        XtWy3 = _compute_XtWy_batch(K3, Z, eps ** 3)
        r3    = _llls_from_normal_equations(XtWX3, XtWy3, eps ** 3)

        return {'r1': r1, 'r2': r2, 'r3': r3, 'eps': eps,
                'h': h_r1, 'h_r2': h_r2, 'h_r3': h_r3}

    else:
        # ── Silverman (default): compute K, XtWX once and reuse ───────
        if h is None:
            h = _bandwidth_silverman(Z)
        h = np.broadcast_to(np.atleast_1d(h), (d,)).copy()

        K    = _compute_K_full(Z, h, chunk_size)
        XtWX = _compute_XtWX_batch(K, Z, ZZT, reg)

        XtWy1 = _compute_XtWy_batch(K, Z, U)
        r1    = _llls_from_normal_equations(XtWX, XtWy1, U)
        eps   = U - r1

        XtWy2 = _compute_XtWy_batch(K, Z, eps ** 2)
        r2    = _llls_from_normal_equations(XtWX, XtWy2, eps ** 2)

        XtWy3 = _compute_XtWy_batch(K, Z, eps ** 3)
        r3    = _llls_from_normal_equations(XtWX, XtWy3, eps ** 3)

        return {'r1': r1, 'r2': r2, 'r3': r3, 'eps': eps,
                'h': h, 'h_r2': h, 'h_r3': h}


def compute_leverages(K, Z):
    """
    Compute hat-matrix diagonal elements h_ii for local linear regression.

    h_ii = [(XtWX_i)^{-1}]_{00}   (under condition K[i,i]=1)

    Used for asymptotic variance estimation:
      AVar(r̂(z_i)) ≈ h_ii × σ²(z_i)

    Parameters
    ----------
    K : (n, n) kernel matrix
    Z : (n, d) independent variables

    Returns
    -------
    h_ii : (n,) leverage values
    """
    n, d = Z.shape
    reg  = 1e-10 * np.eye(d + 1)
    ZZT  = np.einsum('ij,ik->ijk', Z, Z)
    XtWX = _compute_XtWX_batch(K, Z, ZZT, reg)

    e0 = np.zeros((1, d + 1, 1))
    e0[0, 0, 0] = 1.0
    lev_mat = np.linalg.solve(XtWX, e0)    # (n, d+1, 1)
    return lev_mat[:, 0, 0]                 # (n,)
