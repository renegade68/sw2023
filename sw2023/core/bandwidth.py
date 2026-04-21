"""
Bandwidth Selection

SW(2023) paper approach:
  Leave-One-Out Cross-Validation (LOO-CV)
  - Independently optimized for each of U ~ Z, ε̂² ~ Z, ε̂³ ~ Z regressions
  - Per-dimension product kernel: h_k selected independently for k=1,...,d
    (consistent with SW(2023) Table G.2 Banking Application)
  - Lower bound: 0.1 × σ̂_k × n^{-1/(d+4)}
  - Upper bound: 3 × [max(Z_k) - min(Z_k)]

  bandwidth_loocv_product (recommended, SW(2023)-consistent):
    Coordinate descent over h_k; K[i,j] = ∏_k exp(-½((Z_ik-Z_jk)/h_k)²)

  bandwidth_loocv (legacy):
    Scalar multiplier c; h_k = c × h_ref_k  (proportional scaling only)

Hat-matrix LOO trick:
  ŷ_{-i} = (ŷ_i − h_{ii} × y_i) / (1 − h_{ii})
  h_{ii} = [(XtWX_i)^{-1}]_{00}   (holds since K[i,i]=1)
  → LOO-CV can be computed in O(n²) without n re-fits

Reference: Fan & Gijbels (1996) Ch.4, Simar & Wilson (2023) Sec.4, Table G.2
"""

import numpy as np
from scipy.optimize import minimize_scalar

from .frontier import (_compute_K_full, _compute_XtWX_batch,
                        _compute_XtWy_batch, _llls_from_normal_equations)


# ─────────────────────────────────────────────────────────────
# Reference distance matrix (key acceleration for c search)
# ─────────────────────────────────────────────────────────────

def _precompute_S(Z, h_ref, chunk_size=512):
    """
    Squared distance matrix at reference scale.

    S[i,j] = 0.5 × Σ_k ((Z_ik - Z_jk) / h_ref_k)²

    For scalar multiplier c: K_c[i,j] = exp(−S[i,j] / c²)
    → Computing S once allows K to be obtained in O(n²) for each c.

    Parameters
    ----------
    Z         : (n, d)
    h_ref     : (d,) reference bandwidth
    chunk_size: int

    Returns
    -------
    S : (n, n)  ≥ 0, S[i,i] = 0
    """
    n, d = Z.shape
    Zh   = Z / h_ref                      # (n, d) scaled
    S    = np.empty((n, n), dtype=float)

    for start in range(0, n, chunk_size):
        end  = min(start + chunk_size, n)
        diff = Zh[start:end, None, :] - Zh[None, :, :]   # (bs, n, d)
        S[start:end] = 0.5 * np.sum(diff ** 2, axis=-1)  # (bs, n)

    return S


# ─────────────────────────────────────────────────────────────
# LOO-CV score (hat-matrix trick)
# ─────────────────────────────────────────────────────────────

def _loocv_score_from_K(K, Z, y):
    """
    Compute LOO-CV score from a given K matrix.

    Hat-matrix trick:
      h_{ii} = [(XtWX_i)^{-1}]_{00}   (K[i,i] = 1 guaranteed)
      LOO residual = (y_i − ŷ_i) / (1 − h_{ii})
      CV = mean(LOO residual²)

    Parameters
    ----------
    K : (n, n) kernel matrix (K[i,i] = 1)
    Z : (n, d)
    y : (n,)

    Returns
    -------
    cv : float  LOO-CV score (lower is better)
    """
    n, d = Z.shape
    reg  = 1e-10 * np.eye(d + 1)

    ZZT  = np.einsum('ij,ik->ijk', Z, Z)          # (n, d, d)
    XtWX = _compute_XtWX_batch(K, Z, ZZT, reg)    # (n, d+1, d+1)
    XtWy = _compute_XtWy_batch(K, Z, y)           # (n, d+1)

    # Fitted values
    beta  = np.linalg.solve(XtWX, XtWy[:, :, np.newaxis])[:, :, 0]  # (n, d+1)
    y_hat = beta[:, 0]                             # (n,)

    # Leverages: h_{ii} = (XtWX^{-1})[0,0]
    # solve(XtWX, e0): XtWX is (n,d+1,d+1), RHS is (1,d+1,1) → (n,d+1,1)
    e0        = np.zeros((1, d + 1, 1));  e0[0, 0, 0] = 1.0
    lev_mat   = np.linalg.solve(XtWX, e0)         # (n, d+1, 1)
    h_ii      = lev_mat[:, 0, 0]                   # (n,)

    # LOO residuals
    # When h_ii → 1, ŷ_i → y_i (interpolation), resid → 0, denom → 0 simultaneously
    # Numerical 0/0 handling: replace h_ii > 0.95 region with global mean residual
    # (bandwidth too small to find neighbors → worst prediction = global mean)
    resid       = y - y_hat
    global_resid = y - np.mean(y)        # global mean based residual (penalty)

    denom = 1.0 - h_ii
    hat_loo = resid / np.where(denom > 1e-6, denom, 1e-6)

    # h_ii > 0.95: bandwidth too small → apply penalty
    loo_sq = np.where(h_ii > 0.95, global_resid ** 2, hat_loo ** 2)

    return float(np.mean(loo_sq))


# ─────────────────────────────────────────────────────────────
# Lower bound computation based on effective neighbor count
# ─────────────────────────────────────────────────────────────

def _find_c_neff(S, k_min):
    """
    Find via binary search the minimum c such that the average effective
    sample size is at least k_min.

    avg_neff(c) = (1/n) × Σ_i Σ_j K_c[i,j]  (K_c[i,j] = exp(−S[i,j]/c²))

    When c is too small, K[i,j]≈0 (j≠i), each point has only itself as neighbor
    → interpolation occurs.
    Setting c_low = max(paper lower bound, c_neff) avoids searching the degenerate region.

    Parameters
    ----------
    S     : (n, n) reference distance matrix
    k_min : float  target minimum effective neighbor count

    Returns
    -------
    c_low : float  avg_neff(c_low) ≈ k_min
    """
    n = S.shape[0]
    lo_lc, hi_lc = -4.0, 5.0   # c ∈ [exp(-4), exp(5)] ≈ [0.018, 148]

    for _ in range(60):
        mid_lc = 0.5 * (lo_lc + hi_lc)
        K = np.exp(-S / np.exp(2.0 * mid_lc))
        avg_neff = float(K.sum()) / n       # includes K[i,i]=1
        if avg_neff < k_min:
            lo_lc = mid_lc
        else:
            hi_lc = mid_lc

    return float(np.exp(0.5 * (lo_lc + hi_lc)))


# ─────────────────────────────────────────────────────────────
# Scalar LOO-CV bandwidth optimization
# ─────────────────────────────────────────────────────────────

def bandwidth_loocv(Z, y, h_ref=None, n_grid=15, verbose=False):
    """
    Scalar multiplier LOO-CV bandwidth optimization.

    Search for c* minimizing h_k = c* × h_ref_k.
    h_ref_k = σ̂_k × n^{-1/(d+4)}  (optimal convergence rate reference)

    SW(2023) constraints: lower bound c >= max(0.1, c_neff) ensures average
    effective neighbors >= max(5, 2(d+2)); upper bound h_k <= 3 × range(Z_k).

    Parameters
    ----------
    Z       : (n, d) explanatory variables
    y       : (n,)  dependent variable
    h_ref   : (d,) reference bandwidth (auto-computed if None)
    n_grid  : number of coarse grid search points
    verbose : whether to print search progress

    Returns
    -------
    h_opt : (d,) optimal bandwidth
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = Z.shape

    # Reference bandwidth
    std_Z = Z.std(axis=0)
    std_Z = np.where(std_Z < 1e-10, 1e-10, std_Z)

    if h_ref is None:
        h_ref = std_Z * n ** (-1.0 / (d + 4))

    # SW(2023) bounds
    range_Z = Z.max(0) - Z.min(0)
    range_Z = np.where(range_Z < 1e-10, 1e-10, range_Z)
    c_high = float(np.min(3.0 * range_Z / h_ref))
    c_high = min(max(c_high, 1.0), 20.0)   # clip to minimum 1.0, maximum 20.0

    # Reference distance matrix (S is independent of c → computed once)
    S = _precompute_S(Z, h_ref)

    # Adaptive lower bound: average effective neighbors ≥ k_min = max(5, 2×(d+2))
    # Too small values like c=0.1 cause interpolation → degenerate CV curve
    k_min = max(5.0, 2.0 * (d + 2))
    c_neff = _find_c_neff(S, k_min)
    c_low  = max(0.1, c_neff)
    c_low  = min(c_low, c_high * 0.5)      # limit to at most half of upper bound

    if verbose:
        print(f"    LOO-CV bounds: c_low={c_low:.3f} (neff≥{k_min:.0f}), "
              f"c_high={c_high:.3f}")

    def cv_loss(log_c):
        c = np.exp(log_c)
        K = np.exp(-S / (c ** 2))           # K[i,i] = exp(0) = 1
        return _loocv_score_from_K(K, Z, y)

    # Stage 1: coarse grid
    log_c_lo   = np.log(c_low)
    log_c_hi   = np.log(c_high)
    log_c_grid = np.linspace(log_c_lo, log_c_hi, n_grid)
    cv_vals    = [cv_loss(lc) for lc in log_c_grid]
    best_idx   = int(np.argmin(cv_vals))
    best_lc    = log_c_grid[best_idx]

    if verbose:
        print(f"    LOO-CV grid: c={np.exp(best_lc):.3f}, "
              f"CV={cv_vals[best_idx]:.6f}")

    # Stage 2: golden-section refinement
    lo = log_c_grid[max(0, best_idx - 1)]
    hi = log_c_grid[min(n_grid - 1, best_idx + 1)]
    result = minimize_scalar(cv_loss, bounds=(lo, hi),
                              method='bounded',
                              options={'xatol': 0.02})
    c_opt = float(np.exp(result.x))

    if verbose:
        print(f"    LOO-CV refined: c={c_opt:.3f}, "
              f"CV={result.fun:.6f}")

    return c_opt * h_ref


# ─────────────────────────────────────────────────────────────
# Silverman rule (default, backward compatible)
# ─────────────────────────────────────────────────────────────

def bandwidth_silverman(Z):
    """
    Silverman rule bandwidth.

    h_j = 1.06 × σ̂_j × n^{-1/(d+4)}
    """
    Z = np.asarray(Z, dtype=float)
    n, d = Z.shape
    h = 1.06 * Z.std(axis=0) * n ** (-1.0 / (d + 4))
    return np.where(h == 0, 1e-6, h)


# ─────────────────────────────────────────────────────────────
# Per-dimension product-kernel LOO-CV (SW 2023 Table G.2)
# ─────────────────────────────────────────────────────────────

def bandwidth_loocv_product(Z, y, h_ref=None, n_grid=10, max_iter=5,
                             tol=1e-3, verbose=False):
    """
    Per-dimension LOO-CV bandwidth (true product kernel).

    Independently optimizes h_k for each dimension k via coordinate descent,
    consistent with SW(2023) Table G.2 which reports d separate bandwidths.

    Kernel: K[i,j] = exp(-0.5 × Σ_k ((Z_ik - Z_jk) / h_k)²)
            = ∏_k exp(-0.5 × ((Z_ik - Z_jk) / h_k)²)

    Maintains ``log_K = -0.5 × Σ_k D_k / c_k²`` (n×n, O(n²) memory).
    For each dimension k, removes its contribution, optimizes ``c_k`` via
    coarse grid + golden-section, and updates ``log_K``.
    Stops when the maximum relative change in any ``c_k`` is below ``tol``.

    Memory: O(n²) — D_k computed on-the-fly, never stored as (d,n,n) tensor.

    Parameters
    ----------
    Z        : (n, d) conditioning variables
    y        : (n,)  dependent variable
    h_ref    : (d,)  reference bandwidth (Silverman rule if None)
    n_grid   : coarse grid points per 1-D search (default 10)
    max_iter : maximum coordinate-descent iterations (default 5)
    tol      : relative convergence tolerance (default 1e-3)
    verbose  : print iteration progress

    Returns
    -------
    h_opt : (d,) per-dimension optimal bandwidths
    """
    Z = np.asarray(Z, dtype=float)
    y = np.asarray(y, dtype=float)
    n, d = Z.shape

    # Reference bandwidth (Silverman)
    std_Z = Z.std(axis=0)
    std_Z = np.where(std_Z < 1e-10, 1e-10, std_Z)
    if h_ref is None:
        h_ref = std_Z * n ** (-1.0 / (d + 4))

    # Per-dimension bounds on the multiplier c_k  (h_k = c_k × h_ref_k)
    range_Z  = Z.max(0) - Z.min(0)
    range_Z  = np.where(range_Z < 1e-10, 1e-10, range_Z)
    c_high_k = np.minimum(3.0 * range_Z / h_ref, 20.0)
    c_low_k  = np.full(d, 0.1)

    # Warm-start c from scalar LOO-CV
    S_init = _precompute_S(Z, h_ref)
    k_min  = max(5.0, 2.0 * (d + 2))
    c_init = max(_find_c_neff(S_init, k_min), 0.2)
    c = np.full(d, c_init)

    # Helper: per-dimension squared scaled distance matrix  O(n²)
    def _D_k(k):
        zk = Z[:, k] / h_ref[k]
        return (zk[:, None] - zk[None, :]) ** 2

    # Initialise log_K = Σ_k log_K_k (n×n)
    log_K = np.zeros((n, n), dtype=float)
    for k in range(d):
        log_K -= 0.5 * _D_k(k) / (c[k] ** 2)

    for iteration in range(max_iter):
        c_prev = c.copy()

        for k in range(d):
            D_k = _D_k(k)
            log_K_k_old = -0.5 * D_k / (c[k] ** 2)   # current contribution

            if d > 1:
                # K_rest via O(n²) subtraction — no (d-1)-fold sum
                K_rest = np.exp(log_K - log_K_k_old)
            else:
                K_rest = np.ones((n, n))

            def cv_k(log_ck, D_k=D_k, K_rest=K_rest):
                ck = np.exp(log_ck)
                K  = K_rest * np.exp(-0.5 * D_k / (ck ** 2))
                return _loocv_score_from_K(K, Z, y)

            lo_lc = np.log(c_low_k[k])
            hi_lc = np.log(c_high_k[k])
            grid  = np.linspace(lo_lc, hi_lc, n_grid)
            cv_g  = [cv_k(lc) for lc in grid]
            bi    = int(np.argmin(cv_g))

            lo2 = grid[max(0, bi - 1)]
            hi2 = grid[min(n_grid - 1, bi + 1)]
            res = minimize_scalar(cv_k, bounds=(lo2, hi2),
                                  method='bounded',
                                  options={'xatol': 0.02})
            c[k] = float(np.exp(res.x))

            # Update log_K with new c[k]
            log_K_k_new = -0.5 * D_k / (c[k] ** 2)
            log_K += log_K_k_new - log_K_k_old

        if verbose:
            print(f"    product LOO-CV iter {iteration + 1}: "
                  f"c = {np.round(c, 3)}")

        if np.max(np.abs(c - c_prev) / (c_prev + 1e-10)) < tol:
            if verbose:
                print(f"    Converged at iteration {iteration + 1}.")
            break

    return c * h_ref
