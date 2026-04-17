"""
Bootstrap Confidence Intervals

References:
  Parmeter, Simar, Van Keilegom & Zelenyuk (2024, Econometric Reviews):
    "Inference in the Nonparametric Stochastic Frontier Model"
    → Wild bootstrap for hypothesis tests (significance / specification)

  Simar & Wilson (2023, JBES):
    → "variances have to be evaluated by bootstrap techniques"
    → Pairs bootstrap for phi_hat(z), sigma_eta(z), efficiency CIs

Implementation:
  bootstrap_sw       : Pairs bootstrap — phi_hat, efficiency confidence intervals
  bootstrap_panel    : Cluster bootstrap — panel model
  test_r3_significance: Wild bootstrap — test whether r3(z) depends on specific covariates

Correct individual CI computation:
  ① Fit original sample → store Z_orig, U_orig, d_fixed
  ② After resampling, transform with the same direction vector d_fixed → Z_b, U_b
  ③ Estimate moments from Z_b, U_b
  ④ Evaluate estimated moment functions at Z_orig (local_linear eval_points)
  ⑤ Collect phi_hat_b(Z_orig), eff_b(Z_orig) → CI via quantiles

Note: n_jobs>1 (joblib parallelization) is prohibited — causes kernel panic on iMac 2017
"""

import numpy as np
import pandas as pd

from .transform  import transform
from .frontier   import estimate_moments, local_linear
from .decompose  import (estimate_sigma_eta, estimate_sigma_eps,
                         estimate_frontier, jlms_efficiency)
from .preprocess import preprocess_apply


# ─────────────────────────────────────────────────────────────
# Internal: single iteration function
# ─────────────────────────────────────────────────────────────

def _boot_iter_sw(seed, X_prep, Y_prep, d, norm_d,
                  Z_orig, U_orig, method, bandwidth_method):
    """
    One bootstrap iteration for the SW model.

    Resample → transform with same d → estimate moments → evaluate at Z_orig.

    Returns
    -------
    phi_hat_b  : (n,) bootstrap frontier estimate at original Z points
    sigma_eta_b: (n,)
    sigma_eps_b: (n,)
    eff_b      : (n,) BC efficiency based on original U values
    """
    rng = np.random.default_rng(seed)
    n   = len(X_prep)
    idx = rng.integers(0, n, size=n)

    X_b = X_prep[idx]
    Y_b = Y_prep[idx]

    # Rotate with the same direction vector
    Z_b, U_b, _ = transform(X_b, Y_b, d)

    # Estimate moments from bootstrap sample
    moments = estimate_moments(Z_b, U_b, bandwidth_method=bandwidth_method)
    h1   = moments['h']
    h2   = moments['h_r2']
    h3   = moments['h_r3']
    eps_b = moments['eps']   # = U_b - r̂_1(Z_b)

    # Evaluate at original Z points (using eval_points)
    r1_b = local_linear(Z_b, U_b,       h=h1, eval_points=Z_orig)
    r2_b = local_linear(Z_b, eps_b**2,  h=h2, eval_points=Z_orig)
    r3_b = local_linear(Z_b, eps_b**3,  h=h3, eval_points=Z_orig)

    sigma_eta_b = estimate_sigma_eta(r3_b, method=method)
    sigma_eps_b = estimate_sigma_eps(r2_b, sigma_eta_b)
    phi_hat_b, _ = estimate_frontier(r1_b, sigma_eta_b, norm_d)
    eff_b, _     = jlms_efficiency(U_orig, phi_hat_b, sigma_eta_b, sigma_eps_b)

    return phi_hat_b, sigma_eta_b, sigma_eps_b, eff_b


def _boot_iter_panel(seed, X_prep, Y_prep, firm_id, time_id, firms,
                     d, norm_d, Z_orig, U_orig, method, bandwidth_method):
    """
    One bootstrap iteration for the panel model (cluster resampling).

    Resample by firm unit to preserve within-panel time-series dependence.
    """
    from ..panel.four_component import PanelSW2023
    rng    = np.random.default_rng(seed)
    N      = len(firms)
    chosen = rng.choice(firms, size=N, replace=True)

    idx_list, new_fid = [], []
    for k, f in enumerate(chosen):
        mask = firm_id == f
        idx_list.append(np.where(mask)[0])
        new_fid.extend([k] * mask.sum())
    idx_all = np.concatenate(idx_list)
    new_fid = np.array(new_fid)

    X_b = X_prep[idx_all]
    Y_b = Y_prep[idx_all]

    Z_b, U_b, _ = transform(X_b, Y_b, d)
    moments = estimate_moments(Z_b, U_b, bandwidth_method=bandwidth_method)
    h1, h2, h3 = moments['h'], moments['h_r2'], moments['h_r3']
    eps_b = moments['eps']

    r1_b = local_linear(Z_b, U_b,       h=h1, eval_points=Z_orig)
    r2_b = local_linear(Z_b, eps_b**2,  h=h2, eval_points=Z_orig)
    r3_b = local_linear(Z_b, eps_b**3,  h=h3, eval_points=Z_orig)

    sigma_eta_b = estimate_sigma_eta(r3_b, method=method)
    sigma_eps_b = estimate_sigma_eps(r2_b, sigma_eta_b)
    phi_hat_b, _ = estimate_frontier(r1_b, sigma_eta_b, norm_d)
    eff_b, _     = jlms_efficiency(U_orig, phi_hat_b, sigma_eta_b, sigma_eps_b)

    return phi_hat_b, sigma_eta_b, sigma_eps_b, eff_b


# ─────────────────────────────────────────────────────────────
# Public API: bootstrap_sw
# ─────────────────────────────────────────────────────────────

def bootstrap_sw(X, Y, B=200, alpha=0.05,
                 direction='mean', method='HMS',
                 log_transform=True, standardize=True,
                 bandwidth_method='silverman',
                 seed=None, verbose=True):
    """
    Pairs bootstrap confidence intervals for SW2023Model.

    To correctly compute individual observation CIs:
      - Fix the original direction vector d
      - Evaluate moment functions of resampled data at original Z points.

    Parameters
    ----------
    X, Y             : Original data
    B                : Number of bootstrap iterations
    alpha            : Significance level (1-alpha confidence interval)
    direction        : Direction vector specification ('mean' | 'median' | array)
    method           : 'HMS' | 'SVKZ'
    log_transform    : bool
    standardize      : bool
    bandwidth_method : 'silverman' | 'loocv'  ('silverman' recommended for bootstrap)
    seed             : Random seed
    verbose          : Print progress

    Returns
    -------
    dict:
        phi_hat_point      : (n,) original frontier estimates
        phi_hat_ci         : (n, 2) frontier confidence intervals
        eff_mean_point     : float mean efficiency estimate
        eff_mean_ci        : (2,) mean efficiency confidence interval
        eff_individual_point: (n,) individual efficiency estimates
        eff_individual_ci  : (n, 2) individual efficiency confidence intervals
        sigma_eta_ci       : (n, 2) sigma_eta confidence intervals
        B, alpha           : parameters used
    """
    import gc
    from .model      import SW2023Model
    from .preprocess import preprocess
    from .transform  import make_direction

    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)

    # ── Fit original model ──────────────────────────────────────────
    if verbose:
        print("Estimating original model...")
    m0 = SW2023Model(X, Y, direction=direction, method=method,
                     log_transform=log_transform, standardize=standardize,
                     bandwidth_method=bandwidth_method)
    m0.fit(verbose=False)

    # Parameters to fix from original model
    d_fixed   = m0.d_
    norm_d    = m0.norm_d_
    Z_orig    = m0.Z_
    U_orig    = m0.U_
    prep_info = m0.preprocess_info_

    if verbose:
        print(f"Starting bootstrap (B={B}, bandwidth={bandwidth_method})...")

    # ── Preprocessed X, Y (with fixed parameters) ────────────────────
    X_prep, Y_prep = preprocess_apply(X, Y, prep_info)

    # ── Bootstrap loop ─────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=B)
    n     = len(X)

    boot_phi      = np.full((B, n), np.nan)
    boot_sigma_eta = np.full((B, n), np.nan)
    boot_sigma_eps = np.full((B, n), np.nan)
    boot_eff      = np.full((B, n), np.nan)

    n_fail = 0
    for b, s in enumerate(seeds):
        if verbose and (b % 50 == 0):
            print(f"  Bootstrap {b+1}/{B}...")
        try:
            phi_b, seta_b, seps_b, eff_b = _boot_iter_sw(
                s, X_prep, Y_prep, d_fixed, norm_d,
                Z_orig, U_orig, method, bandwidth_method
            )
            boot_phi[b]       = phi_b
            boot_sigma_eta[b] = seta_b
            boot_sigma_eps[b] = seps_b
            boot_eff[b]       = eff_b
        except Exception:
            n_fail += 1
        finally:
            gc.collect()

    if n_fail > 0 and verbose:
        print(f"  Warning: {n_fail}/{B} iterations failed")

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    result = {
        # Frontier phi_hat(z)
        'phi_hat_point'       : m0.phi_hat_,
        'phi_hat_ci'          : np.column_stack([
            np.nanpercentile(boot_phi, lo, axis=0),
            np.nanpercentile(boot_phi, hi, axis=0),
        ]),
        # Mean efficiency
        'eff_mean_point'      : float(np.nanmean(m0.efficiency_)),
        'eff_mean_ci'         : np.nanpercentile(
                                    np.nanmean(boot_eff, axis=1), [lo, hi]),
        # Individual efficiency
        'eff_individual_point': m0.efficiency_,
        'eff_individual_ci'   : np.column_stack([
            np.nanpercentile(boot_eff, lo, axis=0),
            np.nanpercentile(boot_eff, hi, axis=0),
        ]),
        # sigma_eta
        'sigma_eta_point'     : m0.sigma_eta_,
        'sigma_eta_ci'        : np.column_stack([
            np.nanpercentile(boot_sigma_eta, lo, axis=0),
            np.nanpercentile(boot_sigma_eta, hi, axis=0),
        ]),
        'B'    : B,
        'alpha': alpha,
        'n_fail': n_fail,
    }

    if verbose:
        ci = result['eff_mean_ci']
        print(f"\nMean efficiency: {result['eff_mean_point']:.4f}  "
              f"{int((1-alpha)*100)}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    return result


# ─────────────────────────────────────────────────────────────
# Public API: bootstrap_panel
# ─────────────────────────────────────────────────────────────

def bootstrap_panel(X, Y, firm_id, time_id,
                    B=200, alpha=0.05,
                    direction='mean', method='HMS',
                    log_transform=True, standardize=True,
                    bandwidth_method='silverman',
                    seed=None, verbose=True):
    """
    Cluster bootstrap confidence intervals for PanelSW2023.

    Resample by firm unit to preserve within-panel time-series dependence.
    Individual observation CIs are corrected in the same way as bootstrap_sw:
      Resample → transform with same d → evaluate at original Z points.

    Parameters
    ----------
    X, Y             : Original data (n_total × ...)
    firm_id, time_id : Firm and time period identifiers (n_total,)
    (rest: same as bootstrap_sw)

    Returns
    -------
    dict: bootstrap_sw result + decomposed transient/persistent efficiency CIs
    """
    import gc
    from ..panel.four_component import PanelSW2023
    from .preprocess import preprocess
    from .transform  import make_direction

    X       = np.asarray(X, dtype=float)
    Y       = np.asarray(Y, dtype=float)
    firm_id = np.asarray(firm_id)
    time_id = np.asarray(time_id)
    firms   = np.unique(firm_id)

    # ── Fit original model ──────────────────────────────────────────
    if verbose:
        print("Estimating original panel model...")
    m0 = PanelSW2023(X, Y, firm_id, time_id,
                     direction=direction, method=method,
                     log_transform=log_transform, standardize=standardize,
                     bandwidth_method=bandwidth_method)
    m0.fit(verbose=False)

    d_fixed   = m0.d_
    norm_d    = m0.norm_d_
    Z_orig    = m0.Z_
    U_orig    = m0.U_
    prep_info = m0.preprocess_info_

    if verbose:
        print(f"Starting cluster bootstrap (B={B}, N_firms={len(firms)})...")

    X_prep, Y_prep = preprocess_apply(X, Y, prep_info)

    rng   = np.random.default_rng(seed)
    seeds = rng.integers(0, 2**31, size=B)
    n     = len(X)

    boot_phi       = np.full((B, n), np.nan)
    boot_sigma_eta = np.full((B, n), np.nan)
    boot_sigma_eps = np.full((B, n), np.nan)
    boot_eff       = np.full((B, n), np.nan)

    n_fail = 0
    for b, s in enumerate(seeds):
        if verbose and (b % 50 == 0):
            print(f"  Bootstrap {b+1}/{B}...")
        try:
            phi_b, seta_b, seps_b, eff_b = _boot_iter_panel(
                s, X_prep, Y_prep, firm_id, time_id, firms,
                d_fixed, norm_d, Z_orig, U_orig, method, bandwidth_method
            )
            boot_phi[b]        = phi_b
            boot_sigma_eta[b]  = seta_b
            boot_sigma_eps[b]  = seps_b
            boot_eff[b]        = eff_b
        except Exception:
            n_fail += 1
        finally:
            gc.collect()

    if n_fail > 0 and verbose:
        print(f"  Warning: {n_fail}/{B} iterations failed")

    lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

    result = {
        'phi_hat_point'       : m0.phi_hat_,
        'phi_hat_ci'          : np.column_stack([
            np.nanpercentile(boot_phi, lo, axis=0),
            np.nanpercentile(boot_phi, hi, axis=0),
        ]),
        'eff_mean_point'      : float(np.nanmean(m0.efficiency_)),
        'eff_mean_ci'         : np.nanpercentile(
                                    np.nanmean(boot_eff, axis=1), [lo, hi]),
        'eff_individual_point': m0.efficiency_,
        'eff_individual_ci'   : np.column_stack([
            np.nanpercentile(boot_eff, lo, axis=0),
            np.nanpercentile(boot_eff, hi, axis=0),
        ]),
        'sigma_eta_point'     : m0.sigma_eta_,
        'sigma_eta_ci'        : np.column_stack([
            np.nanpercentile(boot_sigma_eta, lo, axis=0),
            np.nanpercentile(boot_sigma_eta, hi, axis=0),
        ]),
        'B'     : B,
        'alpha' : alpha,
        'n_fail': n_fail,
    }

    if verbose:
        ci = result['eff_mean_ci']
        print(f"\nMean efficiency: {result['eff_mean_point']:.4f}  "
              f"{int((1-alpha)*100)}% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

    return result


# ─────────────────────────────────────────────────────────────
# PSVKZ Wild Bootstrap: inefficiency significance test
# ─────────────────────────────────────────────────────────────

def test_r3_significance(X, Y, direction='mean', method='HMS',
                         log_transform=True, standardize=True,
                         bandwidth_method='silverman',
                         B=499, seed=None, verbose=True):
    """
    Test whether inefficiency (eta) genuinely depends on observable covariates Z.

    Null hypothesis H0: E(epsilon^3|Z) = const  (spatially uniform inefficiency)
    Alternative   H1: E(epsilon^3|Z) != const  (heterogeneous inefficiency)

    PSVKZ(2024) Section 3.1 wild bootstrap (Rademacher perturbation):
      1) Estimate r_hat_3(Z_i) → residuals eta_i = eps_hat^3_i - r_hat_3(Z_i)
      2) Center: eta^c_i = eta_i - mean(eta)
      3) Wild perturbation: eta*_i = eta^c_i × V_i,  V_i ~ Rademacher(+-1, 50% each)
      4) Bootstrap response: eps_hat^3'*_i = r_hat_3(Z_i) + eta*_i
      5) Re-estimate r_hat*_3(Z) from eps_hat^3'* → compute test statistic T*_b
      6) p-value = #{T*_b >= T_obs} / B

    Test statistic (variance-weighted average heterogeneity):
      T = (1/n) × sum_i (r_hat_3(Z_i) - mean(r_hat_3))^2 / Var(eps_hat^3)

    Parameters
    ----------
    X, Y             : Original data
    direction        : Direction vector
    method           : 'HMS' | 'SVKZ'
    bandwidth_method : Bandwidth method
    B                : Number of bootstrap iterations (default 499)
    seed             : Random seed
    verbose          : Print output

    Returns
    -------
    dict:
        statistic : float  observed test statistic T
        p_value   : float  bootstrap p-value
        r3_hat    : (n,)   estimated r_hat_3(Z)
        B         : int
    """
    import gc
    from .model      import SW2023Model
    from .preprocess import preprocess

    if verbose:
        print("Inefficiency significance test (PSVKZ Wild Bootstrap)...")

    # ── Fit original model ──────────────────────────────────────────
    m0 = SW2023Model(X, Y, direction=direction, method=method,
                     log_transform=log_transform, standardize=standardize,
                     bandwidth_method=bandwidth_method)
    m0.fit(verbose=False)

    eps_cubed = m0.eps_ ** 3
    r3_obs    = m0.r3_
    resid     = eps_cubed - r3_obs
    resid_c   = resid - resid.mean()   # centered residuals

    # Observed test statistic: spatial variation of r3 / total variance
    r3_var  = np.var(r3_obs, ddof=1)
    eps_var = np.var(eps_cubed, ddof=1)
    T_obs   = r3_var / max(eps_var, 1e-15)

    if verbose:
        print(f"  Observed statistic T = {T_obs:.6f}")
        print(f"  Starting wild bootstrap (B={B})...")

    # ── Wild bootstrap ──────────────────────────────────────────
    rng   = np.random.default_rng(seed)
    T_boot = np.empty(B)

    for b in range(B):
        if verbose and (b % 100 == 0):
            print(f"  Bootstrap {b+1}/{B}...")

        # Rademacher perturbation: +-1 at 50% each
        V        = rng.choice([-1.0, 1.0], size=len(resid_c))
        eps3_star = r3_obs + resid_c * V

        # Re-estimate r_hat*_3(Z)
        r3_star = local_linear(m0.Z_, eps3_star, h=m0.h_r3_)

        r3_star_var  = np.var(r3_star, ddof=1)
        eps3_var_star = np.var(eps3_star, ddof=1)
        T_boot[b]    = r3_star_var / max(eps3_var_star, 1e-15)

        gc.collect()

    p_value = float((T_boot >= T_obs).mean())

    if verbose:
        print(f"\n  p-value = {p_value:.4f}  "
              f"({'Significant (heterogeneous inefficiency)' if p_value < 0.05 else 'Not significant (uniform inefficiency)'})")

    return {
        'statistic': T_obs,
        'p_value'  : p_value,
        'r3_hat'   : r3_obs,
        'T_boot'   : T_boot,
        'B'        : B,
    }
