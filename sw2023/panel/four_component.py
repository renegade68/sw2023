"""
4-Component Panel Stochastic Frontier Model
SW(2023) rotation transformation + Colombi et al.(2014) / Tsionas & Kumbhakar(2014) 4-component decomposition

Model structure:
  U_it = φ(Z_it) + ||d||·v_it - ||d||·u_it + ||d||·α_i - ||d||·μ_i

  v_it  ~ N(0, σ_v²(Z_it))     : transient noise (time-varying, symmetric)
  u_it  ~ N⁺(0, σ_u²(Z_it))    : transient inefficiency (time-varying, one-sided)
  α_i   ~ N(0, σ_α²)            : individual heterogeneity (time-invariant, symmetric)
  μ_i   ~ N⁺(0, σ_μ²)           : persistent inefficiency (time-invariant, one-sided)

Identification strategy (Colombi et al. 2014 / Tsionas & Kumbhakar 2014):
  - Within-individual variation   → transient components (v_it, u_it) identified
  - Between-individual variation  → persistent components (α_i, μ_i) identified

Estimation procedure:
  Step 1: Pooled LLLS estimates φ̂(Z_it) → residuals ε̂_it
  Step 2: Separate within-individual residuals (w_it) / between-individual residuals (b_i)
  Step 3: Third moment of w_it → σ̂_u (transient inefficiency)
  Step 4: Third moment of b_i  → σ̂_μ (persistent inefficiency)
  Step 5: Compute individual efficiency indices via JLMS
"""

import numpy as np
import pandas as pd
from scipy.stats import norm as scipy_norm

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.transform  import make_direction, transform
from core.frontier   import estimate_moments, local_linear, _bandwidth_silverman
from core.decompose  import (A3_PLUS, A3_MINUS,
                              estimate_sigma_eta, estimate_sigma_eps,
                              estimate_frontier)
from core.preprocess import preprocess

PI = np.pi


# ─────────────────────────────────────────────────────────────
# Helper: JLMS efficiency (general purpose)
# ─────────────────────────────────────────────────────────────
def _jlms(composite_resid, sigma_u, sigma_v):
    """
    JLMS conditional expectation of inefficiency.

    E[u | ε] = μ* + σ* × φ(μ*/σ*) / Φ(μ*/σ*)
    where ε = v - u, μ* = -ε·σ_u²/(σ_u²+σ_v²), σ*² = σ_u²σ_v²/(σ_u²+σ_v²)

    Edge case handling:
      - σ_u = σ_v = 0: u_hat = 0
      - σ_v = 0 (σ_u > 0): σ* = 0 → E[u|ε] = max(0, -ε)
      - σ_u = 0 (σ_v > 0): E[u|ε] = 0

    Parameters
    ----------
    composite_resid : (n,)  ε̂ = v - u (or b_i = α_i - μ_i)
    sigma_u         : (n,)  σ_u (one-sided component standard deviation)
    sigma_v         : (n,)  σ_v (two-sided component standard deviation)

    Returns
    -------
    eff  : (n,) exp(-E[u|ε]) ∈ (0,1]
    u_hat: (n,) E[u|ε]
    """
    eps  = np.asarray(composite_resid, dtype=float)
    su   = np.asarray(sigma_u, dtype=float)
    sv   = np.asarray(sigma_v, dtype=float)
    su2  = su ** 2
    sv2  = sv ** 2
    s2   = su2 + sv2

    # Case 1: σ_u = σ_v = 0 → u_hat = 0
    # Case 2: σ_v = 0, σ_u > 0 → σ* = 0, E[u|ε] = max(0, -ε)
    # Case 3: σ_u = 0, σ_v > 0 → E[u|ε] = 0
    # Case 4: general case

    u_hat = np.zeros_like(eps)

    # Case 2: sigma_v ≈ 0, sigma_u > 0
    case2 = (sv < 1e-10) & (su > 1e-10)
    u_hat[case2] = np.maximum(0.0, -eps[case2])

    # Case 4: general case (su > 0, sv > 0)
    case4 = (su > 1e-10) & (sv > 1e-10)
    if case4.any():
        s2_4   = s2[case4]
        mu_s   = -eps[case4] * su2[case4] / s2_4
        sig_s  = np.sqrt(su2[case4] * sv2[case4] / s2_4)
        ratio  = mu_s / sig_s
        pdf_r  = scipy_norm.pdf(ratio)
        cdf_r  = np.maximum(scipy_norm.cdf(ratio), 1e-15)
        u_hat[case4] = np.maximum(0.0, mu_s + sig_s * pdf_r / cdf_r)

    eff = np.exp(-u_hat)
    return eff, u_hat


# ─────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────
class PanelSW2023:
    """
    4-Component Panel SFA with SW(2023) Rotation Transformation.

    Parameters
    ----------
    X          : (n, p)  inputs
    Y          : (n, q)  outputs
    firm_id    : (n,)    individual identifier (farm ID, etc.)
    time_id    : (n,)    time identifier (year, etc.)
    direction  : 'mean' | 'median' | array  direction vector
    method     : 'SVKZ' | 'HMS'
    h          : bandwidth (None = Silverman)
    log_transform : bool
    standardize   : bool
    """

    def __init__(self, X, Y, firm_id, time_id,
                 direction='mean', method='HMS', h=None,
                 log_transform=True, standardize=True):
        self.X_raw     = np.asarray(X, dtype=float)
        self.Y_raw     = np.asarray(Y, dtype=float)
        self.firm_id   = np.asarray(firm_id)
        self.time_id   = np.asarray(time_id)
        self.direction_spec = direction
        self.method    = method
        self.h         = h
        self.log_transform = log_transform
        self.standardize   = standardize
        self._fitted   = False

    # ── Main estimation ────────────────────────────────────────
    def fit(self, verbose=True):
        """Estimate the 4-component model."""

        # 0. Preprocessing
        self.X, self.Y, self.preprocess_info_ = preprocess(
            self.X_raw, self.Y_raw,
            log_transform=self.log_transform,
            standardize=self.standardize
        )
        X, Y = self.X, self.Y
        n, p = X.shape
        q    = Y.shape[1]

        firms  = self.firm_id
        times  = self.time_id
        uniq_firms = np.unique(firms)
        N = len(uniq_firms)
        T_vec = np.array([np.sum(firms == f) for f in uniq_firms])

        if verbose:
            print(f"4-Component Panel SW(2023) estimation start")
            print(f"  n={n}, p={p}, q={q}, N={N} individuals, "
                  f"mean T={T_vec.mean():.1f}")
            print(f"  method={self.method}")

        # 1. Direction vector + rotation transformation
        self.d_ = make_direction(X, Y, method=self.direction_spec)
        self.norm_d_ = np.linalg.norm(self.d_)
        self.Z_, self.U_, self.R_ = transform(X, Y, self.d_)

        if verbose:
            print(f"  direction vector d = {np.round(self.d_, 3)}")
            print(f"  Z shape: {self.Z_.shape}")

        # 2. Pooled LLLS → φ̂(Z_it), composite residuals ε̂_it
        if verbose:
            print(f"  [Step 1] Estimating pooled LLLS...")
        moments = estimate_moments(self.Z_, self.U_, h=self.h)
        self.r1_   = moments['r1']
        self.h_    = moments['h']
        eps_hat    = moments['eps']   # ε̂_it = U_it - r̂_1(Z_it)

        # 3. Within / Between separation
        if verbose:
            print(f"  [Step 2] Separating within-individual / between-individual residuals...")

        # Individual mean residuals (between: b_i)
        eps_mean_firm = np.zeros(n)
        for f in uniq_firms:
            mask = firms == f
            eps_mean_firm[mask] = eps_hat[mask].mean()

        # Within residuals (transient component)
        w_it = eps_hat - eps_mean_firm    # (n,)  transient: v_it - u_it (centered)

        # Between residuals (persistent component) - one value per individual
        firm_idx   = {f: i for i, f in enumerate(uniq_firms)}
        b_i_vals   = np.array([eps_hat[firms == f].mean() for f in uniq_firms])
        # Individual mean Z (auxiliary variable for between estimation)
        Z_mean_firm = np.array([self.Z_[firms == f].mean(axis=0)
                                 for f in uniq_firms])

        # 4. Transient component: third moment of w_it → σ̂_u(z)
        if verbose:
            print(f"  [Step 3] Estimating transient inefficiency (within)...")

        r2_w = local_linear(self.Z_, w_it ** 2, h=self.h_)
        r3_w = local_linear(self.Z_, w_it ** 3, h=self.h_)

        self.sigma_u_ = estimate_sigma_eta(r3_w, method=self.method)
        self.sigma_v_ = estimate_sigma_eps(r2_w, self.sigma_u_)

        # 5. Persistent component: third moment of b_i → σ̂_μ(z̄_i)
        if verbose:
            print(f"  [Step 4] Estimating persistent inefficiency (between)...")

        h_between = _bandwidth_silverman(Z_mean_firm) if self.h is None else self.h_

        r2_b = local_linear(Z_mean_firm, b_i_vals ** 2, h=h_between)
        r3_b = local_linear(Z_mean_firm, b_i_vals ** 3, h=h_between)

        self.sigma_mu_firm_  = estimate_sigma_eta(r3_b, method=self.method)
        self.sigma_alp_firm_ = estimate_sigma_eps(r2_b, self.sigma_mu_firm_)

        # Expand individual-level values to observation level
        sigma_mu_n  = np.array([self.sigma_mu_firm_[firm_idx[f]]
                                 for f in firms])
        sigma_alp_n = np.array([self.sigma_alp_firm_[firm_idx[f]]
                                 for f in firms])
        b_i_n = np.array([b_i_vals[firm_idx[f]] for f in firms])

        # 6. Frontier estimation
        #    r̂_1(z) = φ(z) - ||d||(μ_u + μ_μ)
        #    φ̂(z)   = r̂_1(z) + ||d||·(μ_u(z) + μ_μ(z))
        mu_u_n  = np.sqrt(2 / PI) * self.sigma_u_    # transient mean inefficiency
        mu_mu_n = np.sqrt(2 / PI) * sigma_mu_n       # persistent mean inefficiency
        self.phi_hat_ = self.r1_ + self.norm_d_ * (mu_u_n + mu_mu_n)

        # 7. JLMS efficiency indices
        if verbose:
            print(f"  [Step 5] Computing JLMS efficiency indices...")

        # Transient efficiency: TE_it = exp(-u_it)
        self.eff_transient_, self.u_hat_ = _jlms(
            w_it, self.sigma_u_, self.sigma_v_
        )

        # Persistent efficiency: PE_i = exp(-μ_i)  (individual level → expanded to observations)
        pe_firm, mu_hat_firm = _jlms(
            b_i_vals, self.sigma_mu_firm_, self.sigma_alp_firm_
        )
        self.eff_persistent_ = np.array([pe_firm[firm_idx[f]] for f in firms])
        self.mu_hat_         = np.array([mu_hat_firm[firm_idx[f]] for f in firms])

        # Overall efficiency: OE_it = TE_it × PE_i
        self.efficiency_ = self.eff_transient_ * self.eff_persistent_

        # Store additional information
        self.eps_hat_    = eps_hat
        self.w_it_       = w_it
        self.b_i_n_      = b_i_n
        self.uniq_firms_ = uniq_firms
        self._fitted     = True

        if verbose:
            print(f"  Estimation complete.")
            print(f"  Mean overall efficiency    : {np.nanmean(self.efficiency_):.4f}")
            print(f"  Mean transient efficiency  : {np.nanmean(self.eff_transient_):.4f}")
            print(f"  Mean persistent efficiency : {np.nanmean(self.eff_persistent_):.4f}")
            ws_w = (r3_w > 0).mean() * 100
            ws_b = (r3_b > 0).mean() * 100
            print(f"  Wrong skewness(within)  : {ws_w:.1f}%")
            print(f"  Wrong skewness(between) : {ws_b:.1f}%")

        return self

    # ── Results summary ────────────────────────────────────────
    def summary(self):
        """Print estimation results summary and return DataFrame."""
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")

        print("=" * 55)
        print("4-Component Panel SW(2023) Estimation Results")
        print("=" * 55)
        print(f"Observations: {len(self.efficiency_)}, "
              f"Individuals: {len(self.uniq_firms_)}")
        print(f"Method: {self.method}")

        df = pd.DataFrame({
            'firm_id'        : self.firm_id,
            'time_id'        : self.time_id,
            'U'              : self.U_,
            'phi_hat'        : self.phi_hat_,
            'efficiency'     : self.efficiency_,
            'eff_transient'  : self.eff_transient_,
            'eff_persistent' : self.eff_persistent_,
            'u_hat'          : self.u_hat_,
            'mu_hat'         : self.mu_hat_,
            'sigma_u'        : self.sigma_u_,
            'sigma_mu'       : self.b_i_n_,
        })

        for col in ['efficiency', 'eff_transient', 'eff_persistent']:
            vals = df[col]
            print(f"\n{col}:")
            print(f"  mean={vals.mean():.4f}  median={vals.median():.4f}  "
                  f"std={vals.std():.4f}  "
                  f"[{vals.min():.4f}, {vals.max():.4f}]")

        print("=" * 55)
        return df

    def results_dataframe(self):
        """Return full results as a DataFrame."""
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")
        return pd.DataFrame({
            'firm_id'        : self.firm_id,
            'time_id'        : self.time_id,
            'U'              : self.U_,
            'phi_hat'        : self.phi_hat_,
            'eps_hat'        : self.eps_hat_,
            'w_it'           : self.w_it_,
            'sigma_u'        : self.sigma_u_,
            'sigma_v'        : self.sigma_v_,
            'u_hat'          : self.u_hat_,
            'mu_hat'         : self.mu_hat_,
            'efficiency'     : self.efficiency_,
            'eff_transient'  : self.eff_transient_,
            'eff_persistent' : self.eff_persistent_,
        })
