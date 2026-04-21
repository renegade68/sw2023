"""
SW2023Model: Main estimation class

Usage example:
    model = SW2023Model(X, Y, direction='mean', method='HMS')
    model.fit()
    print(model.summary())
"""

import numpy as np
import pandas as pd

from .transform  import make_direction, transform, inverse_transform
from .frontier   import estimate_moments, local_linear, compute_leverages, _compute_K_full
from .decompose  import (estimate_sigma_eta, estimate_sigma_eps,
                         estimate_frontier, jlms_efficiency)
from .preprocess import preprocess
from .results    import ConfintResult, BootstrapResult


class SW2023Model:
    """
    Simar & Wilson (2023) nonparametric multiple-output stochastic frontier model.

    Parameters
    ----------
    X         : array-like (n, p) inputs
    Y         : array-like (n, q) outputs
    direction : 'mean' | 'median' | array (p+q,) direction vector
    method    : 'SVKZ' | 'HMS'  inefficiency estimation method
    h         : bandwidth (if None, Silverman rule is applied automatically)
    """

    def __init__(self, X, Y, direction='mean', method='HMS', h=None,
                 log_transform=True, standardize=True,
                 bandwidth_method='silverman'):
        self.X_raw = np.asarray(X, dtype=float)
        self.Y_raw = np.asarray(Y, dtype=float)
        self.direction_spec  = direction
        self.method          = method
        self.h               = h
        self.log_transform   = log_transform
        self.standardize     = standardize
        self.bandwidth_method = bandwidth_method
        self._fitted = False

    def __repr__(self):
        n = getattr(self, 'X_raw', None)
        n = len(n) if n is not None else '?'
        p = self.X_raw.shape[1] if hasattr(self, 'X_raw') else '?'
        q = self.Y_raw.shape[1] if hasattr(self, 'Y_raw') else '?'
        if not self._fitted:
            return (f"SW2023Model(n={n}, p={p}, q={q}, "
                    f"method='{self.method}', not fitted)")
        eff_mean = float(np.nanmean(self.efficiency_))
        ws_pct   = float((self.r3_ > 0).mean() * 100)
        return (f"SW2023Model(n={n}, p={p}, q={q}, "
                f"method='{self.method}', "
                f"mean_eff={eff_mean:.4f}, "
                f"wrong_skew={ws_pct:.1f}%)")

    def fit(self, verbose=True):
        """Estimate the SW(2023) nonparametric stochastic frontier model.

        Executes Steps 1--10 of Simar & Wilson (2023): data preprocessing,
        direction-vector rotation, three-moment LLLS regressions with
        LOO-CV bandwidth selection, sigma_eta estimation (SVKZ or HMS),
        frontier recovery, and JLMS individual efficiency scoring.

        Parameters
        ----------
        verbose : bool, default True
            If True, print progress messages during estimation.

        Returns
        -------
        self : SW2023Model
            Fitted model.  Key attributes set after calling fit():

            phi_hat\_ : ndarray, shape (n,)
                Estimated frontier values at each observation.
            efficiency\_ : ndarray, shape (n,)
                JLMS efficiency scores exp(-eta_hat) in (0, 1].
            sigma_eta\_ : ndarray, shape (n,)
                Estimated inefficiency std dev at each observation.
            r1\\_, r2\\_, r3\\_ : ndarray, shape (n,)
                Estimated first, second, and third conditional moments.
            h\\_ : ndarray, shape (n,)
                Hat-matrix leverages for r1 regression (used in CI).
        """
        # Preprocessing
        self.X, self.Y, self.preprocess_info_ = preprocess(
            self.X_raw, self.Y_raw,
            log_transform=self.log_transform,
            standardize=self.standardize
        )
        X, Y = self.X, self.Y
        n, p = X.shape
        q = Y.shape[1]

        if verbose:
            print(f"SW(2023) estimation started: n={n}, p={p}, q={q}, method={self.method}")
            if self.log_transform:
                print(f"  Data preprocessing: log transform + standardization" if self.standardize
                      else "  Data preprocessing: log transform")

        # Step 1: Direction vector
        self.d_ = make_direction(X, Y, method=self.direction_spec)
        self.norm_d_ = np.linalg.norm(self.d_)

        if verbose:
            print(f"  Direction vector d = {np.round(self.d_, 4)}")

        # Step 2: Rotation transform
        self.Z_, self.U_, self.R_ = transform(X, Y, self.d_)

        if verbose:
            print(f"  Rotation transform complete: Z {self.Z_.shape}, U {self.U_.shape}")

        # Steps 3-6: Conditional moment estimation
        if verbose:
            print(f"  Running local linear regression (n={n} points, this may take a while)...")

        moments = estimate_moments(self.Z_, self.U_, h=self.h,
                                   bandwidth_method=self.bandwidth_method)
        self.r1_   = moments['r1']
        self.r2_   = moments['r2']
        self.r3_   = moments['r3']
        self.eps_  = moments['eps']
        self.h_    = moments['h']
        self.h_r2_ = moments['h_r2']
        self.h_r3_ = moments['h_r3']

        if verbose:
            print(f"  Bandwidth h(r1) = {np.round(self.h_, 4)}")
            if self.bandwidth_method == 'loocv':
                print(f"  Bandwidth h(r2) = {np.round(self.h_r2_, 4)}")
                print(f"  Bandwidth h(r3) = {np.round(self.h_r3_, 4)}")

        # Step 7: Estimate sigma_eta
        self.sigma_eta_ = estimate_sigma_eta(self.r3_, method=self.method)

        # Step 8: Estimate sigma_eps
        self.sigma_eps_ = estimate_sigma_eps(self.r2_, self.sigma_eta_)

        # Step 9: Frontier estimation
        self.phi_hat_, self.mu_eta_ = estimate_frontier(
            self.r1_, self.sigma_eta_, self.norm_d_
        )

        # Step 10: JLMS efficiency
        self.efficiency_, self.eta_hat_ = jlms_efficiency(
            self.U_, self.phi_hat_, self.sigma_eta_, self.sigma_eps_
        )

        self._fitted = True

        if verbose:
            print(f"  Estimation complete.")
            print(f"  Mean efficiency: {np.nanmean(self.efficiency_):.4f}")
            print(f"  Wrong skewness ratio: "
                  f"{(self.r3_ > 0).mean()*100:.1f}%")

        return self

    def summary(self):
        """Print a summary of estimation results."""
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")

        eff = self.efficiency_
        eta = self.eta_hat_
        s_eta = self.sigma_eta_
        s_eps = self.sigma_eps_

        df = pd.DataFrame({
            'efficiency'  : eff,
            'eta_hat'     : eta,
            'sigma_eta'   : s_eta,
            'sigma_eps'   : s_eps,
            'mu_eta'      : self.mu_eta_,
            'phi_hat'     : self.phi_hat_,
        })

        print("=" * 50)
        print("SW(2023) Estimation Results Summary")
        print("=" * 50)
        print(f"Number of observations : {len(eff)}")
        print(f"Method                 : {self.method}")
        print(f"Direction vector       : {np.round(self.d_, 4)}")
        print(f"Bandwidth              : {np.round(self.h_, 4)}")
        print()
        print(df[['efficiency', 'eta_hat', 'sigma_eta', 'sigma_eps']].describe().round(4))
        print("=" * 50)

        return df

    def confint_asymptotic(self, alpha=0.05):
        """
        Asymptotic normal confidence intervals (SW 2023 CLT + delta method).

        SW(2023) Eq.(3.8) CLT:
          (nh^{d-1})^{1/2} (r̂_j(z) − r_j(z)) →^L N(0, s²_j(z))

        Hat-matrix based variance estimation:
          AVar(r̂_1(z_i)) ≈ h_ii(r1) × r̂_2(z_i)
          AVar(r̂_3(z_i)) ≈ h_ii(r3) × V̂ar(ε³ | Z=z_i)

        φ̂(z) = r̂_1(z) + ||d|| √(2/π) σ̂_η(z) delta method:
          ∂φ/∂r1 = 1
          ∂φ/∂r3 = −||d|| √(2/π) / (3 × A3_PLUS × σ̂²_η)   [when r3 ≤ 0]

        Note: Asymptotic CI is valid when n is sufficiently large.
              For small samples, bootstrap_sw() CI is more appropriate.

        Parameters
        ----------
        alpha : significance level (1-alpha confidence interval)

        Returns
        -------
        dict:
            phi_hat_ci  : (n, 2)
            r1_ci       : (n, 2)
            r3_ci       : (n, 2)
            se_phi      : (n,) standard error
            se_r1       : (n,)
            se_r3       : (n,)
            alpha       : float
        """
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")

        from scipy.stats import norm as scipy_norm
        from .decompose import A3_PLUS

        z_crit = float(scipy_norm.ppf(1.0 - alpha / 2))

        # ── r̂_1 leverages ────────────────────────────────────────
        K1   = _compute_K_full(self.Z_, self.h_)
        hii1 = compute_leverages(K1, self.Z_)
        del K1

        var_r1 = hii1 * np.maximum(self.r2_, 1e-15)
        se_r1  = np.sqrt(var_r1)

        # ── r̂_3 leverages + conditional variance ─────────────────
        K3   = _compute_K_full(self.Z_, self.h_r3_)
        hii3 = compute_leverages(K3, self.Z_)
        del K3

        # Local estimation of Var(ε³|z): E((ε³ - r̂₃)²|z)
        resid3_sq = (self.eps_ ** 3 - self.r3_) ** 2
        var_eps3  = local_linear(self.Z_, resid3_sq, h=self.h_r3_)
        var_eps3  = np.maximum(var_eps3, 1e-15)

        var_r3 = hii3 * var_eps3
        se_r3  = np.sqrt(var_r3)

        # ── φ̂ delta method ────────────────────────────────────────
        # ∂φ/∂r3 = -||d|| × √(2/π) / (3 × A3_PLUS × σ̂²_η)
        sigma_sq   = np.where(self.sigma_eta_ > 1e-8,
                              self.sigma_eta_ ** 2, np.nan)
        dphi_dr3   = (-self.norm_d_ * np.sqrt(2.0 / np.pi)
                      / (3.0 * A3_PLUS * sigma_sq))
        # Wrong skewness points (unstable σ̂_η): treat gradient as 0
        dphi_dr3   = np.where(self.r3_ <= 0, dphi_dr3, 0.0)
        dphi_dr3   = np.nan_to_num(dphi_dr3, nan=0.0)

        var_phi = var_r1 + dphi_dr3 ** 2 * var_r3
        se_phi  = np.sqrt(np.maximum(var_phi, 0.0))

        # ── Construct CI ───────────────────────────────────────────
        return ConfintResult(
            phi_hat_ci=np.column_stack([
                self.phi_hat_ - z_crit * se_phi,
                self.phi_hat_ + z_crit * se_phi,
            ]),
            r1_ci=np.column_stack([
                self.r1_ - z_crit * se_r1,
                self.r1_ + z_crit * se_r1,
            ]),
            r3_ci=np.column_stack([
                self.r3_ - z_crit * se_r3,
                self.r3_ + z_crit * se_r3,
            ]),
            se_phi=se_phi,
            se_r1=se_r1,
            se_r3=se_r3,
            alpha=alpha,
        )

    def predict_at(self, Z_eval, U_eval=None):
        """
        Predict at new evaluation points Z_eval using the fitted model.

        Used during bootstrap CI computation: fit on bootstrap sample,
        then evaluate at the original Z points.

        Parameters
        ----------
        Z_eval : (m, d) evaluation points (rotated coordinates, same as training Z)
        U_eval : (m,) observed directional distances (if provided, efficiency is also computed)

        Returns
        -------
        dict: phi_hat, sigma_eta, sigma_eps, mu_eta, [efficiency, eta_hat]
        """
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")

        Z_eval = np.asarray(Z_eval, dtype=float)

        r1 = local_linear(self.Z_, self.U_,       h=self.h_,    eval_points=Z_eval)
        r2 = local_linear(self.Z_, self.eps_**2,  h=self.h_r2_, eval_points=Z_eval)
        r3 = local_linear(self.Z_, self.eps_**3,  h=self.h_r3_, eval_points=Z_eval)

        sigma_eta = estimate_sigma_eta(r3, method=self.method)
        sigma_eps = estimate_sigma_eps(r2, sigma_eta)
        phi_hat, mu_eta = estimate_frontier(r1, sigma_eta, self.norm_d_)

        out = {
            'phi_hat'   : phi_hat,
            'sigma_eta' : sigma_eta,
            'sigma_eps' : sigma_eps,
            'mu_eta'    : mu_eta,
            'r1': r1, 'r2': r2, 'r3': r3,
        }

        if U_eval is not None:
            U_eval = np.asarray(U_eval, dtype=float)
            eff_bc, eta_hat = jlms_efficiency(U_eval, phi_hat, sigma_eta, sigma_eps)
            out['efficiency'] = eff_bc
            out['eta_hat']    = eta_hat

        return out

    def results_dataframe(self):
        """Return estimation results as a DataFrame."""
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")
        return pd.DataFrame({
            'U'          : self.U_,
            'phi_hat'    : self.phi_hat_,
            'r1'         : self.r1_,
            'r2'         : self.r2_,
            'r3'         : self.r3_,
            'sigma_eta'  : self.sigma_eta_,
            'sigma_eps'  : self.sigma_eps_,
            'mu_eta'     : self.mu_eta_,
            'eta_hat'    : self.eta_hat_,
            'efficiency' : self.efficiency_,
        })

    # ── Plotting methods ──────────────────────────────────────────────────────

    def plot_efficiency(self, bins=30, figsize=(7, 4), ax=None):
        """
        Histogram of efficiency scores with mean and median lines.

        Parameters
        ----------
        bins    : int, default 30
        figsize : tuple, default (7, 4)
        ax      : matplotlib Axes or None
            If provided, plot into this axes; otherwise a new figure is created.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")
        from .visualize import plot_efficiency_dist
        return plot_efficiency_dist(
            self.efficiency_,
            title=f'Efficiency Distribution  (n={len(self.efficiency_)}, '
                  f'method={self.method})',
            bins=bins, figsize=figsize, ax=ax,
        )

    def plot_frontier(self, dim=0, figsize=(7, 5), ax=None):
        """
        Scatter plot of U vs Z[dim] with the estimated frontier phi_hat(Z).

        Points are colour-coded by efficiency score (red = inefficient,
        green = efficient).

        Parameters
        ----------
        dim     : int, default 0
            Index of the Z dimension to use as the x-axis.
        figsize : tuple, default (7, 5)
        ax      : matplotlib Axes or None

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")
        from .visualize import plot_frontier_1d
        return plot_frontier_1d(self, dim=dim, figsize=figsize, ax=ax)

    def plot_diagnostics(self, figsize=(12, 9)):
        """
        Comprehensive diagnostic panel (2×2 layout).

        Panels
        ------
        (top-left)     Efficiency distribution
        (top-right)    Efficiency ranking (caterpillar)
        (bottom-left)  Residual distribution (eps = U − r̂₁)
        (bottom-right) Wrong-skewness diagnostic (r̂₃ sign)

        Parameters
        ----------
        figsize : tuple, default (12, 9)

        Returns
        -------
        fig : matplotlib Figure
        """
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")
        from .visualize import plot_diagnostics as _plot_diag
        return _plot_diag(self, figsize=figsize)

    def bootstrap(self, B=200, alpha=0.05,
                  bandwidth_method='silverman',
                  seed=None, verbose=True):
        """
        Pairs bootstrap confidence intervals for this fitted model.

        Resamples (X, Y) with replacement, re-estimates the model using the
        same fixed direction vector and preprocessing, then evaluates each
        bootstrap estimate at the original data points.  Yields per-observation
        CIs for phi_hat(z), sigma_eta(z), and individual efficiency, as well
        as a CI for the mean efficiency.

        Parameters
        ----------
        B : int, default 200
            Number of bootstrap draws.
        alpha : float, default 0.05
            Significance level; produces (1 - alpha) confidence intervals.
        bandwidth_method : {'silverman', 'loocv'}, default 'silverman'
            Bandwidth selection for each bootstrap replicate.  'silverman' is
            strongly recommended here — LOO-CV per replicate is very slow.
        seed : int or None
            Random seed for reproducibility.
        verbose : bool, default True
            Print progress messages.

        Returns
        -------
        BootstrapResult
            Object with attributes phi_hat_ci, eff_mean_ci,
            eff_individual_ci, sigma_eta_ci, and a summary() method.

        See Also
        --------
        confint_asymptotic : faster asymptotic alternative (large n).
        """
        if not self._fitted:
            raise RuntimeError("Please run fit() first.")

        from .bootstrap import bootstrap_sw
        return bootstrap_sw(
            self.X_raw, self.Y_raw,
            B=B, alpha=alpha,
            direction=self.direction_spec,
            method=self.method,
            log_transform=self.log_transform,
            standardize=self.standardize,
            bandwidth_method=bandwidth_method,
            seed=seed,
            verbose=verbose,
        )
