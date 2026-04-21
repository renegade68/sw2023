"""
Result classes for SW2023 estimation outputs.

Each class wraps the numerical arrays returned by estimation methods and
provides human-readable __repr__ and summary() output.

Classes
-------
ConfintResult        : asymptotic confidence intervals (confint_asymptotic)
BootstrapResult      : bootstrap confidence intervals (SW2023Model.bootstrap)
SignificanceTestResult: wild bootstrap significance test (test_r3_significance)
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# ConfintResult
# ─────────────────────────────────────────────────────────────────────────────

class ConfintResult:
    """
    Asymptotic confidence intervals from SW2023Model.confint_asymptotic().

    Attributes
    ----------
    phi_hat_ci : ndarray, shape (n, 2)
        Lower and upper confidence bounds for the frontier phi_hat(z).
    r1_ci : ndarray, shape (n, 2)
        CI for the first conditional moment r_1(z).
    r3_ci : ndarray, shape (n, 2)
        CI for the third conditional moment r_3(z).
    se_phi : ndarray, shape (n,)
        Standard errors for phi_hat(z).
    se_r1 : ndarray, shape (n,)
        Standard errors for r_1(z).
    se_r3 : ndarray, shape (n,)
        Standard errors for r_3(z).
    alpha : float
        Significance level used (1 - alpha is the confidence level).
    """

    def __init__(self, phi_hat_ci, r1_ci, r3_ci,
                 se_phi, se_r1, se_r3, alpha):
        self.phi_hat_ci = np.asarray(phi_hat_ci)
        self.r1_ci      = np.asarray(r1_ci)
        self.r3_ci      = np.asarray(r3_ci)
        self.se_phi     = np.asarray(se_phi)
        self.se_r1      = np.asarray(se_r1)
        self.se_r3      = np.asarray(se_r3)
        self.alpha      = float(alpha)

    def __repr__(self):
        n    = len(self.se_phi)
        conf = int(round((1 - self.alpha) * 100))
        w    = float(np.mean(self.phi_hat_ci[:, 1] - self.phi_hat_ci[:, 0]))
        return (
            f"ConfintResult(n={n}, conf={conf}%, "
            f"mean_se_phi={self.se_phi.mean():.4f}, "
            f"mean_ci_width={w:.4f})"
        )

    def summary(self):
        """Print a formatted summary of the asymptotic confidence intervals."""
        n    = len(self.se_phi)
        conf = int(round((1 - self.alpha) * 100))
        w    = self.phi_hat_ci[:, 1] - self.phi_hat_ci[:, 0]
        print("=" * 55)
        print(f"Asymptotic Confidence Intervals  ({conf}%,  n={n})")
        print("=" * 55)
        print(f"{'':30s}  {'Mean':>8}  {'Std':>8}")
        print("-" * 55)
        print(f"{'SE  phi_hat':30s}  {self.se_phi.mean():8.4f}  "
              f"{self.se_phi.std():8.4f}")
        print(f"{'SE  r1':30s}  {self.se_r1.mean():8.4f}  "
              f"{self.se_r1.std():8.4f}")
        print(f"{'SE  r3':30s}  {self.se_r3.mean():8.4f}  "
              f"{self.se_r3.std():8.4f}")
        print(f"{'CI width  phi_hat':30s}  {w.mean():8.4f}  "
              f"{w.std():8.4f}")
        print(f"{'CI lower  phi_hat (mean)':30s}  "
              f"{self.phi_hat_ci[:,0].mean():8.4f}")
        print(f"{'CI upper  phi_hat (mean)':30s}  "
              f"{self.phi_hat_ci[:,1].mean():8.4f}")
        print("=" * 55)
        print("Use .phi_hat_ci, .r1_ci, .r3_ci, .se_phi for arrays.")


# ─────────────────────────────────────────────────────────────────────────────
# BootstrapResult
# ─────────────────────────────────────────────────────────────────────────────

class BootstrapResult:
    """
    Bootstrap confidence intervals from SW2023Model.bootstrap() or
    bootstrap_sw() / bootstrap_panel().

    Attributes
    ----------
    phi_hat_point : ndarray, shape (n,)
        Point estimates of the frontier phi_hat(z).
    phi_hat_ci : ndarray, shape (n, 2)
        Bootstrap CI for phi_hat(z).
    eff_mean_point : float
        Point estimate of the mean efficiency.
    eff_mean_ci : ndarray, shape (2,)
        Bootstrap CI for the mean efficiency.
    eff_individual_point : ndarray, shape (n,)
        Point estimates of individual efficiency scores.
    eff_individual_ci : ndarray, shape (n, 2)
        Bootstrap CI for individual efficiency scores.
    sigma_eta_point : ndarray, shape (n,)
        Point estimates of sigma_eta(z).
    sigma_eta_ci : ndarray, shape (n, 2)
        Bootstrap CI for sigma_eta(z).
    B : int
        Number of bootstrap draws used.
    alpha : float
        Significance level (1 - alpha is the confidence level).
    n_fail : int
        Number of bootstrap iterations that failed (should be 0).
    """

    def __init__(self, phi_hat_point, phi_hat_ci,
                 eff_mean_point, eff_mean_ci,
                 eff_individual_point, eff_individual_ci,
                 sigma_eta_point, sigma_eta_ci,
                 B, alpha, n_fail=0):
        self.phi_hat_point        = np.asarray(phi_hat_point)
        self.phi_hat_ci           = np.asarray(phi_hat_ci)
        self.eff_mean_point       = float(eff_mean_point)
        self.eff_mean_ci          = np.asarray(eff_mean_ci)
        self.eff_individual_point = np.asarray(eff_individual_point)
        self.eff_individual_ci    = np.asarray(eff_individual_ci)
        self.sigma_eta_point      = np.asarray(sigma_eta_point)
        self.sigma_eta_ci         = np.asarray(sigma_eta_ci)
        self.B      = int(B)
        self.alpha  = float(alpha)
        self.n_fail = int(n_fail)

    def __repr__(self):
        conf = int(round((1 - self.alpha) * 100))
        lo, hi = self.eff_mean_ci
        fail  = f", n_fail={self.n_fail}" if self.n_fail > 0 else ""
        return (
            f"BootstrapResult(B={self.B}, conf={conf}%, "
            f"mean_eff={self.eff_mean_point:.4f} "
            f"[{lo:.4f}, {hi:.4f}]{fail})"
        )

    def summary(self):
        """Print a formatted summary of bootstrap confidence intervals."""
        n    = len(self.phi_hat_point)
        conf = int(round((1 - self.alpha) * 100))
        phi_w = float(np.mean(
            self.phi_hat_ci[:, 1] - self.phi_hat_ci[:, 0]))
        eff_w = float(np.mean(
            self.eff_individual_ci[:, 1] - self.eff_individual_ci[:, 0]))
        lo, hi = self.eff_mean_ci

        print("=" * 55)
        print(f"Bootstrap Confidence Intervals  (B={self.B}, {conf}%, n={n})")
        if self.n_fail > 0:
            print(f"  Warning: {self.n_fail} iterations failed")
        print("=" * 55)
        print(f"Mean efficiency  (point)    : {self.eff_mean_point:.4f}")
        print(f"Mean efficiency  ({conf}% CI)  : [{lo:.4f}, {hi:.4f}]")
        print(f"Mean CI width — phi_hat     : {phi_w:.4f}")
        print(f"Mean CI width — eff (indiv) : {eff_w:.4f}")
        print("=" * 55)
        print("Use .phi_hat_ci, .eff_individual_ci, .eff_mean_ci for arrays.")


# ─────────────────────────────────────────────────────────────────────────────
# SignificanceTestResult
# ─────────────────────────────────────────────────────────────────────────────

class SignificanceTestResult:
    """
    Wild bootstrap significance test result from test_r3_significance().

    Tests H0: E(epsilon^3 | Z) = const  (spatially uniform inefficiency)
    against H1: E(epsilon^3 | Z) != const (heterogeneous inefficiency).

    Attributes
    ----------
    statistic : float
        Observed test statistic T (variance of r_hat_3 relative to Var(eps^3)).
    p_value : float
        Bootstrap p-value.  Small values (< 0.05) indicate heterogeneous
        inefficiency; large values indicate the null is not rejected.
    r3_hat : ndarray, shape (n,)
        Estimated third conditional moment r_hat_3(Z) from the original sample.
    T_boot : ndarray, shape (B,)
        Bootstrap distribution of the test statistic.
    B : int
        Number of bootstrap draws used.
    """

    def __init__(self, statistic, p_value, r3_hat, T_boot, B):
        self.statistic = float(statistic)
        self.p_value   = float(p_value)
        self.r3_hat    = np.asarray(r3_hat)
        self.T_boot    = np.asarray(T_boot)
        self.B         = int(B)

    def __repr__(self):
        sig   = self.p_value < 0.05
        label = "significant" if sig else "not significant"
        return (
            f"SignificanceTestResult(T={self.statistic:.4f}, "
            f"p_value={self.p_value:.4f}, B={self.B}, {label})"
        )

    def summary(self):
        """Print a formatted summary of the significance test."""
        sig   = self.p_value < 0.05
        label = ("Reject H0 — heterogeneous inefficiency"
                 if sig else
                 "Do not reject H0 — uniform inefficiency")
        print("=" * 55)
        print("Wild Bootstrap Significance Test (PSVKZ 2024)")
        print("  H0: E(eps^3 | Z) = const  (uniform inefficiency)")
        print("=" * 55)
        print(f"Test statistic T  : {self.statistic:.6f}")
        print(f"p-value           : {self.p_value:.4f}  (B={self.B})")
        print(f"Bootstrap T range : [{self.T_boot.min():.4f}, "
              f"{self.T_boot.max():.4f}]")
        print(f"Conclusion        : {label}")
        print("=" * 55)
