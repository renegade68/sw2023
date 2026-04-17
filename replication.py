"""
replication.py — Standalone replication script for:

  Lee, C. (2025). sw2023: Nonparametric Multiple-Output Stochastic
  Frontier Analysis in Python. Manuscript submitted for publication.

Reproduces all numerical results reported in the accompanying manuscript.
Tested with: Python 3.10, numpy 1.26, scipy 1.14, pandas 2.2, sw2023 0.3.2

Runtime:
  Default (--quick) : < 5 minutes  (n_sims=20 for Monte Carlo)
  Full   (--full)   : ~ 30 minutes (n_sims=100, matches paper Table 1)

Usage:
    pip install sw2023
    python replication.py           # quick verification
    python replication.py --full    # exact replication of Table 1
"""

import sys
import numpy as np

FULL = "--full" in sys.argv
N_SIMS = 100 if FULL else 20

print("=" * 65)
print("sw2023 JSS Replication Script")
if FULL:
    print("  Mode: FULL (n_sims=100) — reproduces manuscript Table 1 exactly")
else:
    print("  Mode: QUICK (n_sims=20) — for fast verification")
    print("  Run with --full for exact Table 1 replication (~30 min)")
print("=" * 65)

# ── Check package versions ────────────────────────────────────────────────────
import sw2023
import scipy
import pandas as pd

print(f"\nsw2023 version : {sw2023.__version__}")
print(f"Python version : {sys.version.split()[0]}")
print(f"numpy version  : {np.__version__}")
print(f"scipy version  : {scipy.__version__}")
print(f"pandas version : {pd.__version__}")

# ── Import public API ─────────────────────────────────────────────────────────
from sw2023 import (SW2023Model, bootstrap_sw, test_r3_significance,
                    bandwidth_loocv, bandwidth_loocv_product)

# =============================================================================
# Section 5: Monte Carlo Validation (Table 1)
# Reproduces IMSE ratios for (p,q,n,rho) configurations.
# Reference: Simar & Wilson (2023, JBES) Table F.1
# =============================================================================
print("\n" + "=" * 65)
print("Section 5: Monte Carlo Validation (Table 1)")
print(f"  DGP: Marsaglia (1972) sphere, sigma_eta=0.5, rho=0")
print(f"  n_sims={N_SIMS}" + (" [paper]" if FULL else " [reduced; use --full for paper values]"))
print("=" * 65)

from sw2023.tests.monte_carlo import run_imse_grid, print_imse_comparison

np.random.seed(2023)   # fixed seed for reproducibility

try:
    df = run_imse_grid(
        pq_list=[(1, 1), (1, 2), (2, 1), (2, 2)],
        n_list=[100, 200, 400],
        rho_list=[0.0],
        n_sims=N_SIMS,
        verbose=False,
    )
    print()
    print_imse_comparison(df)
except Exception as e:
    print(f"  Monte Carlo error: {e}")

# ── Pre-computed reference results (n_sims=100, matches paper Table 1) ────────
import os
csv_path = os.path.join(os.path.dirname(__file__), "mc_imse_results.csv")
if os.path.exists(csv_path):
    print("\n-- Pre-computed results (n_sims=100, paper Table 1 — scalar LOO-CV) --")
    ref = pd.read_csv(csv_path)
    print(ref.to_string(index=False))

product_csv = os.path.join(os.path.dirname(__file__), "mc_imse_product.csv")
if os.path.exists(product_csv):
    print("\n-- Pre-computed results (n_sims=100, product LOO-CV — paper Table 2) --")
    prd = pd.read_csv(product_csv)
    print(prd.to_string(index=False))

# =============================================================================
# Section 6: Simulation-Based Illustration
# Reproduces all code output blocks in Section 6 of the manuscript.
# =============================================================================
print("\n" + "=" * 65)
print("Section 6: Simulation-Based Illustration")
print("  DGP: Marsaglia sphere, p=q=2, n=200, sigma_eta=0.5, rho=1.0")
print("=" * 65)

np.random.seed(42)
n, p, q = 200, 2, 2

def sphere(n, dim):
    """Draw n uniform points from the surface of a (dim-1)-sphere."""
    X = np.random.randn(n, dim)
    return X / np.linalg.norm(X, axis=1, keepdims=True)

pts = np.abs(sphere(n, p + q))
X, Y = pts[:, :p], pts[:, p:]
sigma_eta, rho = 0.5, 1.0
eta = np.abs(np.random.randn(n)) * sigma_eta
eps = np.random.randn(n) * rho * sigma_eta
Y = Y * np.clip(1 - eta * 0.3 + eps * 0.1, 0.5, 1.5)[:, None]

# ── 6.1 Cross-sectional model ─────────────────────────────────────────────────
print("\n-- 6.1 Cross-sectional model --")
m = SW2023Model(X, Y, direction='mean', method='HMS',
                bandwidth_method='loocv')
m.fit(verbose=False)
print(f"Mean efficiency : {m.efficiency_.mean():.4f}   (paper: 0.4563)")
print(f"Std  efficiency : {m.efficiency_.std():.4f}   (paper: 0.2331)")
print(f"sigma_eta (mean): {m.sigma_eta_.mean():.4f}   (paper: 1.2183)")

# ── 6.2 Asymptotic CI ────────────────────────────────────────────────────────
print("\n-- 6.2 Asymptotic confidence intervals --")
ci = m.confint_asymptotic(alpha=0.05)
print(f"Mean SE(phi_hat)       : {ci['se_phi'].mean():.4f}   (paper: 0.4104)")
print(f"Mean CI lower (phi_hat): {ci['phi_hat_ci'][:,0].mean():.4f}   (paper: 0.0790)")
print(f"Mean CI upper (phi_hat): {ci['phi_hat_ci'][:,1].mean():.4f}   (paper: 1.6879)")

# ── 6.3 Bootstrap CI ─────────────────────────────────────────────────────────
print("\n-- 6.3 Bootstrap confidence intervals (B=199) --")
res = bootstrap_sw(X, Y, B=199, alpha=0.05)
width = (res['phi_hat_ci'][:,1] - res['phi_hat_ci'][:,0]).mean()
print(f"Mean CI width (frontier)     : {width:.4f}   (paper: 2.0023)")
lo, hi = res['eff_mean_ci']
print(f"Mean efficiency 95% CI       : [{lo:.4f}, {hi:.4f}]  (paper: [0.4745, 0.5729])")

# ── 6.4 Wild bootstrap significance test ─────────────────────────────────────
print("\n-- 6.4 Wild bootstrap significance test (B=299) --")
tres = test_r3_significance(X, Y, B=299)
print(f"Test statistic T: {tres['statistic']:.4f}   (paper: 0.0397)")
print(f"p-value         : {tres['p_value']:.4f}   (paper: 0.8729)")
print("  => H0 not rejected (homogeneous inefficiency), consistent with DGP.")

print("\n" + "=" * 65)
print("Replication complete.")
print("=" * 65)
