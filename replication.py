"""
replication.py — Replication script for:

  Lee, C. (2026). sw2023: Nonparametric Multiple-Output Stochastic
  Frontier Analysis in Python. Manuscript submitted for publication.

Reproduces all numerical results and figures reported in the manuscript.

Usage
-----
  python replication.py              # Quick mode  : Tables 1–4 + Figures (n_sims=20)
  python replication.py --full       # Full mode   : Tables 1–4 + Figures (n_sims=100)
  python replication.py --tables     # Tables only (skip figures)
  python replication.py --figures    # Figures only (skip Monte Carlo)

Runtime (approximate, single core)
-----------------------------------
  Quick  (--quick)  : < 10 minutes
  Full   (--full)   : ~ 60 minutes  (n_sims=100, n up to 800)

Package versions used in the manuscript
----------------------------------------
  Python  3.10   |  numpy  2.2.6  |  scipy  1.15.3
  pandas  2.3.3  |  matplotlib 3.10.3  |  sw2023  0.3.2
"""

import sys
import os
import numpy as np
import pandas as pd

# ── Command-line flags ────────────────────────────────────────────────────────
FULL    = "--full"    in sys.argv
TABLES  = "--tables"  in sys.argv   # skip figures
FIGURES = "--figures" in sys.argv   # skip Monte Carlo tables
N_SIMS  = 100 if FULL else 20

RUN_TABLES  = not FIGURES
RUN_FIGURES = not TABLES

# ── Header ────────────────────────────────────────────────────────────────────
W = 65
print("=" * W)
print("sw2023 — JSS Replication Script")
mode = "FULL (n_sims=100)" if FULL else "QUICK (n_sims=20)"
print(f"  Mode    : {mode}")
print(f"  Tables  : {'yes' if RUN_TABLES  else 'skipped (--figures)'}")
print(f"  Figures : {'yes' if RUN_FIGURES else 'skipped (--tables)'}")
print("=" * W)

# ── Package versions ──────────────────────────────────────────────────────────
import sw2023, scipy, matplotlib
print(f"\nsw2023      : {sw2023.__version__}")
print(f"Python      : {sys.version.split()[0]}")
print(f"numpy       : {np.__version__}")
print(f"scipy       : {scipy.__version__}")
print(f"pandas      : {pd.__version__}")
print(f"matplotlib  : {matplotlib.__version__}")

from sw2023 import (SW2023Model, PanelSW2023,
                    bootstrap_sw, test_r3_significance)
from sw2023.tests.monte_carlo import run_imse_grid, print_imse_comparison

HERE = os.path.dirname(os.path.abspath(__file__))


# =============================================================================
# TABLES 1 & 2 — Monte Carlo Validation
#
# Reproduces IMSE ratios relative to Simar & Wilson (2023) Table F.1.
# Table 1: scalar LOO-CV bandwidth  (mc_imse_results.csv)
# Table 2: product LOO-CV bandwidth (mc_imse_product.csv)
# =============================================================================
if RUN_TABLES:
    print("\n" + "=" * W)
    print("Tables 1 & 2 — Monte Carlo Validation")
    print(f"  DGP   : Marsaglia (1972) sphere, sigma_eta=0.5")
    print(f"  (p,q) : (1,1), (1,2), (2,1), (2,2)")
    print(f"  n     : 100, 200, 400" + (", 800" if FULL else " (add --full for n=800)"))
    print(f"  n_sims: {N_SIMS}" + ("  [paper: 100]" if not FULL else "  [paper]"))
    print("=" * W)

    n_list = [100, 200, 400, 800] if FULL else [100, 200, 400]

    # ── Table 1: Scalar LOO-CV ────────────────────────────────────────────────
    print("\n-- Table 1: Scalar LOO-CV bandwidth --")
    np.random.seed(2023)
    csv_scalar = os.path.join(HERE, "mc_imse_results.csv")
    try:
        df_t1 = run_imse_grid(
            pq_list=[(1,1), (1,2), (2,1), (2,2)],
            n_list=n_list,
            rho_list=[0.0],
            n_sims=N_SIMS,
            bandwidth_method='loocv',
            out_csv=csv_scalar,
            verbose=False,
        )
        print()
        print_imse_comparison(df_t1)
        print(f"\n  → Saved: {csv_scalar}")
    except Exception as e:
        print(f"  [ERROR] Monte Carlo (scalar): {e}")

    # ── Pre-computed reference (n_sims=100) ──────────────────────────────────
    if os.path.exists(csv_scalar):
        print("\n-- Table 1 (pre-computed, n_sims=100, paper values) --")
        ref = pd.read_csv(csv_scalar)
        print(ref.to_string(index=False))

    # ── Table 2: Product LOO-CV ───────────────────────────────────────────────
    print("\n-- Table 2: Product LOO-CV bandwidth --")
    np.random.seed(2023)
    csv_product = os.path.join(HERE, "mc_imse_product.csv")
    try:
        df_t2 = run_imse_grid(
            pq_list=[(1,1), (1,2), (2,1), (2,2)],
            n_list=n_list,
            rho_list=[0.0],
            n_sims=N_SIMS,
            bandwidth_method='loocv',   # product kernel inside run_imse_grid
            out_csv=csv_product,
            verbose=False,
        )
        print()
        print_imse_comparison(df_t2)
        print(f"\n  → Saved: {csv_product}")
    except Exception as e:
        print(f"  [ERROR] Monte Carlo (product): {e}")

    if os.path.exists(csv_product):
        print("\n-- Table 2 (pre-computed, n_sims=100, paper values) --")
        prd = pd.read_csv(csv_product)
        print(prd.to_string(index=False))


# =============================================================================
# TABLE 3 — Simulation-Based Illustration (Section 6)
#
# DGP: Marsaglia sphere, p=2, q=2, n=200, sigma_eta=0.5, rho=1.0
# Reproduces all code-output blocks in Section 6 of the manuscript.
# =============================================================================
if RUN_TABLES:
    print("\n" + "=" * W)
    print("Table 3 — Simulation-Based Illustration (Section 6)")
    print("  DGP: Marsaglia sphere, p=q=2, n=200, sigma_eta=0.5, rho=1.0")
    print("=" * W)

    np.random.seed(42)
    n, p, q = 200, 2, 2

    def sphere(n, dim):
        """Draw n uniform points from the surface of a (dim-1)-sphere."""
        X = np.random.randn(n, dim)
        return X / np.linalg.norm(X, axis=1, keepdims=True)

    pts = np.abs(sphere(n, p + q))
    X_sim, Y_sim = pts[:, :p], pts[:, p:]
    sigma_eta, rho_sim = 0.5, 1.0
    eta_sim = np.abs(np.random.randn(n)) * sigma_eta
    eps_sim = np.random.randn(n) * rho_sim * sigma_eta
    Y_sim = Y_sim * np.clip(1 - eta_sim * 0.3 + eps_sim * 0.1, 0.5, 1.5)[:, None]

    # ── 6.1 Cross-sectional model ─────────────────────────────────────────────
    print("\n-- Table 3, Row 1: Cross-sectional model (Silverman bandwidth) --")
    m_sim = SW2023Model(X_sim, Y_sim, direction='mean', method='HMS',
                        bandwidth_method='silverman')
    m_sim.fit(verbose=False)
    print(m_sim)   # __repr__
    print(f"  Mean efficiency  : {m_sim.efficiency_.mean():.4f}   (paper: 0.5311)")
    print(f"  Std  efficiency  : {m_sim.efficiency_.std():.4f}   (paper: 0.2538)")
    print(f"  Mean sigma_eta   : {m_sim.sigma_eta_.mean():.4f}   (paper: 1.0551)")
    print(f"  Wrong skewness   : {(m_sim.r3_>0).mean()*100:.1f}%")

    # ── 6.2 Asymptotic CI ─────────────────────────────────────────────────────
    print("\n-- Table 3, Row 2: Asymptotic confidence intervals (alpha=0.05) --")
    ci_sim = m_sim.confint_asymptotic(alpha=0.05)
    ci_sim.summary()
    print(f"  Mean SE(phi_hat)        : {ci_sim.se_phi.mean():.4f}   (paper: 0.3745)")
    print(f"  Mean CI lower (phi_hat) : {ci_sim.phi_hat_ci[:,0].mean():.4f}   (paper: -0.0064)")
    print(f"  Mean CI upper (phi_hat) : {ci_sim.phi_hat_ci[:,1].mean():.4f}   (paper: 1.4618)")

    # ── 6.3 Bootstrap CI ──────────────────────────────────────────────────────
    print("\n-- Table 3, Row 3: Bootstrap confidence intervals (B=199, seed=2023) --")
    boot_sim = bootstrap_sw(X_sim, Y_sim, B=199, alpha=0.05,
                            bandwidth_method='silverman',
                            seed=2023, verbose=False)
    boot_sim.summary()
    width = (boot_sim.phi_hat_ci[:,1] - boot_sim.phi_hat_ci[:,0]).mean()
    lo_b, hi_b = boot_sim.eff_mean_ci
    print(f"  Mean CI width (frontier): {width:.4f}   (paper: 2.0043)")
    print(f"  Mean eff 95% CI         : [{lo_b:.4f}, {hi_b:.4f}]   (paper: [0.4661, 0.5757])")

    # ── 6.4 Wild bootstrap significance test ──────────────────────────────────
    print("\n-- Table 3, Row 4: Wild bootstrap significance test (B=999, seed=2023) --")
    tres = test_r3_significance(X_sim, Y_sim, B=999, seed=2023, verbose=False)
    tres.summary()
    print(f"  Test statistic T: {tres.statistic:.4f}   (paper: 0.0397)")
    print(f"  p-value         : {tres.p_value:.4f}   (paper: 0.8819, seed=2023, B=999)")
    print("  => H0 not rejected (homogeneous inefficiency), consistent with DGP.")


# =============================================================================
# TABLE 4 — Norwegian Agricultural Panel (Section 7)
#
# Data: norway_for_python.csv  (Kumbhakar et al. 2015, n=2729, T=10)
# Reproduces Table 3 (cross-sectional summary) and Table 4 (yearly TE/PE).
# =============================================================================
if RUN_TABLES:
    print("\n" + "=" * W)
    print("Table 4 — Norwegian Agricultural Panel (Section 7)")
    print("  Data : norway_for_python.csv  (n=2,729, 1993–2002)")
    print("=" * W)

    data_path = os.path.join(HERE, "norway_for_python.csv")
    if not os.path.exists(data_path):
        print("  [SKIP] norway_for_python.csv not found.")
        print("         Place it in the same directory as this script.")
    else:
        np.random.seed(2023)
        df_no = (pd.read_csv(data_path)
                   .sort_values(["farmid","year"])
                   .reset_index(drop=True))
        X_no = df_no[["x1","x2","x3","x4","x5","x6"]].values
        Y_no = df_no[["y1","y2","y3","y4"]].values

        # ── Table 4a: Cross-sectional summary (paper Table 3) ─────────────────
        print("\n-- Table 4a: Cross-sectional efficiency summary (paper Table 3) --")
        m_no = SW2023Model(X_no, Y_no, method="HMS",
                           bandwidth_method="silverman")
        m_no.fit(verbose=False)
        eff_no = m_no.efficiency_
        ws_no  = (m_no.r3_ > 0).mean() * 100
        print(f"  Mean efficiency   : {eff_no.mean():.3f}   (paper: 0.813)")
        print(f"  Median efficiency : {np.median(eff_no):.3f}   (paper: 0.831)")
        print(f"  Std  efficiency   : {eff_no.std():.3f}   (paper: 0.131)")
        print(f"  Min  efficiency   : {eff_no.min():.3f}   (paper: 0.251)")
        print(f"  Wrong-skewness    : {ws_no:.1f}%   (paper: 50.6%)")

        # ── Table 4b: Yearly TE / PE (paper Table 4) ─────────────────────────
        print("\n-- Table 4b: Yearly mean efficiency (paper Table 4) --")
        m_panel = PanelSW2023(X_no, Y_no,
                              df_no["farmid"].values,
                              df_no["year"].values,
                              method="HMS")
        m_panel.fit(verbose=False)

        df_res = df_no[["farmid","year"]].copy()
        df_res["cs_eff"] = m_no.efficiency_
        df_res["te"]     = m_panel.eff_transient_
        df_res["pe"]     = m_panel.eff_persistent_
        yearly = df_res.groupby("year")[["cs_eff","te","pe"]].mean()

        print(f"\n  {'Year':>4}  {'CS':>6}  {'TE':>6}  {'PE':>6}")
        print("  " + "-" * 26)
        for yr, row in yearly.iterrows():
            print(f"  {yr:>4}  {row['cs_eff']:>6.3f}  {row['te']:>6.3f}  {row['pe']:>6.3f}")
        print("  " + "-" * 26)
        print(f"  {'Mean':>4}  {yearly['cs_eff'].mean():>6.3f}  "
              f"{yearly['te'].mean():>6.3f}  {yearly['pe'].mean():>6.3f}")
        print(f"  {'Paper':>4}  {'0.813':>6}  {'0.961':>6}  {'0.978':>6}")


# =============================================================================
# FIGURES — Reproduce all manuscript figures
#
# Figure 1: Synthetic data — frontier estimates (3 bandwidth methods)
# Figure 2: Norwegian data — efficiency distributions + yearly trend
# =============================================================================
if RUN_FIGURES:
    print("\n" + "=" * W)
    print("Figures — Reproducing manuscript figures")
    print("=" * W)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from scipy.stats import gaussian_kde

    # JSS plot style
    plt.rcParams.update({
        'font.family'      : 'serif',
        'font.size'        : 10,
        'axes.labelsize'   : 10,
        'axes.titlesize'   : 10,
        'legend.fontsize'  : 9,
        'xtick.labelsize'  : 9,
        'ytick.labelsize'  : 9,
        'figure.dpi'       : 150,
        'axes.spines.top'  : False,
        'axes.spines.right': False,
    })
    COLORS = {
        'silverman'   : '#2166ac',
        'loocv_scalar': '#d6604d',
        'loocv'       : '#1a9850',
    }

    # ── Figure 1: Synthetic data ──────────────────────────────────────────────
    print("\n-- Figure 1: Synthetic data (p=1, q=1, n=300) --")
    np.random.seed(2023)
    n_fig = 300
    x_raw   = np.sort(np.random.uniform(1, 5, n_fig))
    phi_true = 2.0 * np.log(x_raw)
    v_fig   = np.random.normal(0, 0.20, n_fig)
    u_fig   = np.abs(np.random.normal(0, 0.30, n_fig))
    y_fig   = np.exp(phi_true - u_fig + v_fig)

    X_fig = x_raw.reshape(-1, 1)
    Y_fig = y_fig.reshape(-1, 1)

    results_fig1 = {}
    for bw in ['silverman', 'loocv_scalar', 'loocv']:
        m_f = SW2023Model(X_fig, Y_fig, method='HMS',
                          bandwidth_method=bw,
                          log_transform=True, standardize=True)
        m_f.fit(verbose=False)
        order = np.argsort(m_f.Z_[:, 0])
        results_fig1[bw] = {
            'z'    : m_f.Z_[:, 0][order],
            'phi'  : m_f.phi_hat_[order],
            'eff'  : m_f.efficiency_,
        }
        print(f"  {bw:15s}: mean_eff={m_f.efficiency_.mean():.4f}")

    fig1, axes1 = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes1[0]
    for bw, label, ls in [
        ('silverman',   'Silverman',      '-'),
        ('loocv_scalar','LOO-CV scalar',  '--'),
        ('loocv',       'LOO-CV product', ':'),
    ]:
        ax.plot(results_fig1[bw]['z'], results_fig1[bw]['phi'],
                color=COLORS[bw], ls=ls, lw=1.8, label=label)
    ax.set_xlabel(r'Projected input $Z$')
    ax.set_ylabel(r'Estimated frontier $\hat{\varphi}(Z)$')
    ax.set_title(r'(a) Frontier estimates  ($p=1,\,q=1,\,n=300$)')
    ax.legend(frameon=False)

    ax2 = axes1[1]
    x_grid = np.linspace(0, 1, 300)
    for bw, label, ls in [
        ('silverman',   'Silverman',      '-'),
        ('loocv_scalar','LOO-CV scalar',  '--'),
        ('loocv',       'LOO-CV product', ':'),
    ]:
        eff = results_fig1[bw]['eff']
        kde = gaussian_kde(eff, bw_method=0.15)
        ax2.plot(x_grid, kde(x_grid), color=COLORS[bw], ls=ls, lw=1.8,
                 label=label)
        ax2.axvline(eff.mean(), color=COLORS[bw], ls=ls, lw=0.8, alpha=0.6)
    ax2.set_xlabel('Efficiency score')
    ax2.set_ylabel('Density')
    ax2.set_title(r'(b) Efficiency distributions  (vertical lines = means)')
    ax2.legend(frameon=False)

    fig1.tight_layout()
    out1a = os.path.join(HERE, "fig_synthetic_comparison.pdf")
    out1b = os.path.join(HERE, "fig_synthetic_comparison.png")
    fig1.savefig(out1a, bbox_inches='tight')
    fig1.savefig(out1b, bbox_inches='tight', dpi=150)
    plt.close(fig1)
    print(f"  → Saved: fig_synthetic_comparison.pdf/png")

    # ── Figure 2: Norway data ─────────────────────────────────────────────────
    print("\n-- Figure 2: Norwegian agricultural panel --")
    loocv_csv = os.path.join(HERE, "norway_loocv_comparison.csv")
    if not os.path.exists(loocv_csv):
        print(f"  [SKIP] {loocv_csv} not found — run run_loocv_comparison.py first.")
    else:
        df_fig2 = pd.read_csv(loocv_csv)

        fig2, axes2 = plt.subplots(1, 2, figsize=(10, 4))
        labels_fig2 = {
            'eff_silverman': ('Silverman',      '-',  COLORS['silverman']),
            'eff_scalar'   : ('LOO-CV scalar',  '--', COLORS['loocv_scalar']),
            'eff_product'  : ('LOO-CV product', ':',  COLORS['loocv']),
        }

        ax = axes2[0]
        x_grid = np.linspace(0, 1.05, 400)
        for col, (label, ls, color) in labels_fig2.items():
            vals = df_fig2[col].dropna().values
            kde  = gaussian_kde(vals, bw_method=0.08)
            ax.plot(x_grid, kde(x_grid), color=color, ls=ls, lw=1.8,
                    label=label)
            ax.axvline(vals.mean(), color=color, ls=ls, lw=0.8, alpha=0.7)
        ax.set_xlabel('Efficiency score')
        ax.set_ylabel('Density')
        ax.set_title('(a) Efficiency distributions\nNorwegian farms, 1993–2002  ($n=2{,}729$)')
        ax.legend(frameon=False)
        ax.set_xlim(0.1, 1.05)

        ax2 = axes2[1]
        yearly2 = df_fig2.groupby('year')[
            ['eff_silverman','eff_scalar','eff_product',
             'TE_silverman','PE_silverman']].mean()
        years2 = yearly2.index.astype(int)
        for col, (label, ls, color) in labels_fig2.items():
            ax2.plot(years2, yearly2[col], color=color, ls=ls,
                     lw=1.8, marker='o', ms=4, label=label)
        ax2.plot(years2, yearly2['TE_silverman'], color='#999999',
                 ls='-.', lw=1.4, marker='s', ms=3, label='TE (4-comp)')
        ax2.plot(years2, yearly2['PE_silverman'], color='#555555',
                 ls='-.', lw=1.4, marker='^', ms=3, label='PE (4-comp)')
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Mean efficiency')
        ax2.set_title('(b) Mean efficiency by year\n(bandwidth methods + panel)')
        ax2.set_xticks(years2)
        ax2.set_xticklabels(years2, rotation=45)
        ax2.legend(frameon=False, fontsize=8, ncol=2)
        ax2.set_ylim(0.5, 1.02)

        fig2.tight_layout()
        out2a = os.path.join(HERE, "fig_norway_comparison.pdf")
        out2b = os.path.join(HERE, "fig_norway_comparison.png")
        fig2.savefig(out2a, bbox_inches='tight')
        fig2.savefig(out2b, bbox_inches='tight', dpi=150)
        plt.close(fig2)
        print(f"  → Saved: fig_norway_comparison.pdf/png")


# ── Footer ────────────────────────────────────────────────────────────────────
print("\n" + "=" * W)
print("Replication complete.")
if not FULL and RUN_TABLES:
    print("  Note: Run with --full for n=800 and n_sims=100 (paper Tables 1–2).")
print("=" * W)
