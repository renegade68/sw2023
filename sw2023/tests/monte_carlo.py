"""
Monte Carlo Validation

Purpose: Generate data from a known DGP (Data Generating Process) and
         verify how well the estimators recover the true values.

DGP design:
  - Production technology: Cobb-Douglas (multiple inputs, multiple outputs)
  - 2-component: y = f(x) + v - u
  - 4-component: y = f(x) + α_i + v_it - u_it - μ_i

Validation criteria:
  1. Mean efficiency bias (Bias)
  2. Efficiency rank correlation (Spearman rank correlation)
  3. RMSE (root mean squared error)
  4. Residual skewness direction (fraction with r3 < 0)
  5. σ_η estimation accuracy

References:
  SW(2023) Appendix F: Monte Carlo experiment design
  Parmeter & Kumbhakar(2025): 4-component validation
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, halfnorm
from scipy.linalg import null_space


# ─────────────────────────────────────────────────────────────
# DGP generation functions
# ─────────────────────────────────────────────────────────────

def dgp_2component(n=500, p=2, q=2, seed=42,
                   sigma_v=0.2, sigma_u=0.3,
                   alpha_inputs=None, alpha_outputs=None):
    """
    2-component DGP: multiple inputs + multiple outputs Cobb-Douglas.

    Production technology:
      For each output m: ln(y_m) = Σ_j α_mj × ln(x_j) + v - u_m

    Uses output distance function:
      ln D_O(x,y) = -ln(y1) - α × Σ_m ln(y_m/y1) - β × Σ_j ln(x_j)
    where normalization: IDF formula applied with y1 as reference

    Simple DGP (SW 2023 Appendix F approach):
      Frontier: (x-1)'(x-1) + y'y = 1 (sphere surface)
      + noise ε ~ N(0, σ_v²)
      - ineff η ~ N⁺(0, σ_u²)

    Parameters
    ----------
    n            : sample size
    p            : number of inputs
    q            : number of outputs
    sigma_v      : noise standard deviation
    sigma_u      : inefficiency standard deviation
    alpha_inputs : (p,) input elasticities (None=equal)
    alpha_outputs: (q,) output elasticities (None=equal)

    Returns
    -------
    dict:
        X, Y         : observed data
        X_true, Y_true: frontier data
        u_true       : true inefficiency
        eff_true     : true efficiency exp(-u)
        sigma_u, sigma_v : parameters used
    """
    rng = np.random.default_rng(seed)

    # Input elasticities
    if alpha_inputs is None:
        alpha_inputs = np.ones(p) / p       # equal
    if alpha_outputs is None:
        alpha_outputs = np.ones(q) / q      # equal

    # Generate inputs: log-normal (mean=1 in log space → non-zero direction vector)
    ln_X = rng.normal(1.0, 0.5, size=(n, p))
    X    = np.exp(ln_X)

    # Output frontier: Cobb-Douglas
    # ln(y_m*) = α_m × Σ_j β_j × ln(x_j)  (equal elasticities)
    ln_X_weighted = ln_X @ alpha_inputs.reshape(-1, 1)  # (n, 1)
    ln_Y_true = ln_X_weighted * alpha_outputs.reshape(1, -1)  # (n, q)
    Y_true = np.exp(ln_Y_true)

    # Inefficiency: η ~ N⁺(0, σ_u²)
    u_true = halfnorm.rvs(scale=sigma_u, size=n, random_state=rng)

    # Noise: ε ~ N(0, σ_v²)
    v = rng.normal(0, sigma_v, size=(n, q))

    # Observed output: inefficiency reduces output
    ln_Y_obs = ln_Y_true - u_true.reshape(-1, 1) + v
    Y_obs    = np.exp(ln_Y_obs)

    eff_true = np.exp(-u_true)

    return {
        'X'        : X,
        'Y'        : Y_obs,
        'X_true'   : X,
        'Y_true'   : Y_true,
        'u_true'   : u_true,
        'eff_true' : eff_true,
        'sigma_u'  : sigma_u,
        'sigma_v'  : sigma_v,
        'p': p, 'q': q, 'n': n,
    }


def dgp_4component(N=100, T=5, p=2, q=2, seed=42,
                   sigma_v=0.15, sigma_u=0.20,
                   sigma_alpha=0.10, sigma_mu=0.20,
                   alpha_inputs=None, alpha_outputs=None):
    """
    4-component panel DGP.

    U_it = φ(Z_it) + v_it - u_it + α_i - μ_i

    v_it   ~ N(0, σ_v²)     : transient noise
    u_it   ~ N⁺(0, σ_u²)    : transient inefficiency
    α_i    ~ N(0, σ_α²)     : individual heterogeneity
    μ_i    ~ N⁺(0, σ_μ²)    : persistent inefficiency

    Parameters
    ----------
    N, T : number of individuals, number of periods
    p, q : number of inputs, number of outputs
    sigma_v, sigma_u, sigma_alpha, sigma_mu : standard deviations

    Returns
    -------
    dict: X, Y, firm_id, time_id, true efficiency components
    """
    rng = np.random.default_rng(seed)
    n   = N * T

    if alpha_inputs is None:
        alpha_inputs = np.ones(p) / p
    if alpha_outputs is None:
        alpha_outputs = np.ones(q) / q

    # Individual fixed components
    alpha_i = rng.normal(0, sigma_alpha, size=N)
    mu_i    = halfnorm.rvs(scale=sigma_mu, size=N, random_state=rng)

    # Generate observations
    X_list, Y_list = [], []
    firm_ids, time_ids = [], []
    u_list, v_list = [], []
    eff_t_list, eff_p_list = [], []

    for i in range(N):
        for t in range(T):
            # Inputs (mean=1 in log space → non-zero direction vector)
            ln_x = rng.normal(1.0, 0.5, size=p)
            x    = np.exp(ln_x)

            # Frontier output
            ln_y_star = (ln_x @ alpha_inputs) * alpha_outputs

            # Transient components
            u_it = halfnorm.rvs(scale=sigma_u, random_state=rng)
            v_it = rng.normal(0, sigma_v, size=q)

            # Observed output
            ln_y = ln_y_star + alpha_i[i] - u_it - mu_i[i] + v_it
            y    = np.exp(ln_y)

            X_list.append(x)
            Y_list.append(y)
            firm_ids.append(i)
            time_ids.append(t)
            u_list.append(u_it)
            eff_t_list.append(np.exp(-u_it))
            eff_p_list.append(np.exp(-mu_i[i]))

    X       = np.array(X_list)
    Y       = np.array(Y_list)
    firm_id = np.array(firm_ids)
    time_id = np.array(time_ids)
    u_arr   = np.array(u_list)
    eff_t   = np.array(eff_t_list)
    eff_p   = np.array([np.exp(-mu_i[f]) for f in firm_id])
    eff_all = eff_t * eff_p

    return {
        'X': X, 'Y': Y,
        'firm_id': firm_id, 'time_id': time_id,
        'u_true'         : u_arr,
        'mu_true'        : np.array([mu_i[f] for f in firm_id]),
        'alpha_true'     : np.array([alpha_i[f] for f in firm_id]),
        'eff_true'       : eff_all,
        'eff_transient'  : eff_t,
        'eff_persistent' : eff_p,
        'sigma_u': sigma_u, 'sigma_v': sigma_v,
        'sigma_mu': sigma_mu, 'sigma_alpha': sigma_alpha,
        'N': N, 'T': T, 'p': p, 'q': q, 'n': n,
    }


# ─────────────────────────────────────────────────────────────
# Validation metric computation
# ─────────────────────────────────────────────────────────────

def validation_metrics(eff_true, eff_est, label=''):
    """
    Compare estimated efficiency with true efficiency.

    Metrics:
      - Bias: E[ê - e]
      - RMSE: √E[(ê - e)²]
      - Spearman rank correlation
      - Mean estimated value vs. true mean
    """
    mask = ~(np.isnan(eff_true) | np.isnan(eff_est))
    et = eff_true[mask]
    ee = eff_est[mask]

    bias   = np.mean(ee - et)
    rmse   = np.sqrt(np.mean((ee - et) ** 2))
    rho, p = spearmanr(et, ee)

    metrics = {
        'label'       : label,
        'n_valid'     : mask.sum(),
        'true_mean'   : et.mean(),
        'est_mean'    : ee.mean(),
        'bias'        : bias,
        'abs_bias'    : abs(bias),
        'rmse'        : rmse,
        'spearman_rho': rho,
        'spearman_p'  : p,
    }
    return metrics


# ─────────────────────────────────────────────────────────────
# SW(2023)-native DGP
# ─────────────────────────────────────────────────────────────

def dgp_sw_native(n=500, p=2, q=2, seed=42,
                  sigma_v=0.2, sigma_u=0.3,
                  intercept=2.0):
    """
    DGP fully consistent with the SW(2023) model.

    Data generated directly in the rotated coordinate system (Z, U):
      U_i = φ(Z_i) + v_i - η_i     ← same as model structure

    Generation procedure:
      1. Direction vector d = (0,...,0, 1/√q,...,1/√q)  (output direction)
      2. Rotation basis V = null_space(d')  : (r, r-1)
      3. Z ~ N(0,1)  (r-1 dimensional latent covariate)
      4. φ(Z) = intercept + Z @ β  (linear frontier)
      5. v ~ N(0,σ_v²), η ~ half-normal(σ_u)
      6. U = φ(Z) + v - η
      7. W = Z @ V' + U[:,None] @ d[None,:]
      8. X = exp(W[:,:p]), Y = exp(W[:,p:])

    Role of intercept:
      mean(W[:,p:]) ≈ intercept × (1/√q) > 0  →  ensures d_Y > 0 in direction vector
    """
    rng = np.random.default_rng(seed)
    r   = p + q

    # True direction vector
    d_true = np.zeros(r)
    d_true[p:] = 1.0 / np.sqrt(q)

    # Rotation basis (orthogonal to d)
    V = null_space(d_true.reshape(1, -1))     # (r, r-1)

    # Latent covariates (standard normal)
    Z    = rng.normal(0.0, 1.0, size=(n, r-1))
    beta = np.ones(r-1) / (r-1)
    phi  = intercept + Z @ beta               # (n,)

    # Noise + inefficiency
    v   = rng.normal(0, sigma_v, size=n)
    eta = halfnorm.rvs(scale=sigma_u, size=n, random_state=rng)
    U   = phi + v - eta
    eff_true = np.exp(-eta)

    # Recover original space
    W = Z @ V.T + U[:, None] @ d_true[None, :]
    X = np.exp(W[:, :p])
    Y = np.exp(W[:, p:])

    return {
        'X'       : X,
        'Y'       : Y,
        'eff_true': eff_true,
        'eta_true': eta,
        'sigma_u' : sigma_u,
        'sigma_v' : sigma_v,
        'd_true'  : d_true,
        'n': n, 'p': p, 'q': q,
    }


# ─────────────────────────────────────────────────────────────
# 2-component Monte Carlo
# ─────────────────────────────────────────────────────────────

def run_mc_2component(n_sims=50, n=500, p=2, q=2,
                      sigma_v=0.2, sigma_u=0.3,
                      verbose=True):
    """
    2-component Monte Carlo experiment.

    Each iteration:
      1. Generate DGP
      2. Estimate SW2023Model
      3. Compute efficiency validation metrics
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.model import SW2023Model

    results = []

    for sim in range(n_sims):
        if verbose and sim % 10 == 0:
            print(f"  Simulation {sim+1}/{n_sims}...")

        data = dgp_2component(n=n, p=p, q=q, seed=sim,
                               sigma_v=sigma_v, sigma_u=sigma_u)
        try:
            m = SW2023Model(data['X'], data['Y'],
                             direction='mean', method='HMS',
                             log_transform=True, standardize=True)
            m.fit(verbose=False)

            m_ = validation_metrics(data['eff_true'], m.efficiency_,
                                     label=f'sim_{sim}')
            m_['wrong_skew_pct'] = (m.r3_ > 0).mean() * 100
            m_['sigma_u_est']    = np.nanmean(m.sigma_eta_)
            m_['sigma_u_true']   = sigma_u
            results.append(m_)

        except Exception as e:
            if verbose:
                print(f"    Error (sim={sim}): {e}")

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n=== 2-Component MC Results Summary (n_sims={len(df)}) ===")
        print(f"  Mean bias      : {df['bias'].mean():.4f} ± {df['bias'].std():.4f}")
        print(f"  Mean RMSE      : {df['rmse'].mean():.4f} ± {df['rmse'].std():.4f}")
        print(f"  Mean Spearman ρ: {df['spearman_rho'].mean():.4f}")
        print(f"  True mean eff  : {df['true_mean'].mean():.4f}")
        print(f"  Est mean eff   : {df['est_mean'].mean():.4f}")
        print(f"  Wrong skew (%) : {df['wrong_skew_pct'].mean():.1f}")
        print(f"  σ_u true       : {sigma_u:.4f}")
        print(f"  σ_u estimated  : {df['sigma_u_est'].mean():.4f}")

    return df


def run_mc_native(n_sims=50, n=500, p=2, q=2,
                  sigma_v=0.2, sigma_u=0.3,
                  verbose=True):
    """
    2-component MC using SW(2023)-native DGP.

    Provides cleaner validation results since there is no DGP-model mismatch.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.model import SW2023Model

    results = []

    for sim in range(n_sims):
        if verbose and sim % 10 == 0:
            print(f"  Simulation {sim+1}/{n_sims}...")

        data = dgp_sw_native(n=n, p=p, q=q, seed=sim,
                              sigma_v=sigma_v, sigma_u=sigma_u)
        try:
            m = SW2023Model(data['X'], data['Y'],
                             direction='mean', method='HMS',
                             log_transform=True, standardize=True)
            m.fit(verbose=False)

            m_ = validation_metrics(data['eff_true'], m.efficiency_,
                                     label=f'sim_{sim}')
            m_['wrong_skew_pct'] = (m.r3_ > 0).mean() * 100
            m_['sigma_u_est']    = np.nanmean(m.sigma_eta_)
            m_['sigma_u_true']   = sigma_u
            results.append(m_)
        except Exception as e:
            if verbose:
                print(f"    Error (sim={sim}): {e}")

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n=== Native DGP MC Results (n_sims={len(df)}, n={n}, p={p}, q={q}) ===")
        print(f"  Mean bias      : {df['bias'].mean():.4f} ± {df['bias'].std():.4f}")
        print(f"  Mean RMSE      : {df['rmse'].mean():.4f} ± {df['rmse'].std():.4f}")
        print(f"  Mean Spearman ρ: {df['spearman_rho'].mean():.4f}")
        print(f"  True mean eff  : {df['true_mean'].mean():.4f}")
        print(f"  Est mean eff   : {df['est_mean'].mean():.4f}")
        print(f"  Wrong skew (%) : {df['wrong_skew_pct'].mean():.1f}")
        print(f"  σ_u true       : {sigma_u:.4f}")
        print(f"  σ_u estimated  : {df['sigma_u_est'].mean():.4f}")

    return df


# ─────────────────────────────────────────────────────────────
# 4-component Monte Carlo
# ─────────────────────────────────────────────────────────────

def run_mc_4component(n_sims=50, N=100, T=5, p=2, q=2,
                      sigma_v=0.15, sigma_u=0.20,
                      sigma_alpha=0.10, sigma_mu=0.20,
                      verbose=True):
    """
    4-component panel Monte Carlo experiment.
    """
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from panel.four_component import PanelSW2023

    results = []

    for sim in range(n_sims):
        if verbose and sim % 10 == 0:
            print(f"  Simulation {sim+1}/{n_sims}...")

        data = dgp_4component(N=N, T=T, p=p, q=q, seed=sim,
                               sigma_v=sigma_v, sigma_u=sigma_u,
                               sigma_alpha=sigma_alpha, sigma_mu=sigma_mu)
        try:
            m = PanelSW2023(data['X'], data['Y'],
                             data['firm_id'], data['time_id'],
                             direction='mean', method='HMS',
                             log_transform=True, standardize=True)
            m.fit(verbose=False)

            # Overall efficiency
            m_all = validation_metrics(data['eff_true'], m.efficiency_,
                                        label='overall')
            # Transient efficiency
            m_te  = validation_metrics(data['eff_transient'], m.eff_transient_,
                                        label='transient')
            # Persistent efficiency
            m_pe  = validation_metrics(data['eff_persistent'], m.eff_persistent_,
                                        label='persistent')

            results.append({
                'sim'             : sim,
                'bias_all'        : m_all['bias'],
                'rmse_all'        : m_all['rmse'],
                'rho_all'         : m_all['spearman_rho'],
                'bias_te'         : m_te['bias'],
                'rmse_te'         : m_te['rmse'],
                'rho_te'          : m_te['spearman_rho'],
                'bias_pe'         : m_pe['bias'],
                'rmse_pe'         : m_pe['rmse'],
                'rho_pe'          : m_pe['spearman_rho'],
                'est_mean_all'    : np.nanmean(m.efficiency_),
                'true_mean_all'   : data['eff_true'].mean(),
            })
        except Exception as e:
            if verbose:
                print(f"    Error (sim={sim}): {e}")

    df = pd.DataFrame(results)

    if verbose:
        print(f"\n=== 4-Component MC Results Summary (n_sims={len(df)}) ===")
        for comp, label in [('all','overall'), ('te','transient'), ('pe','persistent')]:
            print(f"\n  [{label} efficiency]")
            print(f"    Bias  : {df[f'bias_{comp}'].mean():.4f}")
            print(f"    RMSE  : {df[f'rmse_{comp}'].mean():.4f}")
            print(f"    ρ     : {df[f'rho_{comp}'].mean():.4f}")

    return df


# ─────────────────────────────────────────────────────────────
# SW(2023) paper reproduction DGP and IMSE computation
# ─────────────────────────────────────────────────────────────

def dgp_sw_sphere(n, p, q, sigma_eta=0.5, rho=1.0, seed=42):
    """
    Sphere-based DGP from SW(2023) Appendix F.

    Generates uniformly distributed points on the surface of a (p+q)-dimensional
    sphere using the Muller(1959)/Marsaglia(1972) method, and transforms them
    into input/output frontier points.

    Procedure:
      1. a_i ~ Uniform(S^{p+q-1})  (standard normal → normalize)
      2. X^∂_i = 1 - |a_{p,i}|,  Y^∂_i = |a_{q,i}|
      3. d = normalized([-1_p, 1_q])
      4. U^∂_i = d' W^∂_i  (U value on frontier)
      5. σ_ε = ρ × √((π-2)/π) × σ_η   (SW paper eq. F.1)
      6. ε_i ~ N(0,σ_ε²),  η_i ~ N⁺(0,σ_η²)
      7. U_i = U^∂_i + ε_i - η_i
      8. Z_i = Z^∂_i  (same)
      9. (X_i,Y_i) inverse transform

    Parameters
    ----------
    n          : sample size
    p, q       : number of inputs/outputs
    sigma_eta  : η standard deviation (paper fixed value 0.5)
    rho        : noise-to-signal ratio (σ_ε / √((π-2)/π) / σ_η)
    seed       : random seed

    Returns
    -------
    dict: X, Y (observed), X_star, Y_star (frontier), U_star (true U^∂),
          Z_star, d, sigma_eta, sigma_eps, rho
    """
    rng = np.random.default_rng(seed)
    r   = p + q

    # ── 1. Uniform distribution on sphere surface (Muller/Marsaglia) ──────────
    a_raw = rng.standard_normal(size=(n, r))
    a     = a_raw / np.linalg.norm(a_raw, axis=1, keepdims=True)

    # ── 2. Frontier points ────────────────────────────────────
    X_star = 1.0 - np.abs(a[:, :p])   # (n, p) ∈ [0, 1]
    Y_star = np.abs(a[:, p:])          # (n, q) ∈ [0, 1]
    W_star = np.hstack([X_star, Y_star])  # (n, r)

    # ── 3. Direction vector (same as SW paper) ────────────────
    d_raw = np.concatenate([-np.ones(p), np.ones(q)])
    d     = d_raw / np.linalg.norm(d_raw)     # (r,) normalized

    # ── 4. Rotation basis ──────────────────────────────────────
    V = null_space(d.reshape(1, -1))   # (r, r-1): basis orthogonal to d

    # ── 5. Frontier → (Z^∂, U^∂) transform ───────────────────
    U_star = W_star @ d         # (n,)
    Z_star = W_star @ V         # (n, r-1)

    # ── 6. Determine σ_ε (SW paper eq. F.1) ──────────────────
    sigma_eps = rho * np.sqrt((np.pi - 2) / np.pi) * sigma_eta

    # ── 7. Generate noise and inefficiency ────────────────────
    eps = rng.normal(0.0, sigma_eps, size=n) if sigma_eps > 0 \
          else np.zeros(n)
    eta = halfnorm.rvs(scale=sigma_eta, size=n, random_state=rng)

    # ── 8. Observed values ────────────────────────────────────
    xi    = eps - eta
    U_obs = U_star + xi
    Z_obs = Z_star.copy()

    # ── 9. (Z, U) → (X, Y) inverse transform ──────────────────
    W_obs = Z_obs @ V.T + U_obs[:, None] * d[None, :]
    X_obs = W_obs[:, :p]
    Y_obs = W_obs[:, p:]

    return {
        'X': X_obs, 'Y': Y_obs,
        'X_star': X_star, 'Y_star': Y_star,
        'U_star': U_star, 'Z_star': Z_star,
        'd': d, 'V': V,
        'eta': eta, 'eps': eps,
        'sigma_eta': sigma_eta, 'sigma_eps': sigma_eps, 'rho': rho,
        'n': n, 'p': p, 'q': q,
    }


def run_imse_grid(n_sims=100,
                  n_list=None, pq_list=None, rho_list=None,
                  sigma_eta=0.5,
                  method='HMS',
                  bandwidth_method='loocv',
                  out_csv='mc_imse_results.csv',
                  verbose=True):
    """
    IMSE grid experiment for reproducing SW(2023) Table F.1.

    IMSE_m = (1/n) × Σ_i (φ̂(Z_i) − U^∂_i)²   [per-observation normalization]

    Model is run with log_transform=False, standardize=False, and
    direction vector set identical to the DGP.

    Parameters
    ----------
    n_sims    : Monte Carlo replications per cell (paper: 1000, default: 100)
    n_list    : list of sample sizes
    pq_list   : list of (p,q) pairs
    rho_list  : list of noise-to-signal ratios
    sigma_eta : inefficiency standard deviation (paper fixed value 0.5)
    method    : 'HMS' | 'SVKZ'
    out_csv   : output CSV path
    verbose   : whether to print progress

    Returns
    -------
    pd.DataFrame : IMSE grid results + paper Table F.1 values (if available)
    """
    import gc, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.model import SW2023Model

    if n_list   is None: n_list   = [100, 200, 400, 800]
    if pq_list  is None: pq_list  = [(1,1), (2,2)]
    if rho_list is None: rho_list = [0.0, 0.5, 1.0, 2.0]

    # SW(2023) Table F.1 HMS reference values (partial)
    # key: (p,q,rho,n) → IMSE   (source: Simar & Wilson 2023 Supplementary, Table F.1)
    TABLE_F1_HMS = {
        # rho=0.0 (sigma_eps=0, pure half-normal)
        (1,1,0.0,100):0.0161,(1,1,0.0,200):0.0082,(1,1,0.0,400):0.0042,
        (1,1,0.0,800):0.0021,(1,1,0.0,1600):0.0010,(1,1,0.0,3200):0.0006,
        (1,2,0.0,100):0.0274,(1,2,0.0,200):0.0187,(1,2,0.0,400):0.0113,
        (1,2,0.0,800):0.0066,(1,2,0.0,1600):0.0036,(1,2,0.0,3200):0.0020,
        (2,2,0.0,100):0.0344,(2,2,0.0,200):0.0228,(2,2,0.0,400):0.0153,
        (2,2,0.0,800):0.0111,(2,2,0.0,1600):0.0077,(2,2,0.0,3200):0.0051,
        (2,3,0.0,100):0.0388,(2,3,0.0,200):0.0263,(2,3,0.0,400):0.0170,
        (2,3,0.0,800):0.0121,(2,3,0.0,1600):0.0092,(2,3,0.0,3200):0.0073,
        (3,3,0.0,100):0.0437,(3,3,0.0,200):0.0293,(3,3,0.0,400):0.0192,
        (3,3,0.0,800):0.0125,(3,3,0.0,1600):0.0094,(3,3,0.0,3200):0.0075,
        # rho=0.25
        (1,1,0.25,100):0.0174,(1,1,0.25,200):0.0090,
        # rho=0.5
        (1,1,0.5,100):0.0216,(1,1,0.5,200):0.0118,(1,1,0.5,400):0.0063,
        (2,2,0.5,100):0.0438,(2,2,0.5,200):0.0293,(2,2,0.5,400):0.0198,
        # rho=1.0
        (1,1,1.0,100):0.0571,(1,1,1.0,200):0.0303,(1,1,1.0,400):0.0166,
        (2,2,1.0,100):0.1057,(2,2,1.0,200):0.0686,(2,2,1.0,400):0.0408,
        # rho=2.0
        (1,1,2.0,100):0.4107,(1,1,2.0,200):0.2902,(1,1,2.0,400):0.1980,
        (2,2,2.0,100):0.5573,(2,2,2.0,200):0.4266,(2,2,2.0,400):0.2907,
    }

    # Resume from existing results
    done_keys = set()
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        for _, row in existing.iterrows():
            done_keys.add((int(row['n']), int(row['p']), int(row['q']),
                           float(row['rho'])))
        if verbose:
            print(f"[Resume] Existing {len(existing)} rows → skipping {len(done_keys)} cells")

    total = len(n_list) * len(pq_list) * len(rho_list)
    cell_idx = 0
    all_rows  = []

    for n in n_list:
        for (p, q) in pq_list:
            for rho in rho_list:
                cell_idx += 1
                key = (n, p, q, rho)
                if key in done_keys:
                    if verbose:
                        print(f"[{cell_idx}/{total}] Skipping: "
                              f"n={n}, ({p},{q}), ρ={rho}")
                    continue

                ref = TABLE_F1_HMS.get((p, q, rho, n), float('nan'))
                if verbose:
                    print(f"\n[{cell_idx}/{total}] "
                          f"n={n}, (p,q)=({p},{q}), ρ={rho}  "
                          f"[paper ref: {ref:.4f}]" if not np.isnan(ref)
                          else f"\n[{cell_idx}/{total}] "
                          f"n={n}, (p,q)=({p},{q}), ρ={rho}")

                imse_list = []
                n_fail = 0

                for sim in range(n_sims):
                    data = dgp_sw_sphere(n=n, p=p, q=q,
                                         sigma_eta=sigma_eta,
                                         rho=rho, seed=sim)
                    try:
                        m = SW2023Model(
                            data['X'], data['Y'],
                            direction=data['d'],
                            method=method,
                            log_transform=False,
                            standardize=False,
                            bandwidth_method=bandwidth_method,
                        )
                        m.fit(verbose=False)

                        # IMSE: (1/n) × Σ (φ̂_i − U^∂_i)²
                        imse = np.mean((m.phi_hat_ - data['U_star']) ** 2)
                        imse_list.append(imse)

                        del m, data
                    except Exception as e:
                        n_fail += 1
                        del data
                    finally:
                        gc.collect()

                if not imse_list:
                    if verbose:
                        print(f"  → All failed, skipping")
                    continue

                imse_arr = np.array(imse_list)
                summary = {
                    'n'          : n,
                    'p'          : p,
                    'q'          : q,
                    'rho'        : rho,
                    'method'     : method,
                    'n_sims'     : len(imse_list),
                    'n_fail'     : n_fail,
                    'imse_mean'  : imse_arr.mean(),
                    'imse_std'   : imse_arr.std(),
                    'imse_median': np.median(imse_arr),
                    'ref_paper'  : ref,
                }
                all_rows.append(summary)

                pd.DataFrame([summary]).to_csv(
                    out_csv, mode='a',
                    header=not os.path.exists(out_csv)
                           or os.path.getsize(out_csv) == 0,
                    index=False)

                if verbose:
                    ratio = imse_arr.mean() / ref if not np.isnan(ref) else float('nan')
                    print(f"  IMSE={imse_arr.mean():.4f} "
                          f"(±{imse_arr.std():.4f})  "
                          f"paper={ref:.4f}  ratio={ratio:.2f}"
                          if not np.isnan(ref) else
                          f"  IMSE={imse_arr.mean():.4f} "
                          f"(±{imse_arr.std():.4f})  failed={n_fail}")

                del imse_list, imse_arr
                gc.collect()

    if verbose:
        print(f"\nIMSE grid experiment complete. Saved: {out_csv}")

    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    return pd.DataFrame(all_rows)


def print_imse_comparison(df):
    """
    Print IMSE results side by side with paper Table F.1.
    """
    for (p, q) in df[['p', 'q']].drop_duplicates().itertuples(index=False):
        sub = df[(df['p'] == p) & (df['q'] == q)].copy()
        sub['ratio'] = (sub['imse_mean'] / sub['ref_paper']).round(2)

        print(f"\n[p={p}, q={q}]  IMSE Comparison")
        print(f"{'ρ':>5}  {'n':>5}  {'estimated':>10}  {'paper':>10}  {'ratio':>6}")
        print("-" * 46)
        for _, row in sub.sort_values(['rho', 'n']).iterrows():
            ref = row['ref_paper']
            ref_str = f"{ref:.4f}" if not np.isnan(ref) else "   —  "
            print(f"{row['rho']:>5.2f}  {int(row['n']):>5}  "
                  f"{row['imse_mean']:>10.4f}  {ref_str:>10}  "
                  f"{row['ratio']:>6.2f}" if not np.isnan(row['ratio'])
                  else f"{row['rho']:>5.2f}  {int(row['n']):>5}  "
                  f"{row['imse_mean']:>10.4f}  {ref_str:>10}     —")


# ─────────────────────────────────────────────────────────────
# Grid experiment (core of JSS paper)
# ─────────────────────────────────────────────────────────────

def run_mc_grid(n_sims=30,
                n_list=None, pq_list=None, ratio_list=None,
                sigma_v_base=0.2,
                out_csv='mc_grid_results.csv',
                verbose=True):
    """
    Sample size × (p,q) dimension × σ_u/σ_v ratio grid Monte Carlo experiment.

    For comparison with SW(2023) Appendix F.  n_jobs=1 sequential execution (prevents kernel panic).

    Parameters
    ----------
    n_sims     : replications per grid cell (default 30)
    n_list     : list of sample sizes (default [300, 500, 1000])
    pq_list    : list of (p, q) pairs (default [(1,1),(2,2),(3,2)])
    ratio_list : list of σ_u/σ_v ratios (default [0.5, 1.0, 2.0])
    sigma_v_base : base σ_v (default 0.2)
    out_csv    : CSV file path to save results
    verbose    : whether to print progress

    Returns
    -------
    pd.DataFrame : full grid results
    """
    import gc, os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from core.model import SW2023Model

    if n_list    is None: n_list    = [300, 500, 1000]
    if pq_list   is None: pq_list   = [(1, 1), (2, 2), (3, 2)]
    if ratio_list is None: ratio_list = [0.5, 1.0, 2.0]

    # Load existing results (supports resume after interruption)
    done_keys = set()
    if os.path.exists(out_csv):
        existing = pd.read_csv(out_csv)
        for _, row in existing.iterrows():
            done_keys.add((int(row['n']), int(row['p']), int(row['q']),
                           float(row['ratio'])))
        if verbose:
            print(f"[Resume] Loaded {len(existing)} existing rows → "
                  f"skipping {len(done_keys)} cells")

    all_rows = []
    total_cells = len(n_list) * len(pq_list) * len(ratio_list)
    cell_idx    = 0

    for n in n_list:
        for (p, q) in pq_list:
            for ratio in ratio_list:
                cell_idx += 1
                key = (n, p, q, ratio)

                if key in done_keys:
                    if verbose:
                        print(f"[{cell_idx}/{total_cells}] Skipping: "
                              f"n={n}, p={p}, q={q}, ratio={ratio}")
                    continue

                sigma_u = sigma_v_base * ratio

                if verbose:
                    print(f"\n[{cell_idx}/{total_cells}] "
                          f"n={n}, p={p}, q={q}, "
                          f"σ_u={sigma_u:.3f}, σ_v={sigma_v_base:.3f} "
                          f"(ratio={ratio})")

                cell_results = []
                n_fail = 0

                for sim in range(n_sims):
                    data = dgp_sw_native(n=n, p=p, q=q, seed=sim,
                                         sigma_v=sigma_v_base,
                                         sigma_u=sigma_u)
                    try:
                        m = SW2023Model(data['X'], data['Y'],
                                        direction='mean', method='HMS',
                                        log_transform=True, standardize=True)
                        m.fit(verbose=False)

                        met = validation_metrics(
                            data['eff_true'], m.efficiency_,
                            label=f'sim_{sim}')
                        met['wrong_skew_pct'] = (m.r3_ > 0).mean() * 100
                        met['sigma_u_est']    = float(np.nanmean(m.sigma_eta_))
                        cell_results.append(met)

                        # Immediately free memory
                        del m, data, met
                    except Exception as e:
                        n_fail += 1
                        del data
                    finally:
                        gc.collect()

                if not cell_results:
                    if verbose:
                        print(f"  → All failed (n_fail={n_fail}), skipping")
                    continue

                df_cell = pd.DataFrame(cell_results)
                summary = {
                    'n'            : n,
                    'p'            : p,
                    'q'            : q,
                    'ratio'        : ratio,
                    'sigma_u_true' : sigma_u,
                    'sigma_v'      : sigma_v_base,
                    'n_sims'       : len(cell_results),
                    'n_fail'       : n_fail,
                    'bias_mean'    : df_cell['bias'].mean(),
                    'bias_std'     : df_cell['bias'].std(),
                    'rmse_mean'    : df_cell['rmse'].mean(),
                    'rmse_std'     : df_cell['rmse'].std(),
                    'rho_mean'     : df_cell['spearman_rho'].mean(),
                    'rho_std'      : df_cell['spearman_rho'].std(),
                    'wrong_skew_pct': df_cell['wrong_skew_pct'].mean(),
                    'sigma_u_est_mean': df_cell['sigma_u_est'].mean(),
                    'true_eff_mean': df_cell['true_mean'].mean(),
                    'est_eff_mean' : df_cell['est_mean'].mean(),
                }
                all_rows.append(summary)

                # Immediately append cell result to CSV
                pd.DataFrame([summary]).to_csv(
                    out_csv, mode='a',
                    header=not os.path.exists(out_csv) or os.path.getsize(out_csv) == 0,
                    index=False)

                if verbose:
                    print(f"  bias={summary['bias_mean']:.4f}  "
                          f"RMSE={summary['rmse_mean']:.4f}  "
                          f"ρ={summary['rho_mean']:.4f}  "
                          f"wrong_skew={summary['wrong_skew_pct']:.1f}%  "
                          f"failed={n_fail}/{n_sims}")

                del df_cell, cell_results
                gc.collect()

    if verbose:
        print(f"\nGrid experiment complete. Results saved: {out_csv}")

    if os.path.exists(out_csv):
        return pd.read_csv(out_csv)
    return pd.DataFrame(all_rows)


def print_grid_table(df, metric='rmse_mean'):
    """
    Print grid results as an n × ratio pivot table.

    Parameters
    ----------
    df     : DataFrame returned by run_mc_grid()
    metric : metric to display (default 'rmse_mean')
    """
    for (p, q) in df[['p', 'q']].drop_duplicates().itertuples(index=False):
        sub = df[(df['p'] == p) & (df['q'] == q)]
        pivot = sub.pivot(index='n', columns='ratio', values=metric)
        print(f"\n[p={p}, q={q}]  {metric}")
        print(pivot.to_string(float_format='{:.4f}'.format))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else 'quick'

    if mode == 'quick':
        # Simplest case: p=1, q=1, Z is 1-dimensional
        print("=" * 55)
        print("Monte Carlo Validation (quick test: p=1,q=1,n=500)")
        print("=" * 55)

        print("\n► 2-Component (n=500, p=1, q=1)")
        df2 = run_mc_2component(n_sims=20, n=500, p=1, q=1,
                                 sigma_v=0.2, sigma_u=0.3, verbose=True)

        print("\n► 4-Component panel (N=100, T=5, p=1, q=1)")
        df4 = run_mc_4component(n_sims=20, N=100, T=5, p=1, q=1,
                                 sigma_v=0.15, sigma_u=0.20,
                                 sigma_alpha=0.10, sigma_mu=0.20,
                                 verbose=True)

    elif mode == 'native':
        # Native DGP validation (DGP-model fully consistent)
        print("=" * 55)
        print("Monte Carlo Validation (Native DGP: p=2,q=2)")
        print("=" * 55)

        for n_obs in [300, 500, 1000]:
            print(f"\n► Native DGP (n={n_obs}, p=2, q=2)")
            run_mc_native(n_sims=30, n=n_obs, p=2, q=2,
                          sigma_v=0.2, sigma_u=0.3, verbose=True)

    elif mode == 'full':
        print("=" * 55)
        print("Monte Carlo Validation (full: p=2,q=2,n=1000)")
        print("=" * 55)

        print("\n► 2-Component (n=1000, p=2, q=2)")
        df2 = run_mc_2component(n_sims=50, n=1000, p=2, q=2,
                                 sigma_v=0.2, sigma_u=0.3, verbose=True)

        print("\n► 4-Component panel (N=200, T=5, p=2, q=2)")
        df4 = run_mc_4component(n_sims=50, N=200, T=5, p=2, q=2,
                                 sigma_v=0.15, sigma_u=0.20,
                                 sigma_alpha=0.10, sigma_mu=0.20,
                                 verbose=True)

    elif mode == 'grid':
        # Core of JSS paper: grid experiment (n_jobs=1 sequential, resumable)
        print("=" * 55)
        print("Monte Carlo Grid Experiment (SW 2023 Appendix F comparison)")
        print("=" * 55)
        df_grid = run_mc_grid(
            n_sims=30,
            n_list=[300, 500, 1000],
            pq_list=[(1, 1), (2, 2), (3, 2)],
            ratio_list=[0.5, 1.0, 2.0],
            sigma_v_base=0.2,
            out_csv='mc_grid_results.csv',
            verbose=True,
        )
        print("\n=== RMSE Summary Table ===")
        print_grid_table(df_grid, metric='rmse_mean')
        print("\n=== Spearman ρ Summary Table ===")
        print_grid_table(df_grid, metric='rho_mean')

    elif mode == 'imse':
        # SW(2023) Table F.1 reproduction (direct comparison with paper)
        print("=" * 60)
        print("SW(2023) Table F.1 IMSE Reproduction Experiment")
        print("=" * 60)
        df_imse = run_imse_grid(
            n_sims=100,
            n_list=[100, 200, 400, 800],
            pq_list=[(1,1), (2,2)],
            rho_list=[0.0, 0.5, 1.0, 2.0],
            sigma_eta=0.5,
            method='HMS',
            out_csv='mc_imse_results.csv',
            verbose=True,
        )
        print_imse_comparison(df_imse)

    elif mode == 'imse_quick':
        # Quick check (n_sims=20, small scale)
        print("=" * 60)
        print("SW(2023) Table F.1 IMSE Quick Check")
        print("=" * 60)
        df_imse = run_imse_grid(
            n_sims=20,
            n_list=[100, 400],
            pq_list=[(1,1), (2,2)],
            rho_list=[0.0, 0.5, 1.0, 2.0],
            sigma_eta=0.5,
            method='HMS',
            bandwidth_method='loocv',
            out_csv='mc_imse_quick.csv',
            verbose=True,
        )
        print_imse_comparison(df_imse)

    elif mode == 'grid_quick':
        # Quick grid check (n_sims=10 per cell, small scale)
        print("=" * 55)
        print("Monte Carlo Grid Experiment (Quick Check)")
        print("=" * 55)
        df_grid = run_mc_grid(
            n_sims=10,
            n_list=[300, 500],
            pq_list=[(1, 1), (2, 2)],
            ratio_list=[0.5, 1.0, 2.0],
            sigma_v_base=0.2,
            out_csv='mc_grid_quick.csv',
            verbose=True,
        )
        print_grid_table(df_grid, metric='rmse_mean')

    else:
        # Default settings
        print("=" * 55)
        print("Monte Carlo Validation (n_sims=10, p=2, q=2)")
        print("=" * 55)

        print("\n► 2-Component Validation (n=300, p=2, q=2)")
        df2 = run_mc_2component(n_sims=10, n=300, p=2, q=2,
                                 sigma_v=0.2, sigma_u=0.3, verbose=True)

        print("\n► 4-Component Panel Validation (N=80, T=5, p=2, q=2)")
        df4 = run_mc_4component(n_sims=10, N=80, T=5, p=2, q=2,
                                 sigma_v=0.15, sigma_u=0.20,
                                 sigma_alpha=0.10, sigma_mu=0.20,
                                 verbose=True)
