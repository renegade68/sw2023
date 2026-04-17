"""
Visualization Module

Visualize results from SW2023Model and PanelSW2023.

Functions:
  plot_efficiency_dist   : Efficiency distribution histogram
  plot_efficiency_rank   : Efficiency rank plot
  plot_frontier_1d       : 1D frontier (U vs Z[0])
  plot_panel_trend       : Panel yearly efficiency trend
  plot_decomposition     : 4-component decomposition (TE vs PE)
"""

import numpy as np
import warnings

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    import matplotlib.font_manager as _fm

    # Auto-configure font (Noto Sans CJK KR → Apple SD Gothic → fallback)
    _korean_candidates = ['Noto Sans CJK KR', 'Apple SD Gothic Neo',
                           'Malgun Gothic', 'NanumGothic']
    _available = {f.name for f in _fm.fontManager.ttflist}
    _kor_font  = next((f for f in _korean_candidates if f in _available), None)
    if _kor_font:
        matplotlib.rcParams['font.family'] = _kor_font
    matplotlib.rcParams['axes.unicode_minus'] = False  # prevent minus sign rendering issues

    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    warnings.warn(
        "matplotlib not found. To use visualization features, "
        "run `pip install matplotlib`.",
        ImportWarning
    )


def _check_mpl():
    if not HAS_MPL:
        raise ImportError(
            "matplotlib is required. Run `pip install matplotlib`."
        )


# ─────────────────────────────────────────────────────────────
# 1. Efficiency distribution
# ─────────────────────────────────────────────────────────────
def plot_efficiency_dist(efficiency, title='Efficiency Distribution',
                          bins=30, figsize=(7, 4),
                          color='steelblue', ax=None):
    """
    Histogram of efficiency index values.

    Parameters
    ----------
    efficiency : array-like  (n,) efficiency values [0,1]
    title      : str
    bins       : int
    figsize    : tuple
    color      : str
    ax         : matplotlib Axes or None
    """
    _check_mpl()
    eff = np.asarray(efficiency)
    eff = eff[~np.isnan(eff)]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.hist(eff, bins=bins, color=color, edgecolor='white',
             alpha=0.8, density=True)

    mean_e = eff.mean()
    med_e  = np.median(eff)
    ax.axvline(mean_e, color='firebrick', linestyle='--', linewidth=1.5,
                label=f'Mean = {mean_e:.3f}')
    ax.axvline(med_e,  color='darkorange', linestyle=':', linewidth=1.5,
                label=f'Median = {med_e:.3f}')

    ax.set_xlabel('Efficiency Index')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# 2. Efficiency rank plot
# ─────────────────────────────────────────────────────────────
def plot_efficiency_rank(efficiency, labels=None,
                          ci=None, title='Efficiency Ranking',
                          figsize=(10, 5), ax=None,
                          top_n=None):
    """
    Efficiency rank plot (Caterpillar plot).

    Parameters
    ----------
    efficiency : (n,) efficiency values
    labels     : (n,) labels (None → index)
    ci         : (n, 2) confidence intervals [lo, hi] (optional)
    title      : str
    figsize    : tuple
    ax         : Axes or None
    top_n      : int, show only top and bottom n observations (None → all)
    """
    _check_mpl()
    eff = np.asarray(efficiency)
    mask = ~np.isnan(eff)
    eff  = eff[mask]

    order = np.argsort(eff)
    eff_s = eff[order]

    if labels is not None:
        labels = np.asarray(labels)[mask][order]
    else:
        labels = order

    if ci is not None:
        ci = np.asarray(ci)[mask][order]

    if top_n is not None:
        half = top_n // 2
        idx  = np.concatenate([np.arange(half),
                                 np.arange(len(eff_s)-half, len(eff_s))])
        eff_s  = eff_s[idx]
        labels = labels[idx]
        if ci is not None:
            ci = ci[idx]

    n = len(eff_s)
    x = np.arange(n)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(x, eff_s, s=8, color='steelblue', zorder=3)

    if ci is not None:
        lo = ci[:, 0]
        hi = ci[:, 1]
        ax.vlines(x, lo, hi, colors='gray', alpha=0.4, linewidth=0.8)

    ax.axhline(eff_s.mean(), color='firebrick', linestyle='--',
                linewidth=1, label=f'Mean={eff_s.mean():.3f}')
    ax.set_xlabel('Rank')
    ax.set_ylabel('Efficiency Index')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# 3. 1D frontier scatter plot (Z vs U)
# ─────────────────────────────────────────────────────────────
def plot_frontier_1d(model, dim=0, n_grid=200,
                      title='Frontier Estimate (U vs Z)',
                      figsize=(7, 5), ax=None):
    """
    Scatter plot of U vs Z[dim] with estimated frontier phi_hat(Z).

    Parameters
    ----------
    model : SW2023Model (fitted)
    dim   : int  which dimension of Z to use as x-axis
    """
    _check_mpl()

    Z   = model.Z_
    U   = model.U_
    phi = model.phi_hat_
    eff = model.efficiency_

    z_dim = Z[:, dim] if Z.ndim > 1 else Z

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    sc = ax.scatter(z_dim, U, c=eff, cmap='RdYlGn',
                     s=8, alpha=0.6, vmin=0, vmax=1)
    plt.colorbar(sc, ax=ax, label='Efficiency')

    # phi_hat line
    sort_idx = np.argsort(z_dim)
    ax.plot(z_dim[sort_idx], phi[sort_idx],
             color='navy', linewidth=1.5, label='phi_hat(Z)')

    ax.set_xlabel(f'Z[{dim}]')
    ax.set_ylabel('U')
    ax.set_title(title)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# 4. Panel yearly trend
# ─────────────────────────────────────────────────────────────
def plot_panel_trend(model, time_id, figsize=(8, 5),
                      title='Mean Efficiency by Period',
                      ax=None):
    """
    Yearly efficiency trend from PanelSW2023 results.

    Parameters
    ----------
    model   : PanelSW2023 (fitted)
    time_id : array-like (n,) period ID
    """
    _check_mpl()

    import pandas as pd
    df = pd.DataFrame({
        'time'   : time_id,
        'eff'    : model.efficiency_,
        'te'     : model.eff_transient_,
        'pe'     : model.eff_persistent_,
    }).groupby('time').mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.plot(df.index, df['eff'], 'o-', label='Overall', color='steelblue')
    ax.plot(df.index, df['te'],  's--', label='Transient', color='tomato')
    ax.plot(df.index, df['pe'],  '^:', label='Persistent', color='seagreen')

    ax.set_xlabel('Period')
    ax.set_ylabel('Mean Efficiency')
    ax.set_title(title)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# 5. 4-component decomposition scatter plot
# ─────────────────────────────────────────────────────────────
def plot_decomposition(model, figsize=(6, 6),
                        title='Transient vs Persistent Efficiency',
                        ax=None):
    """
    TE vs PE scatter plot (4-component decomposition).

    Parameters
    ----------
    model : PanelSW2023 (fitted)
    """
    _check_mpl()

    te = model.eff_transient_
    pe = model.eff_persistent_

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    ax.scatter(pe, te, s=8, alpha=0.4, color='steelblue')
    ax.axline((0.5, 0.5), slope=1, color='gray',
               linestyle='--', linewidth=0.8, label='TE=PE')

    ax.set_xlabel('Persistent Efficiency (PE)')
    ax.set_ylabel('Transient Efficiency (TE)')
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    plt.tight_layout()
    return fig, ax


# ─────────────────────────────────────────────────────────────
# 6. Comprehensive dashboard
# ─────────────────────────────────────────────────────────────
def dashboard_crosssection(model, figsize=(12, 8), title='SW(2023) Results'):
    """
    Comprehensive dashboard for 2-component model.

    Panels:
      [0,0] Efficiency distribution
      [0,1] Efficiency ranking
      [1,0] Z vs U scatter plot
      [1,1] Residual distribution (r3 sign distribution)
    """
    _check_mpl()
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=13)

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # (0,0) Efficiency distribution
    ax0 = fig.add_subplot(gs[0, 0])
    plot_efficiency_dist(model.efficiency_, title='Efficiency Distribution', ax=ax0)

    # (0,1) Efficiency ranking
    ax1 = fig.add_subplot(gs[0, 1])
    plot_efficiency_rank(model.efficiency_, title='Efficiency Ranking', ax=ax1)

    # (1,0) Z vs U
    ax2 = fig.add_subplot(gs[1, 0])
    plot_frontier_1d(model, dim=0,
                      title='Frontier Estimate (Z[0] vs U)', ax=ax2)

    # (1,1) r3 distribution
    ax3 = fig.add_subplot(gs[1, 1])
    r3_vals = model.r3_
    ax3.hist(r3_vals[r3_vals < 0], bins=25, color='steelblue',
              alpha=0.7, label=f'r3<0: {(r3_vals<0).mean()*100:.1f}%',
              density=True)
    ax3.hist(r3_vals[r3_vals >= 0], bins=25, color='tomato',
              alpha=0.7, label=f'r3>=0: {(r3_vals>=0).mean()*100:.1f}%',
              density=True)
    ax3.set_xlabel('r3 (3rd conditional moment)')
    ax3.set_ylabel('Density')
    ax3.set_title('r3 Sign Distribution (negative is normal)')
    ax3.legend(fontsize=9)

    return fig
