"""
그림 A: 합성 데이터 — Silverman / LOO-CV scalar / LOO-CV product 프론티어 비교
그림 B: Norway 데이터 — 세 방법의 효율성 분포 비교
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import gaussian_kde
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

np.random.seed(2023)

# ── JSS 스타일 설정 ────────────────────────────────────────────
plt.rcParams.update({
    'font.family'     : 'serif',
    'font.size'       : 10,
    'axes.labelsize'  : 10,
    'axes.titlesize'  : 10,
    'legend.fontsize' : 9,
    'xtick.labelsize' : 9,
    'ytick.labelsize' : 9,
    'figure.dpi'      : 150,
    'axes.spines.top' : False,
    'axes.spines.right': False,
})

COLORS = {
    'silverman'    : '#2166ac',   # 파랑
    'loocv_scalar' : '#d6604d',   # 빨강
    'loocv'        : '#1a9850',   # 초록
    'scalar'       : '#d6604d',   # 빨강 (alias)
    'product'      : '#1a9850',   # 초록 (alias)
    'truth'        : '#000000',   # 검정
}

# =============================================================================
# 그림 A: 합성 데이터 — p=1, q=1 (1D Z), n=300
# =============================================================================
print("그림 A: 합성 데이터 프론티어 추정 중...")

from sw2023 import SW2023Model

def estimate_phi(X, Y, bw_method):
    """φ̂(z) 및 Z 반환 (SW2023Model 내부 속성 이용)."""
    m = SW2023Model(X, Y, method='HMS', bandwidth_method=bw_method,
                    log_transform=True, standardize=True)
    m.fit(verbose=False)
    return m.Z_[:, 0], m.phi_hat_   # 1D Z, φ̂

# DGP: p=1, q=1, 단일 입력-산출
n = 300
x_true = np.sort(np.random.uniform(1, 5, n))
phi_true = 2.0 * np.log(x_true)          # 진프론티어: φ = 2·ln(x)
sigma_v  = 0.20
sigma_u  = 0.30
v = np.random.normal(0, sigma_v, n)
u = np.abs(np.random.normal(0, sigma_u, n))
y_obs = np.exp(phi_true - u + v)

X_syn = x_true.reshape(-1, 1)
Y_syn = y_obs.reshape(-1, 1)

# 세 방법으로 추정
results_A = {}
for bw in ['silverman', 'loocv_scalar', 'loocv']:
    z_vals, phi_hat = estimate_phi(X_syn, Y_syn, bw)
    # Z를 원래 x 척도로 정렬
    order = np.argsort(z_vals)
    results_A[bw] = (z_vals[order], phi_hat[order])
    print(f"  {bw}: mean φ̂ = {phi_hat.mean():.4f}")

# 진프론티어도 같은 Z 척도로
z_ref = results_A['silverman'][0]

# ── 그림 A 플롯 ───────────────────────────────────────────────
fig_A, axes = plt.subplots(1, 2, figsize=(10, 4))

ax = axes[0]
# 산점도 (관측치)
z_obs, _ = results_A['silverman']
ax.scatter(z_obs, results_A['silverman'][1]*0 + 0,  # 위치용
           alpha=0, s=0)  # dummy

# 효율성 점수 대신 φ̂ 자체를 그린다
for bw, label, ls in [
    ('silverman',   'Silverman',          '-'),
    ('loocv_scalar','LOO-CV scalar',      '--'),
    ('loocv',       'LOO-CV product',     ':'),
]:
    zz, pp = results_A[bw]
    ax.plot(zz, pp, color=COLORS[bw], ls=ls, lw=1.8, label=label)

ax.set_xlabel('Projected input $Z$')
ax.set_ylabel(r'Estimated frontier $\hat{\varphi}(Z)$')
ax.set_title('(a) Frontier estimates on synthetic data\n'
             r'($p=1,\,q=1,\,n=300$)')
ax.legend(frameon=False)

# 잔차(효율성) 분포 비교
ax2 = axes[1]
from sw2023 import SW2023Model
effs_A = {}
for bw, label, ls in [
    ('silverman',   'Silverman',     '-'),
    ('loocv_scalar','LOO-CV scalar', '--'),
    ('loocv',       'LOO-CV product',':'),
]:
    m = SW2023Model(X_syn, Y_syn, method='HMS', bandwidth_method=bw,
                    log_transform=True, standardize=True)
    m.fit(verbose=False)
    effs_A[bw] = m.efficiency_

x_grid = np.linspace(0, 1, 300)
for bw, label, ls in [
    ('silverman',   'Silverman',     '-'),
    ('loocv_scalar','LOO-CV scalar', '--'),
    ('loocv',       'LOO-CV product',':'),
]:
    kde = gaussian_kde(effs_A[bw], bw_method=0.15)
    ax2.plot(x_grid, kde(x_grid), color=COLORS[bw], ls=ls, lw=1.8, label=label)
    ax2.axvline(effs_A[bw].mean(), color=COLORS[bw], ls=ls, lw=0.8, alpha=0.6)

ax2.set_xlabel('Efficiency score')
ax2.set_ylabel('Density')
ax2.set_title('(b) Efficiency distributions on synthetic data\n'
              r'(vertical lines = means)')
ax2.legend(frameon=False)

fig_A.tight_layout()
fig_A.savefig('fig_synthetic_comparison.pdf', bbox_inches='tight')
fig_A.savefig('fig_synthetic_comparison.png', bbox_inches='tight', dpi=150)
print("  → fig_synthetic_comparison.pdf/png 저장")

# =============================================================================
# 그림 B: Norway 데이터 — 세 방법 효율성 분포 + 연도별 추이
# =============================================================================
print("\n그림 B: Norway 효율성 분포 비교 중...")

df = pd.read_csv('norway_loocv_comparison.csv')

fig_B, axes = plt.subplots(1, 2, figsize=(10, 4))

# ── B-1: 밀도 플롯 ────────────────────────────────────────────
ax = axes[0]
labels = {
    'eff_silverman' : ('Silverman',      '-',  COLORS['silverman']),
    'eff_scalar'    : ('LOO-CV scalar',  '--', COLORS['scalar']),
    'eff_product'   : ('LOO-CV product', ':',  COLORS['product']),
}
x_grid = np.linspace(0, 1.05, 400)
for col, (label, ls, color) in labels.items():
    vals = df[col].dropna().values
    kde  = gaussian_kde(vals, bw_method=0.08)
    ax.plot(x_grid, kde(x_grid), color=color, ls=ls, lw=1.8, label=label)
    ax.axvline(vals.mean(), color=color, ls=ls, lw=0.8, alpha=0.7)

ax.set_xlabel('Efficiency score')
ax.set_ylabel('Density')
ax.set_title('(a) Efficiency score distributions\n'
             'Norwegian farms, 1998–2006  ($n=2{,}729$)')
ax.legend(frameon=False)
ax.set_xlim(0.1, 1.05)

# ── B-2: 연도별 평균 효율성 추이 ──────────────────────────────
ax2 = axes[1]
yearly = df.groupby('year')[['eff_silverman','eff_scalar','eff_product',
                              'TE_silverman','PE_silverman']].mean()

years = yearly.index.astype(int)
for col, (label, ls, color) in labels.items():
    ax2.plot(years, yearly[col], color=color, ls=ls, lw=1.8,
             marker='o', ms=4, label=label)

# TE, PE (패널 — silverman)
ax2.plot(years, yearly['TE_silverman'], color='#999999', ls='-.',
         lw=1.4, marker='s', ms=3, label='TE (4-comp, Silverman)')
ax2.plot(years, yearly['PE_silverman'], color='#555555', ls='-.',
         lw=1.4, marker='^', ms=3, label='PE (4-comp, Silverman)')

ax2.set_xlabel('Year')
ax2.set_ylabel('Mean efficiency')
ax2.set_title('(b) Mean efficiency by year\n'
             'Norwegian farms  (bandwidth methods + panel decomposition)')
ax2.set_xticks(years)
ax2.set_xticklabels(years, rotation=45)
ax2.legend(frameon=False, fontsize=8, ncol=2)
ax2.set_ylim(0.5, 1.02)

fig_B.tight_layout()
fig_B.savefig('fig_norway_comparison.pdf', bbox_inches='tight')
fig_B.savefig('fig_norway_comparison.png', bbox_inches='tight', dpi=150)
print("  → fig_norway_comparison.pdf/png 저장")

print("\n완료.")
