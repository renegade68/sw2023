"""
LOO-CV 방법 비교: scalar vs product kernel
  - Norway 실증 예제 (SW2023Model + PanelSW2023)
  - Monte Carlo 시뮬레이션 (IMSE 비율)

결과 저장:
  norway_loocv_comparison.csv   — 연도별 TE/PE 비교
  mc_imse_product.csv           — product LOO-CV IMSE 비율 (Table 1 비교용)
"""

import sys, os, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from sw2023 import SW2023Model, PanelSW2023
from sw2023.tests.monte_carlo import run_imse_grid

np.random.seed(2023)

# =============================================================================
# Part 1: Norway 실증 예제 — Silverman / loocv_scalar / loocv(product) 비교
# =============================================================================
print("=" * 65)
print("Part 1: Norway Agricultural Panel — Bandwidth Method Comparison")
print("=" * 65)

DATA_PATH = os.path.join(os.path.dirname(__file__), 'sfbook_data', 'norway.dta')
df = pd.read_stata(DATA_PATH).sort_values(['farmid', 'year']).reset_index(drop=True)

X       = df[['x1','x2','x3','x4','x5','x6']].values.astype(float)
Y       = df[['y1','y2','y3','y4']].values.astype(float)
firm_id = df['farmid'].values
time_id = df['year'].values

print(f"n={len(df)}, farms={df['farmid'].nunique()}, "
      f"years={int(df['year'].min())}~{int(df['year'].max())}\n")

results_cs = {}
for bw in ['silverman', 'loocv_scalar', 'loocv']:
    print(f"--- SW2023Model  bandwidth={bw} ---")
    t0 = time.time()
    m = SW2023Model(X, Y, method='HMS', bandwidth_method=bw)
    m.fit(verbose=False)
    dt = time.time() - t0
    eff = m.efficiency_
    print(f"  mean={eff.mean():.4f}  median={np.median(eff):.4f}  "
          f"std={eff.std():.4f}  wrong_skew={getattr(m,'wrong_skewness_rate_',float('nan')):.1%}"
          f"  [{dt:.0f}s]")
    results_cs[bw] = eff

# Panel model (Silverman only for comparison — loocv on panel takes very long)
print(f"\n--- PanelSW2023  bandwidth=silverman ---")
t0 = time.time()
m2 = PanelSW2023(X, Y, firm_id, time_id, method='HMS')
m2.fit(verbose=False)
dt = time.time() - t0
print(f"  TE mean={m2.eff_transient_.mean():.4f}  "
      f"PE mean={m2.eff_persistent_.mean():.4f}  [{dt:.0f}s]")

# Save cross-sectional comparison
comp = pd.DataFrame({
    'year'          : time_id,
    'farmid'        : firm_id,
    'eff_silverman' : results_cs['silverman'],
    'eff_scalar'    : results_cs['loocv_scalar'],
    'eff_product'   : results_cs['loocv'],
    'TE_silverman'  : m2.eff_transient_,
    'PE_silverman'  : m2.eff_persistent_,
})

out_cs = os.path.join(os.path.dirname(__file__), 'norway_loocv_comparison.csv')
comp.to_csv(out_cs, index=False)
print(f"\nNorway comparison saved → {out_cs}")

# Yearly summary
print("\n[연도별 평균 효율성 비교]")
yrly = comp.groupby('year')[['eff_silverman','eff_scalar','eff_product','TE_silverman','PE_silverman']].mean()
print(yrly.round(4).to_string())

# =============================================================================
# Part 2: Monte Carlo — product LOO-CV IMSE 비율
# =============================================================================
print("\n" + "=" * 65)
print("Part 2: Monte Carlo — product LOO-CV  (bandwidth_method='loocv')")
print("=" * 65)
print("(이미 mc_imse_results.csv = loocv_scalar 기준)")
print("product LOO-CV 결과를 mc_imse_product.csv 에 저장합니다.\n")

out_mc = os.path.join(os.path.dirname(__file__), 'mc_imse_product.csv')

run_imse_grid(
    n_sims=100,
    sigma_eta=0.5,
    method='HMS',
    bandwidth_method='loocv',      # product kernel
    out_csv=out_mc,
    verbose=True,
)

# =============================================================================
# Part 3: 두 결과 비교 출력
# =============================================================================
print("\n" + "=" * 65)
print("Part 3: IMSE 비율 비교  (scalar vs product LOO-CV)")
print("=" * 65)

scalar_csv = os.path.join(os.path.dirname(__file__), 'mc_imse_results.csv')
product_csv = out_mc

if os.path.exists(scalar_csv) and os.path.exists(product_csv):
    sc = pd.read_csv(scalar_csv).set_index(['p','q','rho','n'])
    pr = pd.read_csv(product_csv).set_index(['p','q','rho','n'])
    common = sc.index.intersection(pr.index)
    cmp = pd.DataFrame({
        'scalar' : sc.loc[common, 'imse_ratio'],
        'product': pr.loc[common, 'imse_ratio'],
    })
    cmp['diff'] = cmp['product'] - cmp['scalar']
    print(cmp.round(3).to_string())
    print(f"\nMean absolute diff: {cmp['diff'].abs().mean():.4f}")
else:
    print("(scalar CSV 없음 — 비교 불가)")

print("\nDone.")
