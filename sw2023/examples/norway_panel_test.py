"""
Norway agricultural panel data: 4-Component Panel SW(2023) test.

Comparison:
  - SW2023Model  : basic cross-sectional model (2-component)
  - PanelSW2023  : 4-component panel (transient + persistent inefficiency)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import time
from sw2023 import SW2023Model, PanelSW2023

# ── Load data ─────────────────────────────────────────────────────────────
df = pd.read_stata('/Users/mac/Documents/msfa/sfbook_data/norway.dta')
df = df.sort_values(['farmid', 'year']).reset_index(drop=True)

input_vars  = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
output_vars = ['y1', 'y2', 'y3', 'y4']

X       = df[input_vars].values.astype(float)
Y       = df[output_vars].values.astype(float)
firm_id = df['farmid'].values
time_id = df['year'].values

print(f"Data: n={len(df)}, farms={df['farmid'].nunique()}, "
      f"years={df['year'].min()}~{df['year'].max()}")
print()

# ══════════════════════════════════════════════════════════════════════════
# Model 1: Basic SW(2023) — 2-component
# ══════════════════════════════════════════════════════════════════════════
print("Model 1: SW(2023) basic model (2-component)")
t0 = time.time()
m1 = SW2023Model(X, Y, method='HMS')
m1.fit(verbose=True)
print(f"  Elapsed: {time.time()-t0:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════
# Model 2: 4-Component Panel SW
# ══════════════════════════════════════════════════════════════════════════
print("Model 2: 4-Component Panel SW")
t0 = time.time()
m2 = PanelSW2023(X, Y, firm_id, time_id, method='HMS')
m2.fit(verbose=True)
print(f"  Elapsed: {time.time()-t0:.1f}s\n")

# ══════════════════════════════════════════════════════════════════════════
# Result comparison
# ══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("Model comparison")
print("=" * 60)

res2 = m2.summary()

print("\n[2-component vs. 4-component overall efficiency comparison]")
comp = pd.DataFrame({
    '2comp_eff' : m1.efficiency_,
    '4comp_eff' : m2.efficiency_,
    '4comp_TE'  : m2.eff_transient_,
    '4comp_PE'  : m2.eff_persistent_,
    'year'      : time_id,
    'farmid'    : firm_id,
})
print(comp[['2comp_eff','4comp_eff','4comp_TE','4comp_PE']].describe().round(4))

print("\n[Mean efficiency by year]")
yearly = comp.groupby('year')[['2comp_eff','4comp_eff','4comp_TE','4comp_PE']].mean()
print(yearly.round(4))

print("\n[Persistent inefficiency: bottom/top 10 farms]")
firm_pe = comp.groupby('farmid')['4comp_PE'].mean().sort_values()
print("Bottom (most inefficient):")
print(firm_pe.head(5).round(4))
print("Top (most efficient):")
print(firm_pe.tail(5).round(4))
