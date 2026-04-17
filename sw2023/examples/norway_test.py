"""
Test SW(2023) basic model on Norway agricultural panel data.

Data: Kumbhakar, Wang & Horncastle (2015) textbook norway.dta
  - 4 outputs: y1, y2, y3, y4
  - 6 inputs: x1 ~ x6
  - Panel: 460 farms x up to 9 years (1998-2006)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
from sw2023 import SW2023Model

# ── Load data ────────────────────────────────────────────────────────────
DATA_PATH = '/Users/mac/Documents/msfa/sfbook_data/norway.dta'
df = pd.read_stata(DATA_PATH)

print(f"Data size: {df.shape}")
print(f"Farms: {df['farmid'].nunique()}, years: {df['year'].min()}~{df['year'].max()}")
print()

# ── Variable selection ───────────────────────────────────────────────────
input_vars  = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6']
output_vars = ['y1', 'y2', 'y3', 'y4']

X = df[input_vars].values.astype(float)
Y = df[output_vars].values.astype(float)

print(f"Inputs: {input_vars}")
print(f"Outputs: {output_vars}")
print()

# ── Small-scale test (first 200 observations) ────────────────────────────
print("=== Small-scale test (n=200) ===")
n_test = 200
X_small = X[:n_test]
Y_small = Y[:n_test]

model = SW2023Model(
    X_small, Y_small,
    direction='mean',
    method='HMS'
)
model.fit(verbose=True)

results = model.summary()

# Efficiency distribution
eff = model.efficiency_
print(f"\nEfficiency distribution:")
print(f"  Min   : {eff.min():.4f}")
print(f"  25%   : {np.percentile(eff, 25):.4f}")
print(f"  Median: {np.median(eff):.4f}")
print(f"  75%   : {np.percentile(eff, 75):.4f}")
print(f"  Max   : {eff.max():.4f}")
print(f"  Mean  : {eff.mean():.4f}")
