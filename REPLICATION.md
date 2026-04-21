# Replication Guide

**Paper:** Lee, C. (2026). sw2023: Nonparametric Multiple-Output Stochastic
Frontier Analysis in Python. *Journal of Statistical Software* (submitted).

---

## Contents of This Archive

| File | Description |
|---|---|
| `replication.py` | Replication script (command-line) |
| `replication.ipynb` | Replication notebook (Jupyter, inline outputs) |
| `REPLICATION.md` | This file |
| `requirements.txt` | Python package dependencies |
| `mc_imse_results.csv` | Pre-computed Monte Carlo results — scalar LOO-CV (Table 1) |
| `mc_imse_product.csv` | Pre-computed Monte Carlo results — product LOO-CV (Table 2) |
| `norway_for_python.csv` | Norwegian agricultural panel data (Section 7) |

The `sw2023` package is submitted separately as an installable tarball
(`sw2023-0.3.2.tar.gz`).  See Step 1 below.

---

## Requirements

- Python >= 3.8
- numpy >= 1.21
- scipy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4

---

## How to Run

**Step 1.** Install the `sw2023` package from the accompanying tarball:

```bash
pip install sw2023-0.3.2.tar.gz
```

Or, equivalently, from PyPI:

```bash
pip install sw2023==0.3.2
```

To install all dependencies at once:

```bash
pip install -r requirements.txt
pip install sw2023-0.3.2.tar.gz
```

**Step 2.** Unzip the replication archive and run the script:

```bash
unzip sw2023_replication.zip
cd sw2023_replication

# Quick verification — < 10 minutes (n_sims=20, n up to 400)
python replication.py

# Full replication — ~ 60 minutes (n_sims=100, n up to 800, matches paper)
python replication.py --full

# Tables only (skip figure generation)
python replication.py --full --tables

# Figures only (skip Monte Carlo)
python replication.py --figures
```

---

## What the Script Reproduces

### Tables 1 & 2 — Monte Carlo Validation (Section 5)

The script runs a fresh Monte Carlo experiment and **generates** the CSV
files, then displays the pre-computed reference values.

- **Table 1** (`mc_imse_results.csv`): IMSE ratios for scalar LOO-CV
  bandwidth, (p,q) ∈ {(1,1),(1,2),(2,1),(2,2)}, n ∈ {100,200,400,**800**}
  with `--full`, ρ = 0.
- **Table 2** (`mc_imse_product.csv`): Same configurations with product
  LOO-CV bandwidth.

With `--full` (n_sims=100, n up to 800), results match the paper exactly.

### Table 3 — Simulation-Based Illustration (Section 6)

Reproduces all code-output blocks in Section 6 of the manuscript.
Random seed is fixed (`np.random.seed(42)`), so results are exactly reproducible.

Expected output (key figures):

| Output | Expected value |
|---|---|
| Mean efficiency | 0.5311 |
| Std efficiency | 0.2538 |
| Mean sigma_eta | 1.0551 |
| Mean SE(phi_hat) | 0.3745 |
| Mean CI width (bootstrap) | 2.0043 |
| Wild bootstrap p-value | 0.8819 (seed=2023, B=999; H0 not rejected) |

### Table 4 — Norwegian Agricultural Panel (Section 7)

Uses `norway_for_python.csv` (Kumbhakar, Wang & Horncastle 2015).
Reproduces cross-sectional efficiency (Table 3 summary statistics) and
yearly transient/persistent efficiency means (Table 4).

Expected output (key figures):

| Output | Expected value |
|---|---|
| Mean efficiency (CS) | 0.813 |
| Median efficiency (CS) | 0.831 |
| Wrong-skewness proportion | 50.6% |
| Mean transient efficiency (TE) | 0.961 |
| Mean persistent efficiency (PE) | 0.978 |

---

## Tested Environment

```
Python  : 3.10
numpy   : 2.2.6
scipy   : 1.15.3
pandas  : 2.3.3
sw2023  : 0.3.2
```

---

## Contact

Choonjoo Lee
Department of AI and Robotics, Korea National Defense University
