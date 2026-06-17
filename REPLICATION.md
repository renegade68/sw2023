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
| `mc_imse_extra.csv` | Pre-computed scalar LOO-CV extension results for larger `(p,q)` settings |
| `mc_imse_product.csv` | Pre-computed Monte Carlo results — product LOO-CV (Table 2) |
| `norway_for_python.csv` | Norwegian agricultural panel data (Section 7) |
| `norway_loocv_comparison.csv` | Pre-computed Norway bandwidth comparison data used for Figure 3 |
| `viz_rotation_3d.py` | Source script for the direction-vector rotation figure |

The `sw2023` package is submitted separately as an installable tarball
(`sw2023-0.3.2.tar.gz`).  Place the tarball in the same directory as this
file before installing the requirements.

---

## Requirements

- Python >= 3.8
- sw2023 0.3.2 (installed from the accompanying source tarball)
- numpy >= 1.21
- scipy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4

---

## How to Run

For operating-system-specific copy-and-paste commands, see the top-level
`README_REPLICATION.txt` included with the submission archive. The commands
below use a Unix-style shell (`python3`, `source`/`.` activation, and `unzip`).

**Step 1.** Unzip the replication archive and enter the folder that contains
`requirements.txt`:

```bash
mkdir -p sw2023_replication
unzip -o sw2023_replication.zip -d sw2023_replication
cd sw2023_replication
```

If your system uses `python` for Python 3, replace `python3` with `python`
throughout the commands below.

**Step 2.** Install the package and dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Confirm that `sw2023` is installed in the same Python environment:

```bash
python3 -c "import sw2023; print(sw2023.__version__)"
```

If this command reports `ModuleNotFoundError: No module named 'sw2023'`, the
installation step did not complete in the Python environment used to run the
script. Re-run the installation command from the folder containing
`requirements.txt` and `sw2023-0.3.2.tar.gz`, and check the pip output for
errors:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip show sw2023
```

For a clean isolated environment, use:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -c "import sw2023; print(sw2023.__version__)"
```

The first line of `requirements.txt` installs the accompanying source tarball:

```text
./sw2023-0.3.2.tar.gz
```

If the tarball is not in the same directory, install it explicitly before
running the replication script:

```bash
python3 -m pip install /path/to/sw2023-0.3.2.tar.gz
python3 -m pip install -r requirements.txt
```

**Step 3.** Run the script:

```bash
# Quick verification — < 10 minutes (n_sims=20, n up to 400)
python3 replication.py

# Longer fresh execution check — ~ 60 minutes (n_sims=100)
python3 replication.py --full

# Tables only (skip figure generation)
python3 replication.py --full --tables

# Figures only (skip Monte Carlo)
python3 replication.py --figures
```

The script automatically writes a log file in the replication folder:

- `replication_tables_quick.log` for `python3 replication.py --tables`
- `replication_tables_full.log` for `python3 replication.py --full --tables`
- `replication_figures.log` for `python3 replication.py --figures`

The table run also writes fresh Monte Carlo check CSV files
(`mc_imse_results_quick.csv` and `mc_imse_product_quick.csv`, or the
corresponding `_full.csv` files in full mode). The figure run writes the
manuscript figure files (`fig_rotation_3d.*`, `fig_synthetic_comparison.*`,
and `fig_norway_comparison.*`).

For ease of comparison with the manuscript, the table log prints compact
``Manuscript Table 1 ratio layout'' and ``Manuscript Table 2 ratio layout''
blocks computed from the pre-computed CSV files. These are formatting summaries
of the archived results, not new estimates.

---

## What the Script Reproduces

### Tables 1 & 2 — Monte Carlo Validation (Section 5)

The script reports two Monte Carlo outputs.  The Monte Carlo exercise is a
focused implementation validation against selected published Table F.1 entries,
not a full replication of the entire Simar and Wilson Monte Carlo appendix.

First, it prints the pre-computed Monte Carlo results used in the manuscript
(`n_sims=100`). These values are stored in the accompanying CSV files and
allow exact reproduction of the numerical values reported in this manuscript.
The original authors' computational code and optimizer settings were not
publicly available, so the validation is based on the published mathematical
specification and reference table values.

Second, it performs a fresh Monte Carlo run as an executable check. By default
this quick run uses `n_sims=20` and writes `*_quick.csv` files. These fresh
quick-run values are stochastic checks of the code path and are not expected
to match the manuscript tables cell by cell. Running `python3 replication.py
--full` performs a longer fresh check with `n_sims=100` and writes
`*_full.csv` files.

- **Table 1** (`mc_imse_results.csv`, `mc_imse_extra.csv`): IMSE ratios
  for scalar LOO-CV bandwidth, (p,q) ∈ {(1,1),(1,2),(2,2),(2,3),(3,3)},
  n ∈ {100,200,400}, ρ = 0.
- **Table 2** (`mc_imse_product.csv`): Same configurations with product
  LOO-CV bandwidth for (p,q) ∈ {(1,1),(2,2)} and ρ ∈ {0,0.5,1,2}.

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
| Bootstrap T range | [0.0144, 0.2358] |

### Tables 4-5 — Norwegian Agricultural Panel (Section 7)

Uses `norway_for_python.csv` (Kumbhakar, Wang & Horncastle 2015).
Reproduces the Norwegian panel summary and cross-sectional efficiency
statistics reported with manuscript Table 4, and the yearly
transient/persistent efficiency means reported in manuscript Table 5.

Expected output (key figures):

| Output | Expected value |
|---|---|
| Mean efficiency (CS) | 0.813 |
| Median efficiency (CS) | 0.831 |
| Wrong-skewness proportion | 50.6% |
| Mean transient efficiency (TE) | 0.961 |
| Mean persistent efficiency (PE) | 0.978 |

### Figures

Running `python3 replication.py --figures` reproduces the manuscript figures:

- `fig_rotation_3d.pdf/png` from `viz_rotation_3d.py`.
- `fig_synthetic_comparison.pdf/png` from the synthetic simulation in
  `replication.py`.
- `fig_norway_comparison.pdf/png` from `norway_for_python.csv` and the
  pre-computed `norway_loocv_comparison.csv`.

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
