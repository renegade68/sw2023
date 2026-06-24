# Replication Guide

**Paper:** Lee, C. (2026). sw2023: Nonparametric Multiple-Output Stochastic
Frontier Analysis in Python. *Journal of Statistical Software* (submitted).

---

## Contents of This Repository

| File | Description |
|---|---|
| `replication.py` | Replication script (command-line) |
| `replication.ipynb` | Replication notebook (Jupyter, inline outputs) |
| `REPLICATION.md` | This file |
| `requirements.txt` | Python package dependencies |
| `mc_imse_results.csv` | Archived scalar LOO-CV Monte Carlo results for comparison |
| `mc_imse_extra.csv` | Archived scalar LOO-CV extension results for comparison |
| `mc_imse_product.csv` | Archived product LOO-CV Monte Carlo results for comparison |
| `norway_for_python.csv` | Norwegian agricultural panel data (Section 7) |
| `viz_rotation_3d.py` | Source script for the direction-vector rotation figure |

For the journal submission archive, `requirements.txt` installs the submitted
source tarball. In this GitHub repository, install the package from the checked
out source tree as shown below.

---

## Requirements

- Python >= 3.8
- sw2023 0.3.2 (installed from this source tree)
- numpy >= 1.21
- scipy >= 1.7
- pandas >= 1.3
- matplotlib >= 3.4

---

## How to Run

The commands below use a Unix-style shell (`python3` and `source`/`.`
activation).

**Step 1.** Clone or download this repository and enter the repository folder:

```bash
cd github_sw2023
```

If your system uses `python` for Python 3, replace `python3` with `python`
throughout the commands below.

**Step 2.** Install the package and dependencies:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install .
```

Confirm that `sw2023` is installed in the same Python environment:

```bash
python3 -c "import sw2023; print(sw2023.__version__)"
```

If this command reports `ModuleNotFoundError: No module named 'sw2023'`, the
installation step did not complete in the Python environment used to run the
script. Re-run the installation commands from the repository root and check the
pip output for errors:

```bash
python3 -m pip install -r requirements.txt
python3 -m pip install .
python3 -m pip show sw2023
```

For a clean isolated environment, use:

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m pip install .
python3 -c "import sw2023; print(sw2023.__version__)"
```

**Step 3.** Run the script:

```bash
# Quick executable check — < 10 minutes (n_sims=20, n up to 400)
python3 replication.py

# Manuscript-scale table validation — ~ 60 minutes (n_sims=100)
python3 replication.py --full --tables

# Figures only (skip Monte Carlo)
python3 replication.py --figures
```

The figure command regenerates all manuscript figure files. The Norwegian
bandwidth-comparison figure is the slowest part because it refits the full
Norway cross-sectional model under Silverman, scalar LOO-CV, and product
LOO-CV bandwidths before writing `norway_loocv_comparison.csv` as an output.

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
``Manuscript Table 2 ratio layout'' and ``Manuscript Table 3 ratio layout''
blocks from the fresh run. In full mode these blocks are the manuscript-scale
validation ratios relative to the published reference values. The archived CSV
files are printed only as comparison copies.

---

## What the Script Reproduces

### Manuscript Tables 2 & 3 — Monte Carlo Validation (Section 5)

The script recomputes the Monte Carlo cells reported in the manuscript. The
Monte Carlo exercise is a focused implementation validation against selected
published Table F.1 entries, not a full replication of the entire Simar and
Wilson Monte Carlo appendix.

By default, `python3 replication.py` runs a quick executable check with
`n_sims=20` and writes `*_quick.csv` files. These quick-run values are
stochastic checks of the code path and are not expected to match the manuscript
tables cell by cell. For manuscript-scale validation, run
`python3 replication.py --full --tables`; this recomputes the reported Monte
Carlo cells with `n_sims=100` and writes `*_full.csv` files. The original
authors' computational code, optimizer settings, tolerances, starting values,
and exact random-number streams were not publicly available, so the validation
is based on the published mathematical specification and reference table
values rather than a bitwise replication of their computational
implementation.

- **Manuscript Table 2**: scalar LOO-CV bandwidth, (p,q) in
  {(1,1),(1,2),(2,2),(2,3),(3,3)}, n in {100,200,400}, rho = 0.
- **Manuscript Table 3**: product LOO-CV bandwidth for (p,q) in
  {(1,1),(2,2)} and rho = 0 for the compact manuscript comparison.

### Section 6 — Simulation-Based Illustration Code Outputs

Reproduces all code-output blocks in Section 6 of the manuscript.
The Section 6 simulation sample uses `np.random.seed(42)`. The pairs
bootstrap and wild bootstrap use the explicit seed `seed=2023`, so the
reported bootstrap output is reproducible for the submitted package and
dependency versions.

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

### Manuscript Tables 4-5 — Norwegian Agricultural Panel (Section 7)

Uses `norway_for_python.csv` (Kumbhakar, Wang & Horncastle 2015).
Reproduces the Norwegian panel summary statistics reported in manuscript
Table 4, the cross-sectional efficiency statistics reported in Section 7,
and the yearly transient/persistent efficiency means reported in manuscript
Table 5.

Expected output (key figures):

| Output | Expected value |
|---|---|
| Table 4 sample size | n = 2,729 farm-year observations, 1998-2006 |
| Table 4 variable statistics | Mean, Std, Min, Q1, Median, Max for x1-x6 and y1-y4 |
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
- `fig_norway_comparison.pdf/png` from fresh fits to
  `norway_for_python.csv`; the figure run writes
  `norway_loocv_comparison.csv` as an output.

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
