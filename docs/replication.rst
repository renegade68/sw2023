Replication
===========

All numerical results in Lee (2026) can be checked using this repository or
the accompanying journal replication archive. The repository contains the
replication script, notebook, requirements file, data, and archived Monte Carlo
CSV files for comparison.

Setup
-----

Clone or download the repository, create a virtual environment, and install
the package and dependencies:

.. code-block:: bash

   cd github_sw2023
   python -m venv .venv
   source .venv/bin/activate
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   pip install .
   python -c "import sw2023; print(sw2023.__version__)"

In the journal submission archive, ``requirements.txt`` installs the submitted
source tarball. In the GitHub repository, the package is installed from the
checked out source tree with ``pip install .``.

Run
---

.. code-block:: bash

   # Tables only; quick Monte Carlo execution check (n_sims=20)
   python replication.py --tables

   # Tables and figures
   python replication.py

   # Manuscript-scale Monte Carlo table validation (n_sims=100)
   python replication.py --full --tables

Reproducibility Notes
---------------------

- The Monte Carlo exercise is a focused implementation validation against
  selected Simar and Wilson Table F.1 entries, not a full replication of the
  entire Monte Carlo appendix. The original authors' computational code,
  optimizer settings, tolerances, starting values, and exact random-number
  streams were not publicly available.
- All bootstrap procedures accept a ``seed`` argument.  The replication
  script fixes ``seed=2023`` for all stochastic results.
- The wild bootstrap p-value (Section 6.4) uses ``B=999`` draws; the
  test statistic ``T`` is deterministic and invariant to the seed.
- The full table command recomputes the manuscript Monte Carlo validation
  cells with ``n_sims=100`` and writes ``*_full.csv`` files.
- Archived Monte Carlo CSV files are provided as comparison copies:
  ``mc_imse_results.csv`` and ``mc_imse_extra.csv`` (scalar LOO-CV), and
  ``mc_imse_product.csv`` (product LOO-CV).
- The default fresh Monte Carlo run is a shorter executable check with
  ``n_sims=20``. Its stochastic outputs are written to separate
  ``*_quick.csv`` files and are not expected to match the manuscript tables
  cell by cell.
- Figure generation uses ``viz_rotation_3d.py`` for the direction-vector
  rotation figure and fresh fits to ``norway_for_python.csv`` for the
  Norwegian bandwidth-comparison figure; the figure run writes
  ``norway_loocv_comparison.csv`` as an output.
- The Section 6 simulation reproduces the reported bootstrap test statistic
  ``T=0.0397``, p-value ``0.8819``, and bootstrap ``T`` range
  ``[0.0144, 0.2358]`` with ``seed=2023``.
