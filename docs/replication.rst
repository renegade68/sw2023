Replication
===========

All numerical results in Lee (2026) can be checked using the accompanying
replication archive. The archive contains the replication script, notebook,
requirements file, data, and the pre-computed Monte Carlo CSV files used for
the manuscript tables.

Setup
-----

Place ``sw2023-0.3.2.tar.gz`` and ``sw2023_replication.zip`` in a clean
working directory, unzip the replication archive, create a virtual
environment, and install the package and dependencies from the revised
``requirements.txt``:

.. code-block:: bash

   python -m venv .venv
   source .venv/bin/activate
   unzip sw2023_replication.zip
   cp /path/to/sw2023-0.3.2.tar.gz .
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt

The first line of ``requirements.txt`` is ``./sw2023-0.3.2.tar.gz``, so the
package itself is installed into the virtual environment together with
``numpy``, ``scipy``, ``pandas``, and ``matplotlib``.

Run
---

.. code-block:: bash

   # Tables only; quick Monte Carlo execution check (n_sims=20)
   python replication.py --tables

   # Tables and figures
   python replication.py

   # Longer fresh Monte Carlo run (n_sims=100)
   python replication.py --full --tables

Reproducibility Notes
---------------------

- The Monte Carlo exercise is a focused implementation validation against
  selected Simar and Wilson Table F.1 entries, not a full replication of the
  entire Monte Carlo appendix. The original authors' computational code and
  optimizer settings were not publicly available.
- All bootstrap procedures accept a ``seed`` argument.  The replication
  script fixes ``seed=2023`` for all stochastic results.
- The wild bootstrap p-value (Section 6.4) uses ``B=999`` draws; the
  test statistic ``T`` is deterministic and invariant to the seed.
- Pre-computed Monte Carlo results used in the manuscript are provided in
  ``mc_imse_results.csv`` and ``mc_imse_extra.csv`` (scalar LOO-CV), and
  ``mc_imse_product.csv`` (product LOO-CV). These files contain the
  ``n_sims=100`` manuscript values.
- The default fresh Monte Carlo run is a shorter executable check with
  ``n_sims=20``. Its stochastic outputs are written to separate
  ``*_quick.csv`` files and are not expected to match the manuscript tables
  cell by cell.
- Figure generation uses ``viz_rotation_3d.py`` for the direction-vector
  rotation figure and ``norway_loocv_comparison.csv`` for the Norwegian
  bandwidth-comparison figure.
- The Section 6 simulation reproduces the reported bootstrap test statistic
  ``T=0.0397``, p-value ``0.8819``, and bootstrap ``T`` range
  ``[0.0144, 0.2358]`` with ``seed=2023``.
