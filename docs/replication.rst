Replication
===========

All numerical results in Lee (2026) can be reproduced using the
accompanying replication archive.

Setup
-----

Install the package and dependencies:

.. code-block:: bash

   pip install sw2023-0.3.2.tar.gz
   pip install -r requirements.txt

Run
---

.. code-block:: bash

   # Quick verification (< 5 minutes, n_sims=20)
   python replication.py

   # Full replication (~30 minutes, n_sims=100)
   python replication.py --full

Reproducibility Notes
---------------------

- All bootstrap procedures accept a ``seed`` argument.  The replication
  script fixes ``seed=2023`` for all stochastic results.
- The wild bootstrap p-value (Section 6.4) uses ``B=999`` draws; the
  test statistic ``T`` is deterministic and invariant to the seed.
- Pre-computed Monte Carlo results are provided in ``mc_imse_results.csv``
  (scalar LOO-CV) and ``mc_imse_product.csv`` (product LOO-CV).
