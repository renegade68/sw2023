Quick Start
===========

Cross-Sectional Model
---------------------

Fit a 2-component SW(2023) model and inspect the results:

.. code-block:: python

   import numpy as np
   from sw2023 import SW2023Model

   # Simulated data: 200 observations, 2 inputs, 2 outputs
   rng = np.random.default_rng(42)
   X = np.abs(rng.standard_normal((200, 2)))
   Y = np.abs(rng.standard_normal((200, 2)))

   m = SW2023Model(X, Y, method='HMS', bandwidth_method='silverman')
   m.fit()

   print(m)                          # SW2023Model(n=200, ..., mean_eff=0.xxxx)
   print(m.efficiency_.mean())       # mean efficiency score
   print(m.sigma_eta_.mean())        # mean inefficiency std dev

Asymptotic Confidence Intervals
--------------------------------

.. code-block:: python

   ci = m.confint_asymptotic(alpha=0.05)
   ci.summary()                      # formatted table
   print(ci.se_phi)                  # (n,) standard errors for phi_hat

Bootstrap Confidence Intervals
--------------------------------

.. code-block:: python

   # Via model method (recommended)
   boot = m.bootstrap(B=199, seed=42)
   boot.summary()

   print(boot.eff_mean_ci)           # [lower, upper] for mean efficiency
   print(boot.phi_hat_ci)            # (n, 2) frontier CI

   # Or via the standalone function
   from sw2023 import bootstrap_sw
   boot = bootstrap_sw(X, Y, B=199, seed=42)

Wild Bootstrap Significance Test
----------------------------------

Test whether inefficiency varies spatially (H0: uniform, H1: heterogeneous):

.. code-block:: python

   from sw2023 import test_r3_significance

   res = test_r3_significance(X, Y, B=999, seed=42)
   res.summary()
   print(res.p_value)               # large p → H0 not rejected

Diagnostic Plots
----------------

.. code-block:: python

   import matplotlib.pyplot as plt

   m.plot_efficiency()              # efficiency distribution
   m.plot_frontier(dim=0)           # U vs Z[0] with frontier
   fig = m.plot_diagnostics()       # 2×2 diagnostic panel
   plt.show()

Panel Model (4-Component)
--------------------------

.. code-block:: python

   from sw2023 import PanelSW2023

   m_panel = PanelSW2023(X, Y, firm_id, time_id, method='HMS')
   m_panel.fit()

   print(m_panel.eff_transient_.mean())   # transient efficiency
   print(m_panel.eff_persistent_.mean())  # persistent efficiency

   # Panel diagnostic plot
   from sw2023 import plot_panel_trend
   plot_panel_trend(m_panel, time_id)
   plt.show()
