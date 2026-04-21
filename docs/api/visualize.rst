Visualization
=============

All plot functions return ``(fig, ax)`` or ``fig`` so that the caller
can further customize the plot before calling ``plt.show()``.
Diagnostic plot methods are also available directly on fitted model
objects: :meth:`SW2023Model.plot_efficiency`,
:meth:`SW2023Model.plot_frontier`, :meth:`SW2023Model.plot_diagnostics`.

.. currentmodule:: sw2023

Cross-Sectional
---------------

.. autofunction:: plot_efficiency_dist

.. autofunction:: plot_efficiency_rank

.. autofunction:: plot_frontier_1d

.. autofunction:: plot_residuals

.. autofunction:: plot_diagnostics

Panel
-----

.. autofunction:: plot_panel_trend

.. autofunction:: plot_decomposition
