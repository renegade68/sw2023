Changelog
=========

0.3.2 (2026)
------------

- Added result classes: :class:`ConfintResult`, :class:`BootstrapResult`,
  :class:`SignificanceTestResult` with ``__repr__`` and ``summary()`` methods.
- Added :meth:`SW2023Model.bootstrap` method (wraps :func:`bootstrap_sw`).
- Added diagnostic plot methods: :meth:`SW2023Model.plot_efficiency`,
  :meth:`SW2023Model.plot_frontier`, :meth:`SW2023Model.plot_diagnostics`.
- Added :func:`plot_residuals` and :func:`plot_diagnostics` to visualization module.
- Added human-readable ``__repr__`` output for :class:`SW2023Model` and
  :class:`PanelSW2023`.
- Updated panel bootstrap compatibility with the panel model bandwidth
  argument and result dataframe output.
- Bootstrap functions now accept ``seed`` argument for reproducibility.
- ``test_r3_significance`` default ``B`` increased to 999.
- ``pyproject.toml``: SPDX license expression, ``MANIFEST.in`` added.
- Updated online documentation and replication instructions for the
  self-contained replication archive, revised source tarball,
  ``requirements.txt`` installation workflow, Python 3.13 checks, manuscript
  figure generation, manuscript-scale full Monte Carlo table validation, and
  archived Monte Carlo comparison copies.

0.3.1 (2025)
------------

- Initial public release.
- Cross-sectional model (:class:`SW2023Model`).
- 4-component panel model (:class:`PanelSW2023`).
- LOO-CV and Silverman bandwidth selection.
- Pairs bootstrap CI (:func:`bootstrap_sw`, :func:`bootstrap_panel`).
- Wild bootstrap significance test (:func:`test_r3_significance`).
- Stata 16.1+ integration.
