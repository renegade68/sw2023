Result Classes
==============

These classes wrap the numerical arrays returned by estimation methods
and provide human-readable output via ``__repr__`` and ``summary()``.

.. currentmodule:: sw2023

ConfintResult
-------------

Returned by :meth:`SW2023Model.confint_asymptotic`.

.. autoclass:: ConfintResult
   :members:
   :special-members: __repr__

BootstrapResult
---------------

Returned by :meth:`SW2023Model.bootstrap` and :func:`bootstrap_sw`.

.. autoclass:: BootstrapResult
   :members:
   :special-members: __repr__

SignificanceTestResult
----------------------

Returned by :func:`test_r3_significance`.

.. autoclass:: SignificanceTestResult
   :members:
   :special-members: __repr__
