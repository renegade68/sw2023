Installation
============

Requirements
------------

- Python >= 3.8
- numpy >= 1.21
- scipy >= 1.7
- pandas >= 1.3

Optional (for plotting):

- matplotlib >= 3.4

From PyPI
---------

.. code-block:: bash

   pip install sw2023          # core (numpy, scipy, pandas)
   pip install sw2023[viz]     # + matplotlib for plots
   pip install sw2023[dev]     # + jupyter, pytest

From Source
-----------

.. code-block:: bash

   git clone https://github.com/renegade68/sw2023.git
   cd sw2023
   pip install -e .[viz,dev]

Stata Integration
-----------------

The ``sw2023`` Stata module (``sw2023.ado``) is included in the package
under ``sw2023/stata/``.  Copy the ``.ado`` and ``.sthlp`` files to your
Stata ado directory, or call them directly:

.. code-block:: stata

   * Add sw2023/stata to Stata's ado path, then:
   python set exec "/usr/local/bin/python3"
   python: import sys; sys.path.insert(0, "/path/to/sw2023/parent")

   local sw_args "x1 x2 x3 | y1 y2 | method=HMS"
   python script "sw2023/stata/sw2023_stata.py"
