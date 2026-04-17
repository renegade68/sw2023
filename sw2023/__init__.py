"""
sw2023: Simar & Wilson (2023) nonparametric multiple-output stochastic frontier analysis

References:
  Simar, L., Wilson, P.W. (2023). Nonparametric, Econometric Analysis of
  Production, Cost, Revenue and Profit Functions.
  Econometric Reviews, 42(4), 1391-1403.

Main classes:
  SW2023Model   - cross-sectional model (2-component: v, η)
  PanelSW2023   - panel extension (4-component: v_it, u_it, α_i, μ_i)

Helper functions:
  bootstrap_sw, bootstrap_panel  - bootstrap confidence intervals
  bandwidth_loocv                - LOO-CV bandwidth selection (SW 2023 paper method)
  bandwidth_silverman            - Silverman rule bandwidth
"""

from .core.model           import SW2023Model
from .panel.four_component import PanelSW2023
from .core.bootstrap       import bootstrap_sw, bootstrap_panel, test_r3_significance
from .core.bandwidth       import bandwidth_loocv, bandwidth_silverman

__version__ = '0.3.1'
__author__  = 'Choonjoo Lee'
__all__     = [
    'SW2023Model',
    'PanelSW2023',
    'bootstrap_sw',
    'bootstrap_panel',
    'test_r3_significance',
    'bandwidth_loocv',
    'bandwidth_silverman',
]
