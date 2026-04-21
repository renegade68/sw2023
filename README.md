# sw2023

**Nonparametric Multiple-Output Stochastic Frontier Analysis in Python**

[![PyPI version](https://img.shields.io/pypi/v/sw2023)](https://pypi.org/project/sw2023/)
[![Documentation](https://readthedocs.org/projects/sw2023/badge/?version=latest)](https://sw2023.readthedocs.io)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)

A Python implementation of the nonparametric stochastic frontier estimator of
Simar & Wilson (2023, *Journal of Business & Economic Statistics*) for production
technologies with multiple inputs and multiple outputs.

> Simar, L., Wilson, P.W. (2023). Nonparametric, Stochastic Frontier Models
> with Multiple Inputs and Outputs. *Journal of Business & Economic Statistics*,
> 41(4), 1391–1403. https://doi.org/10.1080/07350015.2022.2110882

---

## Features

| Feature | Status |
|---|---|
| Multiple inputs & outputs (directional distance function) | ✅ |
| Local Linear Least Squares (LLLS) frontier estimation | ✅ |
| SVKZ and HMS σ_η estimators | ✅ |
| JLMS individual efficiency recovery | ✅ |
| LOO-CV and Silverman bandwidth selection | ✅ |
| Pairs bootstrap confidence intervals | ✅ |
| Asymptotic CI (CLT + delta method) | ✅ |
| Wild bootstrap significance test (Parmeter et al. 2024) | ✅ |
| 4-component panel (transient + persistent inefficiency) | ✅ |
| Stata 16.1+ integration (pure Mata, no Python dependency) | ✅ |

---

## Installation

```bash
pip install sw2023                  # core (numpy, scipy, pandas)
pip install sw2023[viz]             # + matplotlib
pip install sw2023[dev]             # + jupyter, pytest
```

**Requirements:** Python >= 3.8, numpy >= 1.21, scipy >= 1.7, pandas >= 1.3

---

## Quick Start

### Cross-sectional model

```python
import numpy as np
from sw2023 import SW2023Model

X = np.random.lognormal(0, 0.5, size=(200, 2))   # 2 inputs
Y = np.random.lognormal(0, 0.5, size=(200, 3))   # 3 outputs

m = SW2023Model(X, Y, method='HMS', bandwidth_method='loocv')
m.fit()
m.summary()

print(m.efficiency_.mean())         # mean efficiency
print(m.sigma_eta_.mean())          # mean sigma_eta
```

### Bootstrap confidence intervals

```python
from sw2023 import bootstrap_sw

result = bootstrap_sw(X, Y, B=199, alpha=0.05)
print(result['eff_mean_ci'])        # [lower, upper]
print(result['phi_hat_ci'])         # (n, 2) frontier CI
```

### Asymptotic CI (fast)

```python
m = SW2023Model(X, Y, method='HMS')
m.fit()
ci = m.confint_asymptotic(alpha=0.05)
print(ci['phi_hat_ci'])             # (n, 2)
print(ci['se_phi'])                 # (n,) standard errors
```

### Significance test for inefficiency heterogeneity

```python
from sw2023 import test_r3_significance

res = test_r3_significance(X, Y, B=999, seed=42)
print(res['p_value'])               # H0: E(eps^3 | Z) = const
```

### Panel model (4-component)

```python
from sw2023 import PanelSW2023

m = PanelSW2023(X, Y, firm_id, time_id, method='HMS')
m.fit()
print(m.eff_transient_.mean())      # transient efficiency
print(m.eff_persistent_.mean())     # persistent efficiency
```

---

## Stata Integration (16.1+)

```stata
* Cross-sectional
local sw_args "x1 x2 | y1 y2 | method=HMS"
python script "sw2023_stata.py"

* Panel (4-component)
local sw_args "x1 x2 | y1 y2 | method=HMS firm=firmid time=year"
python script "sw2023_stata.py", args("panel")

* Wild bootstrap significance test (pure Mata, no Python needed)
sw2023test y1 y2, inputs(x1 x2) reps(299)
```

---

## Replication

All numerical results in the accompanying manuscript can be reproduced with:

```bash
# Quick verification (< 5 minutes)
python replication.py

# Exact replication of Table 1 (n_sims=100, ~30 minutes)
python replication.py --full
```

Pre-computed Monte Carlo results (100 replications per cell) are provided
in `mc_imse_results.csv`. IMSE ratios relative to Simar & Wilson (2023)
Table F.1 range from 0.94 to 1.29 with median 1.04.

**Note on bootstrap reproducibility.** All bootstrap procedures
(`bootstrap_sw`, `bootstrap_panel`, `test_r3_significance`) accept a `seed`
argument for exact reproducibility. The replication script fixes `seed=2023`
and uses `B=999` for the wild bootstrap significance test; omitting `seed`
produces stochastic results that will differ across runs. The test statistic
`T` is deterministic (computed from the original sample) and is unaffected by
the choice of seed; only the bootstrap p-value varies.

---

## Methodology

The SW(2023) estimator handles **multiple outputs** without imposing a
parametric form on the frontier. Key steps:

1. **Direction vector** `d ∈ R^(p+q)`: defines the efficiency direction
2. **Rotation** `(X, Y) → (Z, U)`: projects onto frontier coordinates
3. **LLLS**: kernel regression of `U` on `Z` → conditional moments r̂₁, r̂₂, r̂₃
4. **σ_η estimation**: `σ̂_η = (-r̂₃/a₃⁺)^(1/3)` (SVKZ) or HMS for wrong-skewness
5. **JLMS**: `E[η | ξ̂]` → individual efficiency score `exp(-η̂)`

The **4-component panel extension** decomposes:
```
U_it = φ(Z_it) + ||d||·v_it - ||d||·u_it + ||d||·α_i - ||d||·μ_i
```
- `v_it`: transient noise
- `u_it ~ N⁺(0, σ_u²)`: transient inefficiency
- `α_i ~ N(0, σ_α²)`: individual heterogeneity
- `μ_i ~ N⁺(0, σ_μ²)`: persistent inefficiency

---

## Citation

If you use this package, please cite:

```bibtex
@misc{Lee2025sw2023,
  author = {Lee, Choonjoo},
  title  = {{sw2023}: Nonparametric Multiple-Output Stochastic Frontier
            Analysis in {Python}},
  year   = {2026},
  note   = {Manuscript submitted for publication}
}
```

---

## References

- Simar, L. & Wilson, P.W. (2023). Nonparametric, Stochastic Frontier Models
  with Multiple Inputs and Outputs. *Journal of Business & Economic Statistics*,
  41(4), 1391–1403.
- Parmeter, C.F., Simar, L., Van Keilegom, I. & Zelenyuk, V. (2024).
  Inference in the nonparametric stochastic frontier model.
  *Econometric Reviews*, 43(7), 518–539.
  https://doi.org/10.1080/07474938.2024.2339193
- Hafner, C.M., Manner, H. & Simar, L. (2018). The "wrong skewness" problem
  in stochastic frontier models: A new approach. *Econometric Reviews*, 37(4), 380–400.
  https://doi.org/10.1080/07474938.2016.1140284
- Jondrow, J., Lovell, C.A.K., Materov, I.S. & Schmidt, P. (1982).
  On the estimation of technical inefficiency in the stochastic frontier
  production function model. *Journal of Econometrics*, 19(2–3), 233–238.

---

## License

GNU General Public License v3 or later (GPL-3.0-or-later).
See [LICENSE](LICENSE) for details.
