/*============================================================
  SW2023 Multiple-Output Stochastic Frontier Analysis
  Stata 16.1+ Example (simulation-based)

  Requirements:
    - Stata 16.1 or later
    - Python 3.8+ configured via: python set exec "/path/to/python3"
    - sw2023 package installed: pip install sw2023

  First-time Python path setup (run once):
    python set exec "/usr/local/bin/python3"   // macOS/Linux
    python set exec "C:\Python310\python.exe"  // Windows
    python query                               // verify
============================================================*/

* ── Verify Python setup ──────────────────────────────────────
python query

* ── Generate simulation data (Marsaglia sphere, p=q=2, n=200) ──
clear
set obs 200
set seed 42

* Draw sphere surface points: abs of normalized Gaussian
drawnorm z1 z2 z3 z4
local norm "sqrt(z1^2 + z2^2 + z3^2 + z4^2)"
generate x1 = abs(z1) / `norm'
generate x2 = abs(z2) / `norm'
generate y1 = abs(z3) / `norm'
generate y2 = abs(z4) / `norm'

* Add inefficiency noise
drawnorm eta_raw eps_raw
generate eta  = abs(eta_raw) * 0.5
generate eps  = eps_raw * 0.5
generate mult = max(0.5, min(1.5, 1 - eta*0.3 + eps*0.1))
replace  y1   = y1 * mult
replace  y2   = y2 * mult
drop z1-z4 eta_raw eps_raw eta eps mult

label variable x1 "Input 1"
label variable x2 "Input 2"
label variable y1 "Output 1"
label variable y2 "Output 2"

describe
summarize


/*==============================================================
  Model 1: Cross-sectional SW2023 (2-component)

  Argument format: "inputs | outputs | options"
  Options: method=HMS|SVKZ  direction=mean|median
           log_transform=1|0  standardize=1|0
==============================================================*/

local sw_args "x1 x2 | y1 y2 | method=HMS"

python script "sw2023/stata/sw2023_stata.py"

* Inspect results
summarize sw_efficiency, detail
label variable sw_efficiency "SW(2023) efficiency"

histogram sw_efficiency, ///
    title("SW(2023) Efficiency Distribution") ///
    xtitle("Efficiency score") ///
    normal name(hist_cs, replace)


/*==============================================================
  Model 2: 4-Component Panel SW2023

  Requires panel identifiers: firm=<id variable> time=<time variable>
  (Add firmid and year variables for panel data)
==============================================================*/

* Illustrative panel setup: treat each obs as a unique firm
generate firmid = _n
generate year   = 2020

xtset firmid year

local sw_args ///
    "x1 x2 | y1 y2 | method=HMS firm=firmid time=year"

python script "sw2023/stata/sw2023_stata.py", args("panel")

* Inspect results
summarize swp_efficiency swp_te swp_pe, detail

label variable swp_efficiency "SW(2023) overall efficiency"
label variable swp_te         "SW(2023) transient efficiency"
label variable swp_pe         "SW(2023) persistent efficiency"

scatter swp_te swp_pe, ///
    msymbol(point) mcolor(blue%30) ///
    title("Transient vs Persistent Efficiency") ///
    xtitle("Persistent efficiency (PE)") ///
    ytitle("Transient efficiency (TE)") ///
    yline(1) xline(1) ///
    name(scatter_te_pe, replace)


/*==============================================================
  Compare cross-sectional vs panel efficiency
==============================================================*/

pwcorr sw_efficiency swp_efficiency swp_te swp_pe, star(0.05)
