/*============================================================
  sw2023.ado / sw2023test.ado — Native Stata Example
  (Pure Stata/Mata, no Python required)

  Requirements:
    - Stata 16.1 or later
    - sw2023.ado and sw2023test.ado on your ado-path, or:
        adopath + "path/to/sw2023/stata"

  Demonstrates:
    1. Point estimation (Silverman & LOO-CV bandwidth)
    2. Pairs bootstrap confidence intervals (SW 2023)
    3. Wild bootstrap significance test (PSVKZ 2024)
============================================================*/

* ── Add ado-path if sw2023.ado is not on system path ────────
* adopath + "sw2023/stata"    // uncomment if needed

* ── Generate simulation data (Marsaglia sphere, p=q=2, n=200) ──
clear
set obs 200
set seed 42

drawnorm z1 z2 z3 z4
local norm "sqrt(z1^2 + z2^2 + z3^2 + z4^2)"
generate x1 = abs(z1) / `norm'
generate x2 = abs(z2) / `norm'
generate y1 = abs(z3) / `norm'
generate y2 = abs(z4) / `norm'

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
  1. Point estimation: Silverman bandwidth
==============================================================*/
sw2023 y1 y2, inputs(x1 x2) direction(mean) method(hms) ///
              bandwidth(silverman) generate(sw_)

summarize sw_efficiency, detail
histogram sw_efficiency, ///
    title("SW(2023) Efficiency Distribution") xtitle("Efficiency") ///
    normal name(hist_sw, replace)


/*==============================================================
  2. Pairs bootstrap 95% CI  (SW 2023 pairs resampling)

  bootstrap(199) : B=199 replications
  level(95)      : 95% CI  (default)

  New variables: sw_eff_lo, sw_eff_hi, sw_phi_lo, sw_phi_hi,
                 sw_seta_lo, sw_seta_hi
==============================================================*/
sw2023 y1 y2, inputs(x1 x2) direction(mean) method(hms)  ///
              bandwidth(silverman) generate(sw_)           ///
              bootstrap(199) level(95)

summarize sw_eff_lo sw_eff_hi, detail

* Plot efficiency with CI
generate sw_eff_width = sw_eff_hi - sw_eff_lo

scatter sw_eff_width sw_efficiency, ///
    msymbol(point) mcolor(navy%40)   ///
    title("Bootstrap CI Width vs Efficiency Score") ///
    xtitle("Efficiency") ytitle("CI width") ///
    name(sc_ci, replace)


/*==============================================================
  3. Wild bootstrap significance test  (PSVKZ 2024)

  H0: E(eps^3 | Z) = const  (homogeneous inefficiency)
  H1: heterogeneous

  reps(499) : B=499 replications
==============================================================*/
sw2023test y1 y2, inputs(x1 x2) direction(mean) method(hms) ///
                  bandwidth(silverman) reps(299)

* Access results
display "T statistic : " r(statistic)
display "p-value     : " r(p_value)


/*==============================================================
  Compare Silverman vs LOO-CV efficiency scores
==============================================================*/
sw2023 y1 y2, inputs(x1 x2) direction(mean) method(hms) ///
              bandwidth(loocv) generate(swcv_)

pwcorr sw_efficiency swcv_efficiency, star(0.05)

scatter sw_efficiency swcv_efficiency, ///
    msymbol(point) mcolor(maroon%40)    ///
    title("Silverman vs LOO-CV Efficiency") ///
    xtitle("LOO-CV") ytitle("Silverman") ///
    name(sc_bw, replace)
