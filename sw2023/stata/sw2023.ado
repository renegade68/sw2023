*! sw2023 v0.2.0 — Nonparametric Multiple-Output Stochastic Frontier Analysis
*! Implements Simar & Wilson (2023, JBES) in Stata Mata
*!
*! Point estimation + SW pairs bootstrap CI + PSVKZ wild bootstrap test.
*!
*! Syntax — point estimation:
*!   sw2023 yvarlist, inputs(xvarlist)
*!          [direction(mean|median) method(hms|svkz)
*!           bandwidth(silverman|loocv) generate(prefix) nolog]
*!
*! With pairs bootstrap (Simar & Wilson 2023):
*!   sw2023 yvarlist, inputs(xvarlist) bootstrap(#) [level(#)]
*!
*! Wild bootstrap significance test (PSVKZ 2024):
*!   sw2023test yvarlist, inputs(xvarlist) [reps(#) level(#)]
*!          [direction(mean|median) method(hms|svkz)
*!           bandwidth(silverman) generate(prefix) nolog]

/* ====================================================================
   Main command: sw2023
   ==================================================================== */

program define sw2023, eclass
    version 16.1

    syntax varlist(min=1 numeric) ,         ///
        INputs(varlist numeric min=1)        ///
        [ DIRection(string)                  ///
          METHod(string)                     ///
          BANDwidth(string)                  ///
          GENerate(string)                   ///
          BOOTstrap(integer 0)              ///
          LEVel(real 95)                     ///
          nolog                              ///
          FIRMid(varname)                    ///
          TIMEid(varname) ]

    /* ── defaults ───────────────────────────────────────────── */
    if "`direction'" == ""  local direction "mean"
    if "`method'"    == ""  local method    "hms"
    if "`bandwidth'" == ""  local bandwidth "silverman"
    if "`generate'"  == ""  local generate  "sw_"

    local direction = lower("`direction'")
    local method    = lower("`method'")
    local bandwidth = lower("`bandwidth'")

    /* ── validate ───────────────────────────────────────────── */
    if !inlist("`direction'", "mean", "median") {
        di as error "direction() must be mean or median"
        exit 198
    }
    if !inlist("`method'", "hms", "svkz") {
        di as error "method() must be hms or svkz"
        exit 198
    }
    if !inlist("`bandwidth'", "silverman", "loocv") {
        di as error "bandwidth() must be silverman or loocv"
        exit 198
    }
    if `bootstrap' < 0 {
        di as error "bootstrap() must be a non-negative integer"
        exit 198
    }
    if `level' <= 0 | `level' >= 100 {
        di as error "level() must be between 0 and 100"
        exit 198
    }
    if `bootstrap' > 0 & "`bandwidth'" == "loocv" {
        di as text "Note: bootstrap uses Silverman bandwidth (loocv too slow for resampling)"
        local boot_bw "silverman"
    }
    else {
        local boot_bw "`bandwidth'"
    }

    /* ── sample ──────────────────────────────────────────────── */
    marksample touse
    markout `touse' `varlist' `inputs'
    quietly count if `touse'
    local N = r(N)
    if `N' < 10 {
        di as error "Insufficient observations (need >= 10)"
        exit 2001
    }

    local yvars `varlist'
    local xvars `inputs'
    local q : word count `yvars'
    local p : word count `xvars'

    /* ── header ─────────────────────────────────────────────── */
    if "`log'" == "" {
        di as text ""
        di as text "[sw2023] Nonparametric Multiple-Output SFA"
        di as text "  Outputs  : `yvars'"
        di as text "  Inputs   : `xvars'"
        di as text "  N        : `N'"
        di as text "  p/q      : `p' / `q'"
        di as text "  direction: `direction'"
        di as text "  method   : `method'"
        di as text "  bandwidth: `bandwidth'"
        if `bootstrap' > 0 {
            di as text "  bootstrap: B=`bootstrap', `= int(`level')'% CI"
        }
        di as text ""
    }

    /* ── load data ───────────────────────────────────────────── */
    quietly {
        tempname Xm Ym
        mkmat `xvars' if `touse', matrix(`Xm')
        mkmat `yvars' if `touse', matrix(`Ym')
    }

    /* ── output variable names ───────────────────────────────── */
    local gen_eff  "`generate'efficiency"
    local gen_phi  "`generate'phi_hat"
    local gen_seta "`generate'sigma_eta"
    local gen_seps "`generate'sigma_eps"
    local gen_r1   "`generate'r1"
    local gen_r2   "`generate'r2"
    local gen_r3   "`generate'r3"

    foreach v in `gen_eff' `gen_phi' `gen_seta' `gen_seps' ///
                 `gen_r1'  `gen_r2'  `gen_r3' {
        capture drop `v'
        quietly generate double `v' = .
    }

    /* ── bootstrap CI output variables ────────────────────────── */
    if `bootstrap' > 0 {
        foreach sfx in phi_lo phi_hi eff_lo eff_hi seta_lo seta_hi {
            local gen_`sfx' "`generate'`sfx'"
            capture drop `gen_`sfx''
            quietly generate double `gen_`sfx'' = .
        }
    }

    /* ── point estimation ────────────────────────────────────── */
    mata: _sw2023_run("`Xm'", "`Ym'", "`touse'",              ///
                      "`direction'", "`method'", "`bandwidth'", ///
                      "`log'",                                   ///
                      "`gen_eff'", "`gen_phi'",                  ///
                      "`gen_seta'", "`gen_seps'",                ///
                      "`gen_r1'", "`gen_r2'", "`gen_r3'")

    /* ── bootstrap CI ────────────────────────────────────────── */
    if `bootstrap' > 0 {
        if "`log'" == "" {
            di as text "[sw2023] Pairs bootstrap (B=`bootstrap')..."
        }
        mata: _sw2023_boot_run("`Xm'", "`Ym'", "`touse'",          ///
                                "`direction'", "`method'",           ///
                                "`boot_bw'", "`log'",                ///
                                `bootstrap', (100 - `level') / 100,  ///
                                "`gen_phi_lo'", "`gen_phi_hi'",       ///
                                "`gen_eff_lo'", "`gen_eff_hi'",       ///
                                "`gen_seta_lo'", "`gen_seta_hi'")
    }

    /* ── label ───────────────────────────────────────────────── */
    quietly {
        label variable `gen_eff'  "SW(2023) efficiency"
        label variable `gen_phi'  "SW(2023) frontier phi_hat(Z)"
        label variable `gen_seta' "SW(2023) sigma_eta(Z)"
        label variable `gen_seps' "SW(2023) sigma_eps(Z)"
        label variable `gen_r1'   "SW(2023) r1_hat(Z) = E[U|Z]"
        label variable `gen_r2'   "SW(2023) r2_hat(Z) = E[eps^2|Z]"
        label variable `gen_r3'   "SW(2023) r3_hat(Z) = E[eps^3|Z]"
        if `bootstrap' > 0 {
            label variable `gen_phi_lo'  "SW(2023) phi_hat CI lower"
            label variable `gen_phi_hi'  "SW(2023) phi_hat CI upper"
            label variable `gen_eff_lo'  "SW(2023) efficiency CI lower"
            label variable `gen_eff_hi'  "SW(2023) efficiency CI upper"
            label variable `gen_seta_lo' "SW(2023) sigma_eta CI lower"
            label variable `gen_seta_hi' "SW(2023) sigma_eta CI upper"
        }
    }

    /* ── summary ─────────────────────────────────────────────── */
    if "`log'" == "" {
        quietly summarize `gen_eff' if `touse'
        di as text "  Mean efficiency : " as result %8.4f r(mean)
        di as text "  Std  efficiency : " as result %8.4f r(sd)
        quietly summarize `gen_seta' if `touse'
        di as text "  Mean sigma_eta  : " as result %8.4f r(mean)
        if `bootstrap' > 0 {
            quietly summarize `gen_eff_lo' if `touse'
            local lo_mean = r(mean)
            quietly summarize `gen_eff_hi' if `touse'
            local hi_mean = r(mean)
            di as text "  Mean eff CI     : [" ///
               as result %6.4f `lo_mean' as text ", " ///
               as result %6.4f `hi_mean' as text "]  (`= int(`level')'%)"
        }
        di as text ""
        di as text "[sw2023] Done."
    }

end


/* ====================================================================
   Wild bootstrap significance test: sw2023test
   H0: E(eps^3 | Z) = const  (homogeneous inefficiency)
   H1: E(eps^3 | Z) != const (heterogeneous inefficiency)
   Reference: PSVKZ (2024) Section 3.1
   ==================================================================== */

program define sw2023test, rclass
    version 16.1

    syntax varlist(min=1 numeric) ,         ///
        INputs(varlist numeric min=1)        ///
        [ DIRection(string)                  ///
          METHod(string)                     ///
          BANDwidth(string)                  ///
          GENerate(string)                   ///
          REPs(integer 499)                  ///
          nolog ]

    if "`direction'" == ""  local direction "mean"
    if "`method'"    == ""  local method    "hms"
    if "`bandwidth'" == ""  local bandwidth "silverman"
    if "`generate'"  == ""  local generate  "sw_"

    local direction = lower("`direction'")
    local method    = lower("`method'")
    local bandwidth = lower("`bandwidth'")

    if `reps' < 99 {
        di as error "reps() should be at least 99"
        exit 198
    }

    marksample touse
    markout `touse' `varlist' `inputs'
    quietly count if `touse'
    local N = r(N)

    if "`log'" == "" {
        di as text ""
        di as text "[sw2023test] Wild Bootstrap Significance Test"
        di as text "  H0: E(eps^3 | Z) = const  (homogeneous inefficiency)"
        di as text "  H1: E(eps^3 | Z) != const (heterogeneous)"
        di as text "  N: `N'   Reps: `reps'"
        di as text ""
    }

    quietly {
        tempname Xm Ym
        mkmat `inputs' if `touse', matrix(`Xm')
        mkmat `varlist' if `touse', matrix(`Ym')
    }

    tempname T_obs p_val
    mata: _sw2023_wild_test("`Xm'", "`Ym'",                   ///
                             "`direction'", "`method'",         ///
                             "`bandwidth'", "`log'",            ///
                             `reps',                            ///
                             "`T_obs'", "`p_val'")

    local T_val = `T_obs'
    local p_value = `p_val'

    if "`log'" == "" {
        di as text "  Test statistic T : " as result %10.6f `T_val'
        di as text "  Bootstrap p-value: " as result %8.4f `p_value'
        if `p_value' < 0.05 {
            di as text "  => Reject H0 at 5%: heterogeneous inefficiency"
        }
        else {
            di as text "  => Fail to reject H0: homogeneous inefficiency"
        }
        di as text ""
        di as text "[sw2023test] Done."
    }

    return scalar statistic = `T_val'
    return scalar p_value   = `p_value'
    return scalar reps      = `reps'

end


/* ====================================================================
   Mata implementation
   ==================================================================== */

mata:

/* ── struct for point-estimation results ────────────────────────── */
struct _sw2023_fit {
    real colvector d
    real matrix    R
    real matrix    Z
    real colvector U
    real rowvector h
    real colvector r1_hat       /* E[U|Z]     */
    real colvector eps          /* U - r1_hat */
    real colvector r2_hat       /* E[eps^2|Z] */
    real colvector r3_hat       /* E[eps^3|Z] */
    real colvector sigma_eta
    real colvector sigma_eps
    real colvector phi_hat      /* frontier   */
    real colvector xi           /* U - phi_hat */
    real colvector eff
}


/* ── core estimation: fills a _sw2023_fit struct ────────────────── */
struct _sw2023_fit scalar _sw2023_fit_model(
        real matrix X, real matrix Y,
        string scalar direction,
        string scalar method,
        string scalar bwmethod)
{
    struct _sw2023_fit scalar m
    real scalar n, p, q
    real matrix W, WR

    n = rows(X); p = cols(X); q = cols(Y)
    W  = X, Y

    m.d = _sw2023_direction(X, Y, direction)
    m.R = _sw2023_rotation(m.d)
    WR  = W * m.R'
    m.Z = WR[., 1..(p+q-1)]
    m.U = WR[., p+q]

    if (bwmethod == "loocv") {
        m.h = _sw2023_loocv(m.Z, m.U)
    } else {
        m.h = _sw2023_silverman(m.Z, n)
    }

    m.r1_hat  = _sw2023_llls(m.Z, m.U, m.h)
    m.eps     = m.U :- m.r1_hat
    m.r2_hat  = _sw2023_llls(m.Z, m.eps:^2, m.h)
    m.r3_hat  = _sw2023_llls(m.Z, m.eps:^3, m.h)

    if (method == "hms") {
        m.sigma_eta = _sw2023_sigma_eta_hms(m.r3_hat)
    } else {
        m.sigma_eta = _sw2023_sigma_eta_svkz(m.r3_hat)
    }
    m.sigma_eps = _sw2023_sigma_eps(m.r2_hat, m.sigma_eta)
    m.phi_hat   = m.r1_hat :+ sqrt(2 / 3.14159265358979) :* m.sigma_eta
    m.xi        = m.U :- m.phi_hat
    m.eff       = _sw2023_jlms(m.xi, m.sigma_eta, m.sigma_eps)

    return(m)
}


/* ── entry point: point estimation ─────────────────────────────── */
void _sw2023_run(
        string scalar Xname, string scalar Yname,
        string scalar touse,
        string scalar direction, string scalar method,
        string scalar bwmethod, string scalar nolog,
        string scalar v_eff,  string scalar v_phi,
        string scalar v_seta, string scalar v_seps,
        string scalar v_r1,   string scalar v_r2, string scalar v_r3)
{
    struct _sw2023_fit scalar m
    real matrix X, Y

    X = st_matrix(Xname)
    Y = st_matrix(Yname)

    m = _sw2023_fit_model(X, Y, direction, method, bwmethod)

    _sw2023_put(v_eff,  m.eff,       touse)
    _sw2023_put(v_phi,  m.phi_hat,   touse)
    _sw2023_put(v_seta, m.sigma_eta, touse)
    _sw2023_put(v_seps, m.sigma_eps, touse)
    _sw2023_put(v_r1,   m.r1_hat,    touse)
    _sw2023_put(v_r2,   m.r2_hat,    touse)
    _sw2023_put(v_r3,   m.r3_hat,    touse)
}


/* ── entry point: pairs bootstrap ──────────────────────────────── */
void _sw2023_boot_run(
        string scalar Xname, string scalar Yname,
        string scalar touse,
        string scalar direction, string scalar method,
        string scalar bwmethod, string scalar nolog,
        real scalar B, real scalar alpha,
        string scalar v_phi_lo, string scalar v_phi_hi,
        string scalar v_eff_lo, string scalar v_eff_hi,
        string scalar v_seta_lo, string scalar v_seta_hi)
{
    struct _sw2023_fit scalar m0
    real matrix X, Y, boot_phi, boot_eff, boot_seta
    real matrix phi_ci, eff_ci, seta_ci
    real colvector phi_b, eff_b, seta_b
    real scalar n, b, n_fail

    X = st_matrix(Xname)
    Y = st_matrix(Yname)
    n = rows(X)

    /* original model */
    m0 = _sw2023_fit_model(X, Y, direction, method, "silverman")

    boot_phi  = J(B, n, .)
    boot_eff  = J(B, n, .)
    boot_seta = J(B, n, .)
    n_fail    = 0

    for (b = 1; b <= B; b++) {
        if (nolog == "" & mod(b, 50) == 1) {
            printf("  Bootstrap %g/%g...\r", b, B)
            displayflush()
        }
        if (_sw2023_one_boot_iter(X, Y, m0.d, m0.R, m0.Z, m0.U,
                                   bwmethod, method,
                                   phi_b, eff_b, seta_b)) {
            boot_phi[b, .]  = phi_b'
            boot_eff[b, .]  = eff_b'
            boot_seta[b, .] = seta_b'
        }
        else {
            n_fail++
        }
    }

    if (nolog == "" & B >= 50) {
        printf("  Bootstrap %g/%g... done\n", B, B)
        if (n_fail > 0) {
            printf("  Warning: %g/%g iterations failed\n", n_fail, B)
        }
        displayflush()
    }

    /* percentile CI */
    phi_ci  = _sw2023_colpercentile(boot_phi,  alpha/2, 1-alpha/2)
    eff_ci  = _sw2023_colpercentile(boot_eff,  alpha/2, 1-alpha/2)
    seta_ci = _sw2023_colpercentile(boot_seta, alpha/2, 1-alpha/2)

    _sw2023_put(v_phi_lo,  phi_ci[., 1],  touse)
    _sw2023_put(v_phi_hi,  phi_ci[., 2],  touse)
    _sw2023_put(v_eff_lo,  eff_ci[., 1],  touse)
    _sw2023_put(v_eff_hi,  eff_ci[., 2],  touse)
    _sw2023_put(v_seta_lo, seta_ci[., 1], touse)
    _sw2023_put(v_seta_hi, seta_ci[., 2], touse)
}


/* ── single pairs-bootstrap iteration ──────────────────────────── */
/* Returns 1 on success, 0 on failure.                              */
/* Resamples (X,Y) with replacement, rotates using FIXED d/R,      */
/* estimates moments on Z_b → evaluates at Z_orig.                 */
real scalar _sw2023_one_boot_iter(
        real matrix X_orig, real matrix Y_orig,
        real colvector d,   real matrix R,
        real matrix Z_orig, real colvector U_orig,
        string scalar bwmethod, string scalar method,
        real colvector phi_b,   /* output by "reference" (Mata pass-by-value; */
        real colvector eff_b,   /*  caller discards return if we return 0)     */
        real colvector seta_b)
{
    real matrix    Xb, Yb, Wb, WRb, Zb
    real colvector Ub, hb, r1_Zb, epsb, r1_b, r2_b, r3_b
    real colvector sigma_eta_b, sigma_eps_b, xi_b
    real scalar    n

    n = rows(X_orig)

    /* resample with replacement */
    real colvector idx
    idx = ceil(n :* runiform(n, 1))
    idx = rowmax((idx, J(n,1,1)))   /* guard: ensure >= 1 */
    idx = rowmin((idx, J(n,1,n)))   /* guard: ensure <= n */
    Xb = X_orig[idx, .]
    Yb = Y_orig[idx, .]

    /* rotate with FIXED d */
    Wb  = Xb, Yb
    WRb = Wb * R'
    real scalar pq
    pq  = cols(X_orig) + cols(Y_orig)
    Zb  = WRb[., 1..(pq-1)]
    Ub  = WRb[., pq]

    /* Silverman bandwidth on bootstrap sample */
    hb = _sw2023_silverman(Zb, n)

    /* get residuals at Zb (self-evaluation, needed for r2/r3) */
    if (_any_missing(Zb) | _any_missing(Ub)) return(0)
    r1_Zb = _sw2023_llls(Zb, Ub, hb)
    epsb  = Ub :- r1_Zb

    /* evaluate moments at Z_orig (external evaluation) */
    r1_b  = _sw2023_llls_ext(Zb, Ub,     hb, Z_orig)
    r2_b  = _sw2023_llls_ext(Zb, epsb:^2, hb, Z_orig)
    r3_b  = _sw2023_llls_ext(Zb, epsb:^3, hb, Z_orig)

    /* decompose */
    if (method == "hms") {
        sigma_eta_b = _sw2023_sigma_eta_hms(r3_b)
    } else {
        sigma_eta_b = _sw2023_sigma_eta_svkz(r3_b)
    }
    sigma_eps_b = _sw2023_sigma_eps(r2_b, sigma_eta_b)

    /* phi at Z_orig */
    phi_b = r1_b :+ sqrt(2 / 3.14159265358979) :* sigma_eta_b

    /* JLMS using U_orig */
    xi_b  = U_orig :- phi_b
    eff_b = _sw2023_jlms(xi_b, sigma_eta_b, sigma_eps_b)
    seta_b = sigma_eta_b

    return(1)
}


/* ── wild bootstrap significance test ──────────────────────────── */
/* PSVKZ (2024) Section 3.1                                         */
/* H0: E(eps^3 | Z) = const                                         */
/* T = Var(r3_hat) / Var(eps^3)                                     */
void _sw2023_wild_test(
        string scalar Xname, string scalar Yname,
        string scalar direction, string scalar method,
        string scalar bwmethod, string scalar nolog,
        real scalar B,
        string scalar T_name, string scalar p_name)
{
    struct _sw2023_fit scalar m0
    real matrix X, Y
    real colvector eps3, r3_obs, resid_c, r3_star, V, eps3_star
    real colvector T_boot
    real scalar T_obs, r3_var, eps3_var, b

    X = st_matrix(Xname)
    Y = st_matrix(Yname)

    m0 = _sw2023_fit_model(X, Y, direction, method, bwmethod)

    eps3  = m0.eps:^3
    r3_obs = m0.r3_hat

    /* centred residual: eta_i = eps3_i - r3_hat_i */
    real colvector resid
    resid   = eps3 :- r3_obs
    resid_c = resid :- mean(resid)   /* centred */

    /* observed test statistic */
    r3_var   = _sw2023_var(r3_obs)
    eps3_var = _sw2023_var(eps3)
    if (eps3_var < 1e-15) eps3_var = 1e-15
    T_obs = r3_var / eps3_var

    if (nolog == "") {
        printf("  T_obs = %10.6f\n", T_obs)
        printf("  Wild bootstrap (B=%g)...\n", B)
        displayflush()
    }

    /* wild bootstrap loop */
    T_boot = J(B, 1, .)
    for (b = 1; b <= B; b++) {
        if (nolog == "" & mod(b, 100) == 1) {
            printf("  %g/%g...\r", b, B)
            displayflush()
        }
        /* Rademacher: V_i = ±1 with prob 1/2 */
        V = 2 :* (runiform(rows(m0.Z), 1) :> 0.5) :- 1
        eps3_star = r3_obs :+ resid_c :* V

        /* re-estimate r3 with same Z and h_r3 */
        r3_star = _sw2023_llls(m0.Z, eps3_star, m0.h)

        real scalar r3s_var, eps3s_var
        r3s_var   = _sw2023_var(r3_star)
        eps3s_var = _sw2023_var(eps3_star)
        if (eps3s_var < 1e-15) eps3s_var = 1e-15
        T_boot[b] = r3s_var / eps3s_var
    }

    if (nolog == "") {
        printf("  %g/%g... done\n", B, B)
        displayflush()
    }

    real scalar p_value
    p_value = mean(T_boot :>= T_obs)

    st_numscalar(T_name, T_obs)
    st_numscalar(p_name, p_value)
}


/* ====================================================================
   Step-by-step building blocks
   ==================================================================== */

/* ── direction vector ───────────────────────────────────────────── */
real colvector _sw2023_direction(real matrix X, real matrix Y,
                                  string scalar direction)
{
    real colvector mx, my, d_raw
    real scalar    nm

    if (direction == "mean") {
        mx = mean(X)'; my = mean(Y)'
    } else {
        mx = colmedian(X)'; my = colmedian(Y)'
    }
    d_raw = (-mx \ my)
    nm    = sqrt(quadsum(d_raw:^2))
    return(d_raw :/ nm)
}


/* ── rotation matrix via Gram-Schmidt ───────────────────────────── */
real matrix _sw2023_rotation(real colvector d)
{
    real scalar    m, k, i
    real matrix    V
    real colvector e, v

    m = rows(d)
    V = J(m, m, 0)
    k = 0

    for (i = 1; i <= m; i++) {
        e    = J(m, 1, 0); e[i] = 1
        v    = e :- quadsum(e :* d) * d
        if (k >= 1) v = v :- V[., 1..k] * (V[., 1..k]' * v)
        if (sqrt(quadsum(v:^2)) > 1e-10) {
            k = k + 1
            V[., k] = v :/ sqrt(quadsum(v:^2))
        }
        if (k == m - 1) break
    }
    V[., m] = d
    return(V')
}


/* ── Silverman bandwidth ────────────────────────────────────────── */
real rowvector _sw2023_silverman(real matrix Z, real scalar n)
{
    real scalar    d
    real rowvector s

    d = cols(Z)
    s = sqrt(diagonal(variance(Z))')
    s = rowmax((s, J(1, d, 1e-6)))
    return(1.06 :* s :* n^(-1 / (d + 4)))
}


/* ── LOO-CV bandwidth (golden section over scale factor c) ──────── */
real rowvector _sw2023_loocv(real matrix Z, real colvector u)
{
    real scalar    c_lo, c_hi, c_opt, cv1, cv2, tol, gold, c1, c2
    real rowvector h0

    h0   = _sw2023_silverman(Z, rows(Z))
    tol  = 1e-4; gold = 0.6180339887
    c_lo = 0.1; c_hi = 3.0

    c1 = c_hi - gold*(c_hi-c_lo); c2 = c_lo + gold*(c_hi-c_lo)
    cv1 = _sw2023_cv_eval(c1, h0, Z, u)
    cv2 = _sw2023_cv_eval(c2, h0, Z, u)

    while ((c_hi - c_lo) > tol) {
        if (cv1 < cv2) {
            c_hi=c2; c2=c1; cv2=cv1
            c1 = c_hi - gold*(c_hi-c_lo)
            cv1 = _sw2023_cv_eval(c1, h0, Z, u)
        } else {
            c_lo=c1; c1=c2; cv1=cv2
            c2 = c_lo + gold*(c_hi-c_lo)
            cv2 = _sw2023_cv_eval(c2, h0, Z, u)
        }
    }
    return((c_lo+c_hi)/2 :* h0)
}

real scalar _sw2023_cv_eval(real scalar c, real rowvector h0,
                              real matrix Z, real colvector u)
{
    real matrix    Zh, diff
    real colvector K, phi, h_ii, cv_err
    real scalar    n, d, i, det_v
    real rowvector xi, h

    h  = c :* h0; Zh = Z :/ h
    n  = rows(Z); d  = cols(Z)
    phi  = J(n, 1, 0); h_ii = J(n, 1, 0)

    for (i = 1; i <= n; i++) {
        xi   = Zh[i, .]
        diff = Zh :- J(n, 1, xi)
        K    = exp(-0.5 :* rowsum(diff:^2))
        real matrix Ai, WA, A, C
        Ai = J(n, d+1, 0)
        Ai[., 1] = J(n, 1, 1)
        Ai[., 2..d+1] = Z :- J(n, 1, Z[i, .])
        WA = K :* Ai; A = Ai'*WA
        det_v = det(A)
        if (abs(det_v) < 1e-14) { h_ii[i]=1; continue }
        C = luinv(A)
        h_ii[i] = C[1,1] * K[i]
        phi[i]  = (lusolve(A, WA'*u))[1]
    }
    cv_err = (u :- phi) :/ (1 :- h_ii)
    return(mean(cv_err:^2))
}


/* ── LLLS: self-evaluation ─────────────────────────────────────── */
real colvector _sw2023_llls(real matrix Z, real colvector u,
                              real rowvector h)
{
    real matrix    Zh, diff, Ai, WA, A
    real colvector phi, K
    real scalar    n, d, i, det_v
    real rowvector xi

    n=rows(Z); d=cols(Z); Zh=Z:/h; phi=J(n,1,0)

    for (i = 1; i <= n; i++) {
        xi   = Zh[i, .]
        diff = Zh :- J(n, 1, xi)
        K    = exp(-0.5 :* rowsum(diff:^2))
        Ai   = J(n, d+1, 0)
        Ai[., 1] = J(n, 1, 1)
        Ai[., 2..d+1] = Z :- J(n, 1, Z[i, .])
        WA = K :* Ai; A = Ai'*WA
        det_v = det(A)
        if (abs(det_v) < 1e-14) { phi[i] = mean(u); continue }
        phi[i] = (lusolve(A, WA'*u))[1]
    }
    return(phi)
}


/* ── LLLS: external evaluation (bootstrap) ─────────────────────── */
/* Fits LLLS on (Zb, ub) but evaluates at Zeval points.            */
/* Key bootstrap property: moment functions are estimated on the    */
/* resampled data but evaluated at the original observation points. */
real colvector _sw2023_llls_ext(
        real matrix Zb, real colvector ub, real rowvector h,
        real matrix Zeval)
{
    real scalar    n_b, n_e, d, i, det_v
    real colvector phi, wi, ub_mean
    real matrix    diff_i, Ai, WA, A
    real rowvector z_i

    n_b   = rows(Zb); n_e = rows(Zeval); d = cols(Zb)
    phi   = J(n_e, 1, 0)
    ub_mean = mean(ub)

    for (i = 1; i <= n_e; i++) {
        z_i    = Zeval[i, .]
        diff_i = (Zb :- J(n_b, 1, z_i)) :/ J(n_b, 1, h)
        wi     = exp(-0.5 :* rowsum(diff_i:^2))
        Ai     = J(n_b, d+1, 0)
        Ai[., 1]       = J(n_b, 1, 1)
        Ai[., 2..d+1]  = Zb :- J(n_b, 1, z_i)
        WA = wi :* Ai; A = Ai'*WA
        det_v = det(A)
        if (abs(det_v) < 1e-14) { phi[i] = ub_mean[1,1]; continue }
        phi[i] = (lusolve(A, WA'*ub))[1]
    }
    return(phi)
}


/* ── sigma_eta: HMS element-wise ────────────────────────────────── */
/* HMS: wrong-skewness correction (Hafner, Manner, Simar 2018)      */
/* sigma_eta_i = cbrt(-r3_i / A3+)  if r3_i <= 0                   */
/*             = cbrt( r3_i / A3-)  if r3_i >  0  (wrong skewness) */
real colvector _sw2023_sigma_eta_hms(real colvector r3)
{
    real scalar    A3P, A3N
    real colvector val_neg, val_pos, seta

    A3P = 0.21773; A3N = 0.016741474

    val_neg = _cbrt_v(-r3 :/ A3P)
    val_pos = _cbrt_v( r3 :/ A3N)

    seta = (r3 :<= 0) :* val_neg :+ (r3 :> 0) :* val_pos
    return(rowmax((seta, J(rows(r3), 1, 0))))
}


/* ── sigma_eta: SVKZ element-wise ────────────────────────────────── */
/* SVKZ: max{0, cbrt(-r3 / A3+)}  (truncates wrong-skewness cases)  */
real colvector _sw2023_sigma_eta_svkz(real colvector r3)
{
    real scalar A3P
    A3P = 0.21773
    return(rowmax((_cbrt_v(-r3 :/ A3P), J(rows(r3), 1, 0))))
}


/* ── sigma_eps element-wise ─────────────────────────────────────── */
/* SW(2023) Eq. (3.10):                                             */
/*   r2_i = sigma_eps_i^2 + (pi-2)/pi * sigma_eta_i^2              */
real colvector _sw2023_sigma_eps(real colvector r2, real colvector sigma_eta)
{
    real scalar    PI
    real colvector s2

    PI = 3.14159265358979
    s2 = r2 :- (PI-2)/PI :* sigma_eta:^2
    return(sqrt(rowmax((s2, J(rows(r2), 1, 0)))))
}


/* ── JLMS efficiency: exp(-E[eta|xi]) ──────────────────────────── */
/* Matches Python jlms_efficiency() in decompose.py exactly.        */
/* xi_i = U_i - phi_hat_i  (composite residual)                     */
/* mu*_i    = -xi_i * sigma_eta_i^2 / (sigma_eta_i^2+sigma_eps_i^2) */
/* sigma*_i = sigma_eps_i*sigma_eta_i / sqrt(sigma_eta_i^2+...)      */
/* E[eta|xi] = mu* + sigma* * phi(mu*/sigma*) / Phi(mu*/sigma*)      */
/* eff_i  = exp(-E[eta_i|xi_i])                                      */
real colvector _sw2023_jlms(real colvector xi,
                              real colvector sigma_eta,
                              real colvector sigma_eps)
{
    real scalar    PI
    real colvector sl2, mu_star, sig_star, ratio, eta_hat, eff
    real scalar    n, i

    n  = rows(xi)
    PI = 3.14159265358979

    sl2      = sigma_eta:^2 :+ sigma_eps:^2
    sl2      = rowmax((sl2, J(n, 1, 1e-15)))

    mu_star  = -xi :* sigma_eta:^2 :/ sl2
    sig_star = sigma_eta :* sigma_eps :/ sqrt(sl2)
    sig_star = rowmax((sig_star, J(n, 1, 1e-15)))

    ratio    = mu_star :/ sig_star

    /* eta_hat = mu* + sigma* * normalden(ratio) / normal(ratio) */
    real colvector pdf_r, cdf_r
    pdf_r  = J(n, 1, 0)
    cdf_r  = J(n, 1, 0)
    for (i = 1; i <= n; i++) {
        pdf_r[i] = normalden(ratio[i])
        cdf_r[i] = normal(ratio[i])
    }
    cdf_r   = rowmax((cdf_r, J(n, 1, 1e-15)))
    eta_hat = mu_star :+ sig_star :* pdf_r :/ cdf_r
    eta_hat = rowmax((eta_hat, J(n, 1, 0)))

    eff = exp(-eta_hat)
    eff = rowmax((eff, J(n, 1, 0)))
    eff = rowmin((eff, J(n, 1, 1)))
    return(eff)
}


/* ====================================================================
   Bootstrap utilities
   ==================================================================== */

/* ── column-wise percentile CI ──────────────────────────────────── */
/* Input:  boot (B × n), p_lo and p_hi in (0,1)                     */
/* Output: ci (n × 2), col1=lower, col2=upper                        */
real matrix _sw2023_colpercentile(real matrix boot, real scalar p_lo,
                                   real scalar p_hi)
{
    real scalar n, b, n_valid
    real matrix ci
    real colvector col, col_s

    n  = cols(boot)
    ci = J(n, 2, .)

    for (b = 1; b <= n; b++) {
        col   = boot[., b]
        col_s = select(col, col :!= .)
        n_valid = rows(col_s)
        if (n_valid < 2) continue
        col_s = sort(col_s, 1)
        ci[b, 1] = _sw2023_quantile_sorted(col_s, p_lo)
        ci[b, 2] = _sw2023_quantile_sorted(col_s, p_hi)
    }
    return(ci)
}


/* ── linear-interpolation quantile from sorted vector ───────────── */
real scalar _sw2023_quantile_sorted(real colvector vs, real scalar p)
{
    real scalar n, idx_r, idx_lo, idx_hi

    n      = rows(vs)
    idx_r  = 1 + p * (n - 1)
    idx_lo = max((1, floor(idx_r)))
    idx_hi = min((n, ceil(idx_r)))
    return(vs[idx_lo] + (idx_r - idx_lo) * (vs[idx_hi] - vs[idx_lo]))
}


/* ── variance (scalar) ──────────────────────────────────────────── */
real scalar _sw2023_var(real colvector v)
{
    real scalar n, m
    n = rows(v); if (n < 2) return(0)
    m = mean(v)
    return(quadsum((v :- m):^2) / (n - 1))
}


/* ── cube root (handles negative) ──────────────────────────────── */
real scalar _cbrt_s(real scalar x)
{
    if (x >= 0) return(x^(1/3))
    else return(-((-x)^(1/3)))
}

real colvector _cbrt_v(real colvector x)
{
    real colvector r
    real scalar    i
    r = J(rows(x), 1, 0)
    for (i = 1; i <= rows(x); i++) r[i] = _cbrt_s(x[i])
    return(r)
}


/* ── check for missing values ───────────────────────────────────── */
real scalar _any_missing(real matrix A)
{
    return(any(vec(A) :== .))
}


/* ── write column vector to Stata variable ──────────────────────── */
void _sw2023_put(string scalar vname, real colvector v,
                  string scalar touse)
{
    real scalar    i, j
    real colvector flag

    flag = st_data(., touse)
    j = 1
    for (i = 1; i <= rows(flag); i++) {
        if (flag[i] == 1) {
            st_store(i, vname, v[j])
            j++
        }
    }
}

end
