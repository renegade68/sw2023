*! sw2023test v0.2.0 — Wild bootstrap significance test
*! H0: E(eps^3 | Z) = const  (homogeneous inefficiency)
*! Reference: PSVKZ (2024, Econometric Reviews) Section 3.1
*!
*! Syntax:
*!   sw2023test yvarlist, inputs(xvarlist)
*!              [direction(mean|median) method(hms|svkz)
*!               bandwidth(silverman|loocv) reps(#) nolog]
*!
*! Notes:
*!   - Self-contained: all Stata/Mata code in this single file.
*!   - If sw2023.ado is already loaded, Mata functions are reused.

program define sw2023test, rclass
    version 16.1

    syntax varlist(min=1 numeric) ,         ///
        INputs(varlist numeric min=1)        ///
        [ DIRection(string)                  ///
          METHod(string)                     ///
          BANDwidth(string)                  ///
          REPs(integer 499)                  ///
          nolog ]

    if "`direction'" == ""  local direction "mean"
    if "`method'"    == ""  local method    "hms"
    if "`bandwidth'" == ""  local bandwidth "silverman"

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
    if `N' < 10 {
        di as error "Insufficient observations (need >= 10)"
        exit 2001
    }

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
        mkmat `inputs'  if `touse', matrix(`Xm')
        mkmat `varlist' if `touse', matrix(`Ym')
    }

    tempname T_obs p_val
    mata: _sw2023t_wild_test("`Xm'", "`Ym'",          ///
                              "`direction'", "`method'", ///
                              "`bandwidth'", "`log'",    ///
                              `reps',                    ///
                              "`T_obs'", "`p_val'")

    local T_val  = `T_obs'
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
   Mata — self-contained implementation for sw2023test
   (Uses _sw2023t_ prefix to avoid conflicts with sw2023.ado)
   ==================================================================== */

mata:

/* ── struct ─────────────────────────────────────────────────────── */
struct _sw2023t_fit {
    real colvector d
    real matrix    R, Z
    real colvector U, h, r1_hat, eps, r2_hat, r3_hat
    real colvector sigma_eta, sigma_eps, phi_hat, xi, eff
}


/* ── core estimation ────────────────────────────────────────────── */
struct _sw2023t_fit scalar _sw2023t_fit_model(
        real matrix X, real matrix Y,
        string scalar direction, string scalar method,
        string scalar bwmethod)
{
    struct _sw2023t_fit scalar m
    real scalar    n, p, q
    real matrix    W, WR

    n=rows(X); p=cols(X); q=cols(Y)
    W = X, Y
    m.d = _sw2023t_direction(X, Y, direction)
    m.R = _sw2023t_rotation(m.d)
    WR  = W * m.R'
    m.Z = WR[., 1..(p+q-1)]
    m.U = WR[., p+q]

    if (bwmethod == "loocv") m.h = _sw2023t_loocv(m.Z, m.U)
    else                     m.h = _sw2023t_silverman(m.Z, n)

    m.r1_hat = _sw2023t_llls(m.Z, m.U, m.h)
    m.eps    = m.U :- m.r1_hat
    m.r2_hat = _sw2023t_llls(m.Z, m.eps:^2, m.h)
    m.r3_hat = _sw2023t_llls(m.Z, m.eps:^3, m.h)

    if (method == "hms") m.sigma_eta = _sw2023t_seta_hms(m.r3_hat)
    else                 m.sigma_eta = _sw2023t_seta_svkz(m.r3_hat)
    m.sigma_eps = _sw2023t_seps(m.r2_hat, m.sigma_eta)
    m.phi_hat   = m.r1_hat :+ sqrt(2/3.14159265358979) :* m.sigma_eta
    m.xi        = m.U :- m.phi_hat
    m.eff       = _sw2023t_jlms(m.xi, m.sigma_eta, m.sigma_eps)
    return(m)
}


/* ── wild bootstrap test ────────────────────────────────────────── */
void _sw2023t_wild_test(
        string scalar Xname, string scalar Yname,
        string scalar direction, string scalar method,
        string scalar bwmethod, string scalar nolog,
        real scalar B, string scalar T_name, string scalar p_name)
{
    struct _sw2023t_fit scalar m0
    real colvector eps3, r3_obs, resid_c, V, eps3_star, r3_star, T_boot
    real scalar    T_obs, r3v, e3v, b, r3sv, e3sv

    X = st_matrix(Xname); Y = st_matrix(Yname)
    m0   = _sw2023t_fit_model(X, Y, direction, method, bwmethod)

    eps3    = m0.eps:^3
    r3_obs  = m0.r3_hat
    resid_c = (eps3 :- r3_obs) :- mean(eps3 :- r3_obs)

    r3v  = _sw2023t_var(r3_obs)
    e3v  = _sw2023t_var(eps3)
    if (e3v < 1e-15) e3v = 1e-15
    T_obs = r3v / e3v

    if (nolog == "") {
        printf("  T_obs = %10.6f\n", T_obs)
        printf("  Wild bootstrap (B=%g)...\n", B)
        displayflush()
    }

    T_boot = J(B, 1, .)
    for (b = 1; b <= B; b++) {
        if (nolog == "" & mod(b, 100) == 1) {
            printf("  %g/%g...\r", b, B)
            displayflush()
        }
        V          = 2 :* (runiform(rows(m0.Z), 1) :> 0.5) :- 1
        eps3_star  = r3_obs :+ resid_c :* V
        r3_star    = _sw2023t_llls(m0.Z, eps3_star, m0.h)
        r3sv       = _sw2023t_var(r3_star)
        e3sv       = _sw2023t_var(eps3_star)
        if (e3sv < 1e-15) e3sv = 1e-15
        T_boot[b]  = r3sv / e3sv
    }

    if (nolog == "") {
        printf("  %g/%g... done\n", B, B)
        displayflush()
    }

    st_numscalar(T_name, T_obs)
    st_numscalar(p_name, mean(T_boot :>= T_obs))
}


/* ── direction ─────────────────────────────────────────────────── */
real colvector _sw2023t_direction(real matrix X, real matrix Y,
                                   string scalar dir)
{
    real colvector mx, my, d
    real scalar    nm
    if (dir == "mean") { mx = mean(X)'; my = mean(Y)' }
    else               { mx = colmedian(X)'; my = colmedian(Y)' }
    d  = (-mx \ my)
    nm = sqrt(quadsum(d:^2))
    return(d :/ nm)
}


/* ── rotation ──────────────────────────────────────────────────── */
real matrix _sw2023t_rotation(real colvector d)
{
    real scalar    m, k, i
    real matrix    V
    real colvector e, v
    m=rows(d); V=J(m,m,0); k=0
    for (i=1; i<=m; i++) {
        e=J(m,1,0); e[i]=1
        v=e :- quadsum(e:*d)*d
        if (k>=1) v=v :- V[.,1..k]*(V[.,1..k]'*v)
        if (sqrt(quadsum(v:^2)) > 1e-10) { k++; V[.,k]=v:/sqrt(quadsum(v:^2)) }
        if (k==m-1) break
    }
    V[.,m]=d; return(V')
}


/* ── Silverman ─────────────────────────────────────────────────── */
real rowvector _sw2023t_silverman(real matrix Z, real scalar n)
{
    real scalar d; real rowvector s
    d=cols(Z); s=sqrt(diagonal(variance(Z))')
    s=rowmax((s, J(1,d,1e-6)))
    return(1.06 :* s :* n^(-1/(d+4)))
}


/* ── LOO-CV ────────────────────────────────────────────────────── */
real rowvector _sw2023t_loocv(real matrix Z, real colvector u)
{
    real scalar    g, clo, chi, c1, c2, v1, v2, tol
    real rowvector h0
    h0=_sw2023t_silverman(Z,rows(Z)); tol=1e-4; g=0.6180339887
    clo=0.1; chi=3.0
    c1=chi-g*(chi-clo); c2=clo+g*(chi-clo)
    v1=_sw2023t_cv_eval(c1,h0,Z,u); v2=_sw2023t_cv_eval(c2,h0,Z,u)
    while((chi-clo)>tol) {
        if (v1<v2) { chi=c2; c2=c1; v2=v1; c1=chi-g*(chi-clo); v1=_sw2023t_cv_eval(c1,h0,Z,u) }
        else       { clo=c1; c1=c2; v1=v2; c2=clo+g*(chi-clo); v2=_sw2023t_cv_eval(c2,h0,Z,u) }
    }
    return((clo+chi)/2 :* h0)
}

real scalar _sw2023t_cv_eval(real scalar c, real rowvector h0,
                               real matrix Z, real colvector u)
{
    real rowvector h; real matrix Zh, diff, Ai, WA, A, C
    real colvector K, phi, hii, err
    real scalar    n, d, i, dv
    h=c:*h0; Zh=Z:/h; n=rows(Z); d=cols(Z)
    phi=J(n,1,0); hii=J(n,1,0)
    for (i=1; i<=n; i++) {
        diff=Zh:-J(n,1,Zh[i,.])
        K=exp(-0.5:*rowsum(diff:^2))
        Ai=J(n,d+1,0); Ai[.,1]=J(n,1,1); Ai[.,2..d+1]=Z:-J(n,1,Z[i,.])
        WA=K:*Ai; A=Ai'*WA; dv=det(A)
        if (abs(dv)<1e-14) { hii[i]=1; continue }
        C=luinv(A); hii[i]=C[1,1]*K[i]
        phi[i]=(lusolve(A,WA'*u))[1]
    }
    err=(u:-phi):/(1:-hii); return(mean(err:^2))
}


/* ── LLLS (self-evaluation) ─────────────────────────────────────── */
real colvector _sw2023t_llls(real matrix Z, real colvector u,
                               real rowvector h)
{
    real matrix    Zh, diff, Ai, WA, A
    real colvector phi, K
    real scalar    n, d, i, dv
    n=rows(Z); d=cols(Z); Zh=Z:/h; phi=J(n,1,0)
    for (i=1; i<=n; i++) {
        diff=Zh:-J(n,1,Zh[i,.]); K=exp(-0.5:*rowsum(diff:^2))
        Ai=J(n,d+1,0); Ai[.,1]=J(n,1,1); Ai[.,2..d+1]=Z:-J(n,1,Z[i,.])
        WA=K:*Ai; A=Ai'*WA; dv=det(A)
        if (abs(dv)<1e-14) { phi[i]=mean(u); continue }
        phi[i]=(lusolve(A,WA'*u))[1]
    }
    return(phi)
}


/* ── sigma_eta: HMS ─────────────────────────────────────────────── */
real colvector _sw2023t_seta_hms(real colvector r3)
{
    real scalar A3P, A3N; real colvector vn, vp, s
    A3P=0.21773; A3N=0.016741474
    vn=_sw2023t_cbrtv(-r3:/A3P); vp=_sw2023t_cbrtv(r3:/A3N)
    s=(r3:<=0):*vn :+ (r3:>0):*vp
    return(rowmax((s, J(rows(r3),1,0))))
}


/* ── sigma_eta: SVKZ ────────────────────────────────────────────── */
real colvector _sw2023t_seta_svkz(real colvector r3)
{
    return(rowmax((_sw2023t_cbrtv(-r3:/0.21773), J(rows(r3),1,0))))
}


/* ── sigma_eps ──────────────────────────────────────────────────── */
real colvector _sw2023t_seps(real colvector r2, real colvector se)
{
    real scalar PI; PI=3.14159265358979
    return(sqrt(rowmax((r2:-(PI-2)/PI:*se:^2, J(rows(r2),1,0)))))
}


/* ── JLMS ───────────────────────────────────────────────────────── */
real colvector _sw2023t_jlms(real colvector xi, real colvector se,
                               real colvector sv)
{
    real scalar    n, i; real colvector sl2, mu, ss, ra, pdr, cdr, eta, eff
    n=rows(xi)
    sl2=rowmax((se:^2:+sv:^2, J(n,1,1e-15)))
    mu=(-xi:*se:^2):/sl2
    ss=rowmax((se:*sv:/sqrt(sl2), J(n,1,1e-15)))
    ra=mu:/ss
    pdr=J(n,1,0); cdr=J(n,1,0)
    for (i=1; i<=n; i++) { pdr[i]=normalden(ra[i]); cdr[i]=normal(ra[i]) }
    cdr=rowmax((cdr, J(n,1,1e-15)))
    eta=rowmax((mu:+ss:*pdr:/cdr, J(n,1,0)))
    eff=rowmax((rowmin((exp(-eta), J(n,1,1))), J(n,1,0)))
    return(eff)
}


/* ── variance ───────────────────────────────────────────────────── */
real scalar _sw2023t_var(real colvector v)
{
    real scalar n, m
    n=rows(v); if (n<2) return(0)
    m=mean(v); return(quadsum((v:-m):^2)/(n-1))
}


/* ── cube root ──────────────────────────────────────────────────── */
real scalar _sw2023t_cbrts(real scalar x) {
    if (x>=0) return(x^(1/3))
    else return(-((-x)^(1/3)))
}
real colvector _sw2023t_cbrtv(real colvector x) {
    real colvector r; real scalar i
    r=J(rows(x),1,0)
    for (i=1; i<=rows(x); i++) r[i]=_sw2023t_cbrts(x[i])
    return(r)
}

end
