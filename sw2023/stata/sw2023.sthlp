{smcl}
{* sw2023.sthlp  v0.2.0  2025-04-13}{...}
{hline}
help for {hi:sw2023} and {hi:sw2023test}
{hline}

{title:Title}

{phang}
{bf:sw2023} {hline 2} Nonparametric multiple-output stochastic frontier analysis
(Simar & Wilson 2023)

{phang}
{bf:sw2023test} {hline 2} Wild bootstrap significance test for inefficiency
heterogeneity (PSVKZ 2024)


{title:Syntax}

{p 8 17 2}
{cmd:sw2023}
{varlist} (output variables)
{cmd:,}
{opt inp:uts(varlist)}
[{it:options}]

{p 8 17 2}
{cmd:sw2023test}
{varlist} (output variables)
{cmd:,}
{opt inp:uts(varlist)}
[{it:test_options}]

{synoptset 24 tabbed}{...}
{synopthdr:sw2023 options}
{synoptline}
{syntab:Required}
{synopt:{opt inp:uts(varlist)}}input variables (factors of production){p_end}

{syntab:Model}
{synopt:{opt dir:ection(mean|median)}}direction vector; default {cmd:mean}{p_end}
{synopt:{opt meth:od(hms|svkz)}}sigma_eta estimator; default {cmd:hms}{p_end}
{synopt:{opt band:width(silverman|loocv)}}bandwidth selection method;
  default {cmd:silverman}{p_end}

{syntab:Output}
{synopt:{opt gen:erate(prefix)}}prefix for generated variables;
  default {cmd:sw_}{p_end}

{syntab:Bootstrap CI}
{synopt:{opt boot:strap(#)}}number of pairs bootstrap replications
  (0 = no bootstrap); default {cmd:0}{p_end}
{synopt:{opt level(#)}}confidence level for bootstrap CI; default {cmd:95}{p_end}

{syntab:Display}
{synopt:{opt nolog}}suppress header and summary output{p_end}
{synoptline}

{synoptset 24 tabbed}{...}
{synopthdr:sw2023test options}
{synoptline}
{syntab:Required}
{synopt:{opt inp:uts(varlist)}}input variables{p_end}

{syntab:Model}
{synopt:{opt dir:ection(mean|median)}}direction vector; default {cmd:mean}{p_end}
{synopt:{opt meth:od(hms|svkz)}}sigma_eta estimator; default {cmd:hms}{p_end}
{synopt:{opt band:width(silverman|loocv)}}bandwidth selection method;
  default {cmd:silverman}{p_end}

{syntab:Test}
{synopt:{opt rep:s(#)}}number of wild bootstrap replications;
  default {cmd:499} (minimum 99){p_end}

{syntab:Display}
{synopt:{opt nolog}}suppress output{p_end}
{synoptline}
{p2colreset}{...}


{title:Description}

{pstd}
{cmd:sw2023} implements the fully nonparametric multiple-output stochastic
frontier estimator of Simar & Wilson (2023) in Stata Mata, with no Python
dependency. The command estimates a directional output distance function
frontier using local linear least squares (LLLS) and recovers individual
efficiency scores via the JLMS method.

{pstd}
The model accommodates any number of inputs {it:p} and outputs {it:q}.
Observations are projected along a direction vector {bf:d} in joint
(input, output) space; the frontier and efficiency scores are estimated
in the rotated coordinate system.

{pstd}
{cmd:sw2023test} tests the null hypothesis of {it:homogeneous} inefficiency
(H0: E(eps^3 | Z) = constant) against heterogeneous inefficiency using
the wild bootstrap method of Parmeter, Simar, Van Keilegom & Zelenyuk (2024).


{title:Options for sw2023}

{dlgtab:Model}

{phang}
{opt inputs(varlist)} specifies the input variables (factors of production).
This option is required.

{phang}
{opt direction(mean|median)} specifies how the direction vector {bf:d} is
constructed from the data.
{cmd:mean} (default) uses the component-wise sample mean of (X, Y);
{cmd:median} uses the component-wise sample median.

{phang}
{opt method(hms|svkz)} specifies the estimator for the conditional standard
deviation sigma_eta(Z).
{cmd:hms} (default) uses the Hafner, Manner & Simar (2018) approach, which
handles the "wrong skewness" case when r_hat_3(Z) > 0;
{cmd:svkz} uses the Parmeter, Simar, Van Keilegom & Zelenyuk (2024) estimator, which
truncates sigma_eta at zero when r_hat_3(Z) > 0.

{phang}
{opt bandwidth(silverman|loocv)} specifies the bandwidth selection method.
{cmd:silverman} (default) uses the Silverman rule-of-thumb
h = sigma * n^{-1/(d+4)};
{cmd:loocv} minimises leave-one-out cross-validation.
Note: {cmd:loocv} is significantly slower and is automatically replaced by
{cmd:silverman} when bootstrap replications are requested.

{dlgtab:Output}

{phang}
{opt generate(prefix)} sets the prefix for all generated output variables.
The default prefix is {cmd:sw_}. See {it:Generated variables} below for
the complete list.

{dlgtab:Bootstrap CI}

{phang}
{opt bootstrap(#)} requests the SW pairs bootstrap with {it:#} replications.
Setting {it:#} = 0 (the default) skips the bootstrap. A minimum of 99
replications is recommended for 95% CIs; 199 or 499 are conventional
choices. Bootstrap CIs use the Silverman bandwidth for speed.

{phang}
{opt level(#)} sets the nominal coverage of the bootstrap confidence
intervals as a percentage. The default is 95.


{title:Options for sw2023test}

{phang}
{opt reps(#)} specifies the number of wild bootstrap replications for the
significance test. The default is 499; at least 99 is required.

{pstd}
All model options ({opt direction()}, {opt method()}, {opt bandwidth()},
{opt generate()}, {opt nolog}) are shared with {cmd:sw2023}.


{title:Generated variables}

{pstd}
{cmd:sw2023} creates the following variables (using prefix {cmd:sw_} by default):

{synoptset 22}{...}
{synopt:{cmd:sw_efficiency}}Individual efficiency score exp(-eta_hat),
  in (0, 1].{p_end}
{synopt:{cmd:sw_phi_hat}}Estimated frontier value phi_hat(Z_i).{p_end}
{synopt:{cmd:sw_sigma_eta}}Estimated conditional inefficiency s.d.
  sigma_eta_hat(Z_i).{p_end}
{synopt:{cmd:sw_sigma_eps}}Estimated conditional noise s.d.
  sigma_eps_hat(Z_i).{p_end}
{synopt:{cmd:sw_r1}}Conditional mean r_hat_1(Z_i) = E[U | Z_i].{p_end}
{synopt:{cmd:sw_r2}}Conditional second moment r_hat_2(Z_i) = E[eps^2 | Z_i].{p_end}
{synopt:{cmd:sw_r3}}Conditional third moment r_hat_3(Z_i) = E[eps^3 | Z_i].{p_end}

{pstd}
When {opt bootstrap(#)} > 0, six additional CI variables are created:

{synoptset 22}{...}
{synopt:{cmd:sw_phi_lo}, {cmd:sw_phi_hi}}Lower and upper bootstrap CI for
  the frontier phi_hat(Z_i).{p_end}
{synopt:{cmd:sw_eff_lo}, {cmd:sw_eff_hi}}Lower and upper bootstrap CI for
  individual efficiency.{p_end}
{synopt:{cmd:sw_seta_lo}, {cmd:sw_seta_hi}}Lower and upper bootstrap CI for
  sigma_eta(Z_i).{p_end}


{title:Stored results}

{pstd}
{cmd:sw2023} is an {cmd:eclass} command; results are stored in {cmd:e()}.

{synoptset 20 tabbed}{...}
{syntab:Matrices}
{synopt:{cmd:e(b)}}Row vector of mean efficiency (scalar summary).{p_end}

{pstd}
{cmd:sw2023test} is an {cmd:rclass} command; results are stored in {cmd:r()}.

{synoptset 20 tabbed}{...}
{syntab:Scalars}
{synopt:{cmd:r(statistic)}}Observed test statistic T =
  Var(r_hat_3(Z)) / Var(eps^3).{p_end}
{synopt:{cmd:r(p_value)}}Bootstrap p-value for the test of H0.{p_end}
{synopt:{cmd:r(reps)}}Number of wild bootstrap replications used.{p_end}


{title:Examples}

{pstd}Setup: load Norwegian agricultural panel data and select a cross-section.{p_end}

{phang2}{cmd:. use "norway.dta", clear}{p_end}
{phang2}{cmd:. keep if year == 2001}{p_end}

{pstd}
{bf:Example 1}: Point estimation with HMS method and Silverman bandwidth.{p_end}

{phang2}{cmd:. sw2023 y1 y2, inputs(x1 x2 x3 x4) direction(mean) method(hms)}{p_end}
{phang2}{cmd:. summarize sw_efficiency}{p_end}
{phang2}{cmd:. histogram sw_efficiency, percent title("Efficiency distribution")}{p_end}

{pstd}
{bf:Example 2}: LOO-CV bandwidth (more accurate, slower).{p_end}

{phang2}{cmd:. sw2023 y1 y2, inputs(x1 x2 x3 x4) bandwidth(loocv)}{p_end}

{pstd}
{bf:Example 3}: Pairs bootstrap confidence intervals (B = 199, 95% CI).{p_end}

{phang2}{cmd:. sw2023 y1 y2, inputs(x1 x2 x3 x4) bootstrap(199) level(95)}{p_end}
{phang2}{cmd:. list sw_efficiency sw_eff_lo sw_eff_hi in 1/10}{p_end}

{pstd}
{bf:Example 4}: Custom output variable prefix.{p_end}

{phang2}{cmd:. sw2023 y1 y2 y3, inputs(x1 x2 x3) generate(myest_) nolog}{p_end}
{phang2}{cmd:. summarize myest_efficiency myest_sigma_eta}{p_end}

{pstd}
{bf:Example 5}: Wild bootstrap significance test for inefficiency heterogeneity.{p_end}

{phang2}{cmd:. sw2023test y1 y2, inputs(x1 x2 x3 x4) reps(499)}{p_end}
{phang2}{cmd:. display "p-value = " r(p_value)}{p_end}

{pstd}
{bf:Example 6}: Full workflow — estimate then test.{p_end}

{phang2}{cmd:. sw2023 y1 y2, inputs(x1 x2 x3 x4) bootstrap(199)}{p_end}
{phang2}{cmd:. sw2023test y1 y2, inputs(x1 x2 x3 x4) reps(299)}{p_end}
{phang2}{cmd:. if r(p_value) < 0.05 {c -(}}{p_end}
{phang2}{cmd:.     di "Reject H0: inefficiency distribution varies with Z"}{p_end}
{phang2}{cmd:. {c )-}}{p_end}


{title:Remarks}

{pstd}
{bf:Minimum sample size.} At least 10 complete observations are required.
Observations with any missing value in the output or input variables are
automatically excluded.

{pstd}
{bf:Log transform.} {cmd:sw2023} does not apply a log transform; input and
output variables must already be in the desired scale (log or level) before
calling the command.

{pstd}
{bf:Bootstrap and bandwidth.} When {opt bootstrap()} is specified together
with {opt bandwidth(loocv)}, the bandwidth is automatically switched to
{cmd:silverman} for the bootstrap replications only (with a notice), since
LOO-CV inside a bootstrap loop would be prohibitively slow.

{pstd}
{bf:Stata version.} Requires Stata 16.1 or later (Mata required).


{title:References}

{phang}
Hafner, C.M., Manner, H. and Simar, L. (2018).
The "wrong skewness" problem in stochastic frontier models: A new approach.
{it:Econometric Reviews}, 37(4), 380–400.

{phang}
Jondrow, J., Lovell, C.A.K., Materov, I.S. and Schmidt, P. (1982).
On the estimation of technical inefficiency in the stochastic frontier
production function model.
{it:Journal of Econometrics}, 19(2–3), 233–238.

{phang}
Parmeter, C.F., Simar, L., Van Keilegom, I. and Zelenyuk, V. (2024).
Inference in the nonparametric stochastic frontier model.
{it:Econometric Reviews}, 43(7), 518–539.

{phang}
Simar, L. and Wilson, P.W. (2023).
Nonparametric, stochastic frontier models with multiple inputs and outputs.
{it:Journal of Business & Economic Statistics}, 41(4), 1391–1403.


{title:Author}

{pstd}
Choonjoo Lee{break}
Department of AI and Robotics, Korea National Defense University{break}
Nonsan, Republic of Korea{break}
{browse "mailto:bloom.rampike@gmail.com":bloom.rampike@gmail.com}

{pstd}
Source code: {browse "https://github.com/sw2023-python/sw2023"}


{title:Also see}

{psee}
Online: {helpb sw2023test} (if installed)
{p_end}
