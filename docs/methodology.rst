Methodology
===========

The SW(2023) Estimator
-----------------------

The estimator of Simar & Wilson (2023) extends nonparametric stochastic
frontier analysis to technologies with **multiple outputs**.  Given
:math:`n` observations :math:`(X_i, Y_i)` where
:math:`X_i \in \mathbb{R}^p` are inputs and
:math:`Y_i \in \mathbb{R}^q` are outputs, the model is:

.. math::

   U_i = \varphi(Z_i) + \|d\| \varepsilon_i, \quad
   \varepsilon_i = v_i - \eta_i

where :math:`v_i \sim N(0, \sigma_\varepsilon^2(Z_i))` is noise and
:math:`\eta_i \sim N^+(0, \sigma_\eta^2(Z_i))` is inefficiency, both
potentially heterogeneous in :math:`Z_i`.

Rotation Transform
------------------

A direction vector :math:`d \in \mathbb{R}^{p+q}` is chosen (typically
the sample mean or median of :math:`(X, Y)`).  The data are rotated to
scalar coordinates:

.. math::

   U_i = d^\top (X_i, Y_i) / \|d\|^2,
   \quad
   Z_i = \text{projection of } (X_i,Y_i) \text{ orthogonal to } d.

Conditional Moment Estimation (LLLS)
--------------------------------------

Three conditional moments of :math:`U` given :math:`Z` are estimated by
**local linear least squares (LLLS)**:

.. math::

   r_1(z) &= E[U \mid Z=z] \\
   r_2(z) &= E[\varepsilon^2 \mid Z=z] \\
   r_3(z) &= E[\varepsilon^3 \mid Z=z]

where :math:`\varepsilon_i = U_i - \hat{r}_1(Z_i)`.  Bandwidth is
selected by leave-one-out cross-validation (LOO-CV) or Silverman's rule.

Inefficiency Estimation
-----------------------

Two estimators for :math:`\sigma_\eta(z)` are implemented:

**SVKZ** (Simar, Van Keilegom & Zelenyuk):

.. math::

   \hat\sigma_\eta(z) = \left(\frac{-\hat{r}_3(z)}{a_3^+}\right)^{1/3}

**HMS** (Hafner, Manner & Simar) — handles *wrong skewness*
(:math:`\hat{r}_3 > 0`) by setting :math:`\hat\sigma_\eta = 0` at those
observations:

.. math::

   \hat\sigma_\eta(z) =
   \begin{cases}
     \left(\dfrac{-\hat{r}_3(z)}{a_3^+}\right)^{1/3} & \hat{r}_3(z) \le 0 \\
     0 & \hat{r}_3(z) > 0
   \end{cases}

where :math:`a_3^+ = -\sqrt{2/\pi}(1 - 4/\pi)`.

JLMS Efficiency
---------------

Individual efficiency scores are obtained via the JLMS formula:

.. math::

   \hat E[\eta_i \mid \hat\varepsilon_i] =
   \mu_i^* + \sigma_i^* \frac{\phi(\mu_i^*/\sigma_i^*)}{\Phi(\mu_i^*/\sigma_i^*)}

giving efficiency index :math:`\exp(-\hat\eta_i) \in (0,1]`.

4-Component Panel Extension
----------------------------

For panel data with :math:`T` time periods per firm, the model decomposes:

.. math::

   U_{it} = \varphi(Z_{it}) + \|d\| v_{it} - \|d\| u_{it}
            + \|d\| \alpha_i - \|d\| \mu_i

- :math:`v_{it} \sim N(0,\sigma_v^2)` — transient noise
- :math:`u_{it} \sim N^+(0,\sigma_u^2)` — transient inefficiency
- :math:`\alpha_i \sim N(0,\sigma_\alpha^2)` — individual heterogeneity
- :math:`\mu_i \sim N^+(0,\sigma_\mu^2)` — persistent inefficiency

Identification follows Colombi et al. (2014): within-individual variation
identifies transient components, between-individual variation identifies
persistent components.

References
----------

- Simar, L. & Wilson, P.W. (2023). Nonparametric, Stochastic Frontier Models
  with Multiple Inputs and Outputs. *JBES*, 41(4), 1391–1403.
- Hafner, C.M., Manner, H. & Simar, L. (2018). The "wrong skewness" problem
  in stochastic frontier models. *Econometric Reviews*, 37(4), 380–400.
- Colombi, R., Kumbhakar, S.C., Martini, G. & Vittadini, G. (2014).
  Closed-skew normality in stochastic frontiers with individual effects and
  long/short-run efficiency. *Journal of Productivity Analysis*, 42, 123–136.
