"""
SW (2023) Step 7-9: Inefficiency Decomposition and Efficiency Index

SVKZ estimator (Simar, Van Keilegom, Zelenyuk):
  sigma_hat_eta(z) = max{0, cbrt(-r_hat_3(z) / a3+)}

HMS estimator (Hafner, Manner, Simar):
  if r_hat_3(z) <= 0: sigma_hat_eta(z) = cbrt(-r_hat_3(z) / a3+)
  if r_hat_3(z) > 0:  sigma_hat_eta(z) = cbrt( r_hat_3(z) / a3-)  <- wrong skewness handling

Constants (based on half-normal distribution):
  a3+ = (4-pi)/pi × sqrt(2/pi) ≈ 0.2177   <- when r3 <= 0 (third central moment)
  a3- ≈ 0.016741474                         <- when r3 > 0 (provided by HMS paper)

Note: When using LOO-CV bandwidth, a large bandwidth is selected for r3 estimation,
      which lowers the wrong-skewness rate, so a3- is rarely used.
      When rho=0 (sigma_eps=0), results match Table F.1.
"""

import numpy as np
from scipy.stats import norm as scipy_norm

PI = np.pi
A3_PLUS  = (4 - PI) / PI * np.sqrt(2 / PI)   # ≈ 0.2177
A3_MINUS = 0.016741474                          # value provided by HMS paper


def estimate_sigma_eta(r3, method='HMS'):
    """
    Estimate sigma_hat_eta(z) from third conditional moment r_hat_3(z).

    Parameters
    ----------
    r3     : (n,) r_hat_3(Z_i) values
    method : 'SVKZ' | 'HMS'

    Returns
    -------
    sigma_eta : (n,) >= 0
    """
    r3 = np.asarray(r3, dtype=float)

    if method == 'SVKZ':
        # max{0, cbrt(-r3 / a3+)}
        # r3 <= 0 → -r3 >= 0 → cbrt >= 0
        # r3 > 0  → -r3 < 0  → cbrt < 0 → max → 0
        val = -r3 / A3_PLUS
        sigma_eta = np.maximum(0.0, np.cbrt(val))

    elif method == 'HMS':
        # Apply alternative formula for wrong skewness (r3 > 0)
        val_neg = np.cbrt(-r3 / A3_PLUS)    # for r3 <= 0
        val_pos = np.cbrt( r3 / A3_MINUS)   # for r3 > 0 (HMS)
        sigma_eta = np.where(r3 <= 0, val_neg, val_pos)
        sigma_eta = np.maximum(0.0, sigma_eta)

    else:
        raise ValueError(f"method must be 'SVKZ' or 'HMS'. Got: {method}")

    return sigma_eta


def estimate_sigma_eps(r2, sigma_eta):
    """
    Estimate sigma_hat_eps(z) from second conditional moment. SW(2023) Eq. (3.10).

    r_2(z) = sigma_eps^2(z) + (pi-2)/pi × sigma_eta^2(z)
    → sigma_eps^2(z) = r_2(z) - (pi-2)/pi × sigma_eta^2(z)

    Parameters
    ----------
    r2        : (n,)
    sigma_eta : (n,)

    Returns
    -------
    sigma_eps : (n,) >= 0
    """
    sigma_eps_sq = r2 - (PI - 2) / PI * sigma_eta ** 2
    return np.sqrt(np.maximum(0.0, sigma_eps_sq))


def estimate_frontier(r1, sigma_eta, norm_d=1.0):
    """
    Estimate frontier function phi_hat(z). SW(2023) Eq. (3.9).

    r_hat_1(z) = phi(z) - ||d|| × mu_eta(z)
    mu_eta(z) = sqrt(2/pi) × sigma_eta(z)

    → phi_hat(z) = r_hat_1(z) + ||d|| × mu_eta(z)

    Parameters
    ----------
    r1        : (n,) conditional mean estimates
    sigma_eta : (n,) sigma_hat_eta
    norm_d    : float ||d|| (1.0 if normalized)

    Returns
    -------
    phi_hat : (n,) frontier estimates
    mu_eta  : (n,) conditional mean inefficiency
    """
    mu_eta = np.sqrt(2 / PI) * sigma_eta
    phi_hat = r1 + norm_d * mu_eta
    return phi_hat, mu_eta


def jlms_efficiency(U, phi_hat, sigma_eta, sigma_eps):
    """
    Estimate individual efficiency index (JLMS approach). SW(2023) Sec 3.1.

    xi_hat_i = U_i - phi_hat(Z_i) ≈ ||d||(eps_i - eta_i)

    Conditional expectations:
      mu*_i  = -xi_hat_i × sigma_eta^2 / (sigma_eps^2 + sigma_eta^2)
      sigma*_i = sigma_eps × sigma_eta / sqrt(sigma_eps^2 + sigma_eta^2)
      E_hat[eta_i | xi_hat_i] = mu*_i + sigma*_i × phi(mu*_i/sigma*_i) / Phi(mu*_i/sigma*_i)

    BC efficiency: exp(-E_hat[eta|xi_hat])

    Parameters
    ----------
    U         : (n,)
    phi_hat   : (n,)
    sigma_eta : (n,)
    sigma_eps : (n,)

    Returns
    -------
    eff_bc   : (n,) Battese-Coelli efficiency index in (0,1]
    eta_hat  : (n,) inefficiency estimate E[eta|xi_hat]
    """
    xi_hat = U - phi_hat                               # composite residual

    sigma_eta_sq = sigma_eta ** 2
    sigma_eps_sq = sigma_eps ** 2
    sigma_sq = sigma_eta_sq + sigma_eps_sq

    # Handle cases where denominator is zero
    safe_sq = np.where(sigma_sq > 0, sigma_sq, np.nan)

    mu_star    = -xi_hat * sigma_eta_sq / safe_sq
    sigma_star = np.sqrt(sigma_eps_sq * sigma_eta_sq / safe_sq)

    # Compute Phi(·), phi(·) (if sigma_star=0, treat ratio=0)
    safe_star = np.where(sigma_star > 0, sigma_star, 1.0)
    ratio = np.where(sigma_star > 0, mu_star / safe_star, 0.0)
    pdf_r = scipy_norm.pdf(ratio)
    cdf_r = scipy_norm.cdf(ratio)

    # Handle cases where denominator is near zero
    safe_cdf = np.where(cdf_r > 1e-15, cdf_r, 1e-15)

    eta_hat = mu_star + sigma_star * pdf_r / safe_cdf
    eta_hat = np.maximum(0.0, eta_hat)                 # inefficiency >= 0

    eff_bc = np.exp(-eta_hat)

    return eff_bc, eta_hat
