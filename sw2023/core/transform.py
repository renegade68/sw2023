"""
SW (2023) Steps 1-3: Rotation Transformation

Simar & Wilson (2023), JBES 41(4), 1391-1403.
Eq. (2.10)-(2.17): Transform (X,Y) → (Z,U) using direction vector d
"""

import numpy as np
from scipy.linalg import null_space


def make_direction(X, Y, method='mean'):
    """
    Generate direction vector d.
    SW(2023) recommendation: negative direction for inputs, positive for outputs.

    Parameters
    ----------
    X : (n, p) inputs
    Y : (n, q) outputs
    method : 'mean' | 'median' | array-like

    Returns
    -------
    d : (p+q,) normalized direction vector
    """
    if isinstance(method, (np.ndarray, list)):
        d = np.asarray(method, dtype=float)
    elif method == 'mean':
        d = np.concatenate([-np.mean(X, axis=0), np.mean(Y, axis=0)])
    elif method == 'median':
        d = np.concatenate([-np.median(X, axis=0), np.median(Y, axis=0)])
    else:
        raise ValueError(f"method must be 'mean', 'median', or array. Got {method}")

    norm = np.linalg.norm(d)
    if norm == 0:
        raise ValueError("The norm of the direction vector is 0.")
    return d / norm


def rotation_matrix(d):
    """
    Compute rotation matrix R_d. SW(2023) Eq. (2.10).

    R_d = [V_d'; d'] orthogonal matrix
    - V_d: (r x r-1) basis matrix orthogonal to d
    - Last row = d (normalized)

    Parameters
    ----------
    d : (r,) normalized direction vector

    Returns
    -------
    R : (r, r) orthogonal rotation matrix
    """
    d = d / np.linalg.norm(d)
    V = null_space(d.reshape(1, -1))   # (r, r-1)
    R = np.vstack([V.T, d.reshape(1, -1)])  # (r, r)
    return R


def transform(X, Y, d):
    """
    Rotation transform (X, Y) → (Z, U). SW(2023) Eq. (2.15).

    W_i = (X_i, Y_i) ∈ R^r
    [Z_i; U_i] = R_d @ W_i

    where U_i = d' @ W_i / ||d|| is the scalar distance along direction d (dependent variable)
    Z_i ∈ R^(r-1) is the remaining coordinates (independent variables)

    Parameters
    ----------
    X : (n, p) inputs
    Y : (n, q) outputs
    d : (p+q,) normalized direction vector

    Returns
    -------
    Z : (n, r-1) independent variables after rotation
    U : (n,)    dependent variable after rotation
    R : (r, r)  rotation matrix (for inverse transform)
    """
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    W = np.hstack([X, Y])              # (n, r)
    R = rotation_matrix(d)             # (r, r)
    WR = W @ R.T                       # (n, r)
    Z = WR[:, :-1]                     # (n, r-1)
    U = WR[:, -1]                      # (n,)
    return Z, U, R


def inverse_transform(Z, U, R):
    """
    Inverse transform (Z, U) → (X, Y).

    Parameters
    ----------
    Z : (n, r-1)
    U : (n,)
    R : (r, r) rotation matrix

    Returns
    -------
    W : (n, r) = recovered (X, Y)
    """
    ZU = np.column_stack([Z, U])       # (n, r)
    W = ZU @ R                         # (n, r)
    return W
