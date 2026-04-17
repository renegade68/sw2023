"""
Data Preprocessing: Log Transform + Standardization

SW(2023) Sec 4: "We standardize each of our variables by dividing by
their standard deviations (this amounts to merely changing units)."
"""

import numpy as np


def preprocess(X, Y, log_transform=True, standardize=True):
    """
    Preprocess inputs and outputs.

    Parameters
    ----------
    X            : (n, p)  inputs
    Y            : (n, q)  outputs
    log_transform: bool    whether to apply log transform (recommended for production function analysis)
    standardize  : bool    whether to standardize (removes differences in variable units)

    Returns
    -------
    X_out, Y_out  : preprocessed arrays
    scaler_info   : dict containing information needed for inverse transform
    """
    X = np.asarray(X, dtype=float).copy()
    Y = np.asarray(Y, dtype=float).copy()

    info = {
        'log_transform': log_transform,
        'standardize'  : standardize,
        'X_log_shift'  : None,
        'Y_log_shift'  : None,
        'X_std'        : None,
        'Y_std'        : None,
    }

    # Log transform (handle zero or negative values)
    if log_transform:
        X_shift = np.where(X <= 0, -X.min(axis=0) + 1e-6, 0)
        Y_shift = np.where(Y <= 0, -Y.min(axis=0) + 1e-6, 0)
        info['X_log_shift'] = X_shift.max(axis=0)
        info['Y_log_shift'] = Y_shift.max(axis=0)

        X = np.log(X + info['X_log_shift'])
        Y = np.log(Y + info['Y_log_shift'])

    # Standardize (divide by standard deviation)
    if standardize:
        info['X_std'] = np.std(X, axis=0, ddof=1)
        info['Y_std'] = np.std(Y, axis=0, ddof=1)
        info['X_std'] = np.where(info['X_std'] == 0, 1.0, info['X_std'])
        info['Y_std'] = np.where(info['Y_std'] == 0, 1.0, info['Y_std'])

        X = X / info['X_std']
        Y = Y / info['Y_std']

    return X, Y, info


def preprocess_apply(X, Y, info):
    """
    Apply the same preprocessing to new data using existing scaler_info.

    Used in bootstrap to reuse the transformation parameters from the original sample.

    Parameters
    ----------
    X, Y  : raw data
    info  : scaler_info dict returned by preprocess()

    Returns
    -------
    X_out, Y_out : preprocessed arrays
    """
    X = np.asarray(X, dtype=float).copy()
    Y = np.asarray(Y, dtype=float).copy()

    if info['log_transform']:
        X = np.log(X + info['X_log_shift'])
        Y = np.log(Y + info['Y_log_shift'])

    if info['standardize']:
        X = X / info['X_std']
        Y = Y / info['Y_std']

    return X, Y
