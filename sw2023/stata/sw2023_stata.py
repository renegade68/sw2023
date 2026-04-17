"""
sw2023 Stata Interface (Stata 16.1+)

Executed via the `python script` command available from Stata 16.1.
Data is exchanged through the `sfi` (Stata Function Interface) module.

Usage (Stata do-file):
    * Cross-sectional (2-component)
    local sw_args "x1 x2 x3 | y1 y2 y3 y4 | method=HMS"
    python script "sw2023_stata.py"

    * Panel (4-component)
    local sw_args "x1 x2 x3 | y1 y2 y3 y4 | method=HMS firm=farmid time=year"
    python script "sw2023_stata.py", args("panel")

Output variables (auto-generated):
    Cross-sectional: sw_efficiency, sw_sigma_eta, sw_r1, sw_r3
    Panel:           swp_efficiency, swp_te, swp_pe

Requirements:
    - Stata 16.1 or later
    - Python 3.8+ specified via `python set exec`
    - sw2023 package installed (pip install sw2023 or set path manually)
"""

import sys
import os
import numpy as np

# ── Auto-search for sw2023 package path ──────────────────────
_this_dir = os.path.dirname(os.path.abspath(__file__))
for _candidate in [
    os.path.dirname(_this_dir),          # package root (sw2023/)
    os.path.dirname(os.path.dirname(_this_dir)),  # parent directory
]:
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


# ── Stata 16.1-compatible sfi helpers ────────────────────────
def _sfi_get_data(sfi, varnames):
    """
    Retrieve Stata variables as numpy arrays.
    Uses a method compatible with both Stata 16 and 17.
    """
    if isinstance(varnames, str):
        varnames = [varnames]

    n = sfi.Data.getObsTotal()
    result = np.full((n, len(varnames)), np.nan)

    for j, vname in enumerate(varnames):
        vidx = sfi.Data.varIndex(vname)
        if vidx < 0:
            raise ValueError(f"Stata variable not found: '{vname}'")
        for i in range(n):
            val = sfi.Data.getAt(vidx, i)
            result[i, j] = val if not sfi.Missing.isMissing(val) else np.nan

    return result.squeeze() if len(varnames) == 1 else result


def _sfi_add_var(sfi, varname, label=''):
    """
    Add a new variable to Stata. Reuse if it already exists.
    Stata 16.1 compatible: existence check via varIndex().
    """
    if sfi.Data.varIndex(varname) < 0:
        sfi.Data.addVarDouble(varname)
    # setVarLabel available in Stata 16.1+
    try:
        sfi.Data.setVarLabel(varname, label)
    except AttributeError:
        pass  # Ignore if Stata version is too old


def _sfi_store(sfi, varname, values, label=''):
    """
    Store a numpy array into a Stata variable.
    NaN values are converted to Stata missing values.
    """
    _sfi_add_var(sfi, varname, label)
    vidx = sfi.Data.varIndex(varname)
    n    = sfi.Data.getObsTotal()

    for i in range(min(n, len(values))):
        v = values[i]
        if np.isnan(v):
            sfi.Data.storeAt(vidx, i, sfi.Missing.getValue())
        else:
            sfi.Data.storeAt(vidx, i, float(v))


def _sfi_display(sfi, msg):
    """Print a message to the Stata output window."""
    try:
        sfi.SFIToolkit.displayln(msg)
    except AttributeError:
        # Fallback for Stata 16.1 and below
        try:
            sfi.SFIToolkit.display(msg + '\n')
        except Exception:
            pass


# ── Argument parsing ──────────────────────────────────────────
def _parse_args(args_str):
    """
    Parse format: 'x1 x2 | y1 y2 | key=val key=val'.

    Returns:
        dict: x_vars, y_vars, options
    """
    parts = [p.strip() for p in args_str.split('|')]
    if len(parts) < 2:
        raise ValueError(
            "Invalid argument format. Use '|' to separate X and Y variables.\n"
            "Example: 'x1 x2 | y1 y2 | method=HMS'"
        )

    x_vars = [v for v in parts[0].split() if v]
    y_vars = [v for v in parts[1].split() if v]

    opts = {}
    if len(parts) >= 3:
        for tok in parts[2].split():
            if '=' in tok:
                k, v = tok.split('=', 1)
                opts[k.strip()] = v.strip()

    # Defaults
    opts.setdefault('method',        'HMS')
    opts.setdefault('direction',     'mean')
    opts.setdefault('log_transform', '1')
    opts.setdefault('standardize',   '1')
    opts.setdefault('verbose',       '0')

    # Type conversion
    def _bool(s):
        return str(s).strip() not in ('0', 'False', 'false', 'no', 'off')

    opts['log_transform'] = _bool(opts['log_transform'])
    opts['standardize']   = _bool(opts['standardize'])
    opts['verbose']       = _bool(opts['verbose'])

    return {'x_vars': x_vars, 'y_vars': y_vars, 'options': opts}


# ── Cross-sectional model ──────────────────────────────────────
def run_crosssection(args_str=None, sfi=None):
    """
    Run SW2023Model (2-component).

    Parameters
    ----------
    args_str : str, optional
        'x1 x2 | y1 y2 | method=HMS'
        If None, reads from Stata local macro `sw_args`
    sfi : module, optional
    """
    if sfi is None:
        import sfi as _sfi
        sfi = _sfi

    if args_str is None:
        args_str = sfi.Macro.getLocal('sw_args')

    cfg   = _parse_args(args_str)
    x_vars = cfg['x_vars']
    y_vars = cfg['y_vars']
    opt    = cfg['options']

    _sfi_display(sfi,
        f"\n[SW2023] 2-component model\n"
        f"  Inputs ({len(x_vars)}): {' '.join(x_vars)}\n"
        f"  Outputs ({len(y_vars)}): {' '.join(y_vars)}\n"
        f"  Method: {opt['method']}"
    )

    # Data
    X_raw = _sfi_get_data(sfi, x_vars)
    Y_raw = _sfi_get_data(sfi, y_vars)

    if X_raw.ndim == 1: X_raw = X_raw.reshape(-1, 1)
    if Y_raw.ndim == 1: Y_raw = Y_raw.reshape(-1, 1)

    n_total = X_raw.shape[0]
    mask    = (~np.any(np.isnan(X_raw), axis=1) &
               ~np.any(np.isnan(Y_raw), axis=1))
    n_valid = mask.sum()

    if n_valid < n_total:
        _sfi_display(sfi, f"  Warning: {n_total-n_valid} missing observations excluded (valid: {n_valid})")

    X = X_raw[mask]
    Y = Y_raw[mask]

    # Estimation
    from sw2023.core.model import SW2023Model
    m = SW2023Model(X, Y,
                     direction=opt['direction'],
                     method=opt['method'],
                     log_transform=opt['log_transform'],
                     standardize=opt['standardize'])
    m.fit(verbose=opt['verbose'])

    # Store results
    def _expand(vals):
        out = np.full(n_total, np.nan)
        out[mask] = vals
        return out

    _sfi_store(sfi, 'sw_efficiency', _expand(m.efficiency_),  'SW(2023) efficiency')
    _sfi_store(sfi, 'sw_sigma_eta',  _expand(m.sigma_eta_),   'SW(2023) sigma_eta')
    _sfi_store(sfi, 'sw_r1',         _expand(m.r1_),          'SW(2023) conditional 1st moment')
    _sfi_store(sfi, 'sw_r3',         _expand(m.r3_),          'SW(2023) conditional 3rd moment')

    _sfi_display(sfi,
        f"\n  Mean efficiency: {np.nanmean(m.efficiency_):.4f}\n"
        f"  Created variables: sw_efficiency, sw_sigma_eta, sw_r1, sw_r3\n"
        f"  Done."
    )
    return m


# ── Panel model ───────────────────────────────────────────────
def run_panel(args_str=None, sfi=None):
    """
    Run PanelSW2023 (4-component).

    Specify IDs in options as firm=<variable> and time=<variable>.
    """
    if sfi is None:
        import sfi as _sfi
        sfi = _sfi

    if args_str is None:
        args_str = sfi.Macro.getLocal('sw_args')

    cfg    = _parse_args(args_str)
    x_vars = cfg['x_vars']
    y_vars = cfg['y_vars']
    opt    = cfg['options']

    firm_var = opt.get('firm', 'firmid')
    time_var = opt.get('time', 'year')

    _sfi_display(sfi,
        f"\n[SW2023] 4-component panel model\n"
        f"  Inputs ({len(x_vars)}): {' '.join(x_vars)}\n"
        f"  Outputs ({len(y_vars)}): {' '.join(y_vars)}\n"
        f"  Entity: {firm_var}, Period: {time_var}\n"
        f"  Method: {opt['method']}"
    )

    # Data
    X_raw   = _sfi_get_data(sfi, x_vars)
    Y_raw   = _sfi_get_data(sfi, y_vars)
    firm_id = _sfi_get_data(sfi, firm_var)
    time_id = _sfi_get_data(sfi, time_var)

    if X_raw.ndim == 1: X_raw = X_raw.reshape(-1, 1)
    if Y_raw.ndim == 1: Y_raw = Y_raw.reshape(-1, 1)

    n_total = X_raw.shape[0]
    mask    = (~np.any(np.isnan(X_raw), axis=1) &
               ~np.any(np.isnan(Y_raw), axis=1) &
               ~np.isnan(firm_id) & ~np.isnan(time_id))

    X       = X_raw[mask]
    Y       = Y_raw[mask]
    firm_id = firm_id[mask].astype(int)
    time_id = time_id[mask].astype(int)

    # Estimation
    from sw2023.panel.four_component import PanelSW2023
    m = PanelSW2023(X, Y, firm_id, time_id,
                     direction=opt['direction'],
                     method=opt['method'],
                     log_transform=opt['log_transform'],
                     standardize=opt['standardize'])
    m.fit(verbose=opt['verbose'])

    # Store results
    def _expand(vals):
        out = np.full(n_total, np.nan)
        out[mask] = vals
        return out

    _sfi_store(sfi, 'swp_efficiency', _expand(m.efficiency_),     'SW(2023) overall efficiency')
    _sfi_store(sfi, 'swp_te',         _expand(m.eff_transient_),  'SW(2023) transient efficiency')
    _sfi_store(sfi, 'swp_pe',         _expand(m.eff_persistent_), 'SW(2023) persistent efficiency')

    _sfi_display(sfi,
        f"\n  Overall efficiency:    {np.nanmean(m.efficiency_):.4f}\n"
        f"  Transient efficiency:  {np.nanmean(m.eff_transient_):.4f}\n"
        f"  Persistent efficiency: {np.nanmean(m.eff_persistent_):.4f}\n"
        f"  Created variables: swp_efficiency, swp_te, swp_pe\n"
        f"  Done."
    )
    return m


# ── Entry point ───────────────────────────────────────────────
if __name__ == '__main__':
    try:
        import sfi

        mode = sys.argv[1] if len(sys.argv) > 1 else 'cross'
        if mode == 'panel':
            run_panel(sfi=sfi)
        else:
            run_crosssection(sfi=sfi)

    except ImportError:
        print("Stata sfi module not found.")
        print("This script must be run within a Stata 16.1 or later environment.")
        print("\nStata do-file example:")
        print('  python set exec "/usr/local/bin/python3"')
        print('  local sw_args "x1 x2 | y1 y2 | method=HMS"')
        print('  python script "sw2023_stata.py"')
