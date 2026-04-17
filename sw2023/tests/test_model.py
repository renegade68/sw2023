"""
Basic unit tests (pytest)

Run:
    pytest sw2023/tests/test_model.py -v
"""

import numpy as np
import pandas as pd
import pytest
from sw2023 import SW2023Model, PanelSW2023


# ── Common fixtures ──────────────────────────────────────────
@pytest.fixture
def simple_data():
    rng = np.random.default_rng(0)
    n = 100
    X = np.exp(rng.normal(1.0, 0.5, size=(n, 2)))
    Y = np.exp(rng.normal(1.0, 0.5, size=(n, 2)))
    return X, Y


@pytest.fixture
def panel_data():
    rng = np.random.default_rng(0)
    N, T = 30, 5
    n = N * T
    X = np.exp(rng.normal(1.0, 0.5, size=(n, 2)))
    Y = np.exp(rng.normal(1.0, 0.5, size=(n, 2)))
    firm_id = np.repeat(np.arange(N), T)
    time_id = np.tile(np.arange(T), N)
    return X, Y, firm_id, time_id


# ── SW2023Model tests ─────────────────────────────────────────
class TestSW2023Model:

    def test_fit_runs(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        assert m._fitted

    def test_efficiency_range(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        eff = m.efficiency_
        valid = eff[~np.isnan(eff)]
        assert (valid >= 0).all() and (valid <= 1).all(), \
            f"Efficiency out of [0,1] range: min={valid.min():.4f}, max={valid.max():.4f}"

    def test_efficiency_length(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        assert len(m.efficiency_) == len(X)

    def test_sigma_eta_nonneg(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        assert (m.sigma_eta_ >= 0).all()

    def test_single_input_output(self):
        rng = np.random.default_rng(42)
        n = 80
        X = np.exp(rng.normal(1.0, 0.5, size=(n, 1)))
        Y = np.exp(rng.normal(1.0, 0.5, size=(n, 1)))
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        assert m._fitted

    def test_method_svkz(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y, method='SVKZ')
        m.fit(verbose=False)
        assert m._fitted

    def test_no_log_transform(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y, log_transform=False, standardize=False)
        m.fit(verbose=False)
        assert m._fitted

    def test_results_dataframe(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        df = m.results_dataframe()
        assert 'efficiency' in df.columns
        assert len(df) == len(X)

    def test_attributes_exist(self, simple_data):
        X, Y = simple_data
        m = SW2023Model(X, Y)
        m.fit(verbose=False)
        for attr in ['Z_', 'U_', 'r1_', 'r2_', 'r3_',
                      'sigma_eta_', 'sigma_eps_', 'phi_hat_',
                      'efficiency_', 'eta_hat_', 'd_', 'h_']:
            assert hasattr(m, attr), f"Attribute missing: {attr}"


# ── PanelSW2023 tests ─────────────────────────────────────────
class TestPanelSW2023:

    def test_fit_runs(self, panel_data):
        X, Y, firm_id, time_id = panel_data
        m = PanelSW2023(X, Y, firm_id, time_id)
        m.fit(verbose=False)
        assert m._fitted

    def test_efficiency_range(self, panel_data):
        X, Y, firm_id, time_id = panel_data
        m = PanelSW2023(X, Y, firm_id, time_id)
        m.fit(verbose=False)
        for attr in ['efficiency_', 'eff_transient_', 'eff_persistent_']:
            eff = getattr(m, attr)
            valid = eff[~np.isnan(eff)]
            assert (valid >= 0).all() and (valid <= 1).all(), \
                f"{attr} range error"

    def test_efficiency_decomp(self, panel_data):
        """Overall efficiency ≤ min(TE, PE) (mathematical property)."""
        X, Y, firm_id, time_id = panel_data
        m = PanelSW2023(X, Y, firm_id, time_id)
        m.fit(verbose=False)
        oe  = m.efficiency_
        te  = m.eff_transient_
        pe  = m.eff_persistent_
        valid = ~(np.isnan(oe) | np.isnan(te) | np.isnan(pe))
        # OE = TE × PE (defined as product)
        np.testing.assert_allclose(oe[valid], (te * pe)[valid], rtol=1e-5)

    def test_summary_runs(self, panel_data):
        X, Y, firm_id, time_id = panel_data
        m = PanelSW2023(X, Y, firm_id, time_id)
        m.fit(verbose=False)
        result = m.summary()
        assert isinstance(result, pd.DataFrame)
        assert 'efficiency' in result.columns
        assert 'eff_transient' in result.columns
        assert 'eff_persistent' in result.columns
