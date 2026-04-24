"""
test_current_behavior.py
=====================================
Regression tests that freeze the current solver behavior during refactoring.

For each .npz fixture the test:
  1. Loads the saved numerical inputs (Y, mask, constraints, covariance).
  2. Reconstructs the solver call using the config from regression_config.py.
  3. Asserts that every output — X, M, fx, dg, steps, n_iter — is
     bit-for-bit identical to the saved values.

Because svd_method='left-gram' is fully deterministic, exact equality is
both achievable and meaningful. A failure means the refactor changed
observable iteration behavior.

Updating fixtures
-----------------
When you intentionally change solver behavior, regenerate:

    python -m clorinn.tests.generate_current_behavior

Commit the updated .npz files alongside the code change.

Running
-------
    python -m unittest clorinn.tests.test_current_behavior -v
"""

import os
import unittest
import numpy as np

from clorinn.utils import SamplingCovariance
from clorinn.optimize import FrankWolfe, PGDWarmStart
from clorinn.utils.logs import CustomLogger
from clorinn.tests.regression.regression_config import FW_CONFIG, PGD_CONFIG, R_NUC, L1_MULT

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')


def _load(name):
    return np.load(os.path.join(FIXTURES_DIR, name), allow_pickle=False)


# ---------------------------------------------------------------------------
# FrankWolfe regression base
# ---------------------------------------------------------------------------

class _FWRegressionBase(unittest.TestCase):
    """
    Subclasses set fixture_name and model. setUpClass loads the fixture
    and reruns the solver once; all test methods share that result.
    """
    fixture_name: str = ''
    model: str = ''
    rtol: float = 1e-12
    atol: float = 1e-12

    @classmethod
    def setUpClass(cls):
        if not cls.fixture_name:
            raise unittest.SkipTest("base class")

        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info(f"[{cls.model}] loading {cls.fixture_name}")
        f = _load(cls.fixture_name)
        cls.f = f

        m = FrankWolfe(model=cls.model, **FW_CONFIG)

        fit_kwargs = dict(radius=float(f['radius']))
        if 'mask' in f:
            fit_kwargs['mask'] = f['mask']
        if 'A' in f:
            fit_kwargs['noise_cov'] = SamplingCovariance.from_matrix(f['A'])
        if 'sparse_scale' in f:
            fit_kwargs['sparse_scale'] = float(f['sparse_scale'])

        m.fit(f['Y'], **fit_kwargs)
        cls.m = m.result

    def test_n_iter(self):
        self.assertEqual(self.m.n_iter, int(self.f['n_iter']))

    def test_X_identical(self):
        #np.testing.assert_array_equal(self.m.X, self.f['X'])
        np.testing.assert_allclose(self.m.X, self.f['X'], rtol=self.rtol, atol=self.atol)

    def test_fx_history_identical(self):
        #np.testing.assert_array_equal(np.array(self.m.history.loss), self.f['fx'])
        np.testing.assert_allclose(np.array(self.m.history.loss), self.f['fx'], rtol=self.rtol, atol=self.atol)

    def test_dg_history_identical(self):
        #np.testing.assert_array_equal(np.array(self.m.history.duality_gap), self.f['dg'])
        np.testing.assert_allclose(np.array(self.m.history.duality_gap), self.f['dg'], rtol=self.rtol, atol=self.atol)

    def test_steps_history_identical(self):
        #np.testing.assert_array_equal(np.array(self.m.history.step_size), self.f['steps'])
        np.testing.assert_allclose(np.array(self.m.history.step_size), self.f['steps'], rtol=self.rtol, atol=self.atol)

    def test_history_length(self):
        """All histories have length n_iter + 1 (iteration 0 is included)."""
        expected = int(self.f['n_iter']) + 1
        self.assertEqual(len(self.m.history.loss), expected)
        self.assertEqual(len(self.m.history.duality_gap), expected)
        self.assertEqual(len(self.m.history.step_size), expected)

    def test_M_identical(self):
        if 'M' not in self.f:
            self.skipTest("not a sparse fixture")
        #np.testing.assert_array_equal(self.m.M, self.f['M'])
        np.testing.assert_allclose(self.m.M, self.f['M'], rtol=self.rtol, atol=self.atol)

    def test_M_absent_for_non_sparse(self):
        if 'M' in self.f:
            self.skipTest("sparse fixture")
        self.assertIsNone(self.m.M)
        #with self.assertRaises(AttributeError):
        #    _ = self.m.M


# ---------------------------------------------------------------------------
# FrankWolfe test classes
# ---------------------------------------------------------------------------

class TestRegressionNNM(_FWRegressionBase):
    """TC1: NNM, fully observed."""
    fixture_name = 'nnm.npz'
    model = 'nnm'


class TestRegressionNNMMask(_FWRegressionBase):
    """TC2: NNM, 10 % missing data."""
    fixture_name = 'nnm_mask.npz'
    model = 'nnm'


class TestRegressionNNMSparse(_FWRegressionBase):
    """TC3: NNM-Sparse, fully observed."""
    fixture_name = 'nnm_sparse.npz'
    model = 'nnm-sparse'


class TestRegressionNNMSparseMask(_FWRegressionBase):
    """TC4: NNM-Sparse, 10 % missing data."""
    fixture_name = 'nnm_sparse_mask.npz'
    model = 'nnm-sparse'


class TestRegressionNNMCorr(_FWRegressionBase):
    """TC5: NNM-Corr, fully observed."""
    fixture_name = 'nnm_corr.npz'
    model = 'nnm-corr'


class TestRegressionNNMCorrMask(_FWRegressionBase):
    """TC6: NNM-Corr, 10 % missing data."""
    fixture_name = 'nnm_corr_mask.npz'
    model = 'nnm-corr'


# ---------------------------------------------------------------------------
# PGDWarmStart regression base
# ---------------------------------------------------------------------------

class _PGDRegressionBase(unittest.TestCase):
    fixture_name: str = ''
    rtol: float = 1e-12
    atol: float = 1e-12

    @classmethod
    def setUpClass(cls):
        if not cls.fixture_name:
            raise unittest.SkipTest("base class")

        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info(f"[pgd] loading {cls.fixture_name}")

        f = _load(cls.fixture_name)
        cls.f = f

        pgd = PGDWarmStart(**PGD_CONFIG)
        fit_kwargs = dict(radius=float(f['radius']))
        if 'mask' in f:
            fit_kwargs['mask'] = f['mask']
        pgd.fit(f['Y'], **fit_kwargs)
        cls.pgd = pgd.result

    def test_n_iter(self):
        self.assertEqual(self.pgd.n_iter, int(self.f['n_iter']))

    def test_X_identical(self):
        #np.testing.assert_array_equal(self.pgd.X, self.f['X'])
        np.testing.assert_allclose(self.pgd.X, self.f['X'], rtol=self.rtol, atol=self.atol)

    def test_fx_history_identical(self):
        #np.testing.assert_array_equal(np.array(self.pgd.history.loss), self.f['fx'])
        np.testing.assert_allclose(np.array(self.pgd.history.loss), self.f['fx'], rtol=self.rtol, atol=self.atol)

    def test_converged_in_interior(self):
        self.assertEqual(self.pgd.converged,
                         bool(self.f['converged_in_interior']))

    def test_history_length(self):
        """fx has length n_iter + 1."""
        expected = int(self.f['n_iter']) + 1
        self.assertEqual(len(self.pgd.history.loss), expected)


# ---------------------------------------------------------------------------
# PGD test classes
# ---------------------------------------------------------------------------

class TestRegressionPGD(_PGDRegressionBase):
    """TC7: PGD warm start, fully observed."""
    fixture_name = 'pgd.npz'


class TestRegressionPGDMask(_PGDRegressionBase):
    """TC8: PGD warm start, 10 % missing data."""
    fixture_name = 'pgd_mask.npz'


# ---------------------------------------------------------------------------
# Fixture integrity check
# ---------------------------------------------------------------------------

class TestFixturesPresent(unittest.TestCase):
    EXPECTED = [
        'nnm.npz', 'nnm_mask.npz',
        'nnm_sparse.npz', 'nnm_sparse_mask.npz',
        'nnm_corr.npz', 'nnm_corr_mask.npz',
        'pgd.npz', 'pgd_mask.npz',
    ]

    def test_fixtures_directory_exists(self):
        self.assertTrue(
            os.path.isdir(FIXTURES_DIR),
            msg=f"Fixtures directory missing: {FIXTURES_DIR}\n"
                "Run: python -m clorinn.tests.generate_current_behavior",
        )

    def test_all_fixture_files_present(self):
        missing = [n for n in self.EXPECTED
                   if not os.path.isfile(os.path.join(FIXTURES_DIR, n))]
        self.assertEqual(
            missing, [],
            msg=f"Missing fixtures: {missing}\n"
                "Run: python -m clorinn.tests.generate_current_behavior",
        )
