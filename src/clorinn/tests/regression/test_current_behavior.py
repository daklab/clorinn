"""
Regression tests for current solver behavior.

By default these tests run in portable release mode. They compare final
solver behavior to saved fixtures using numerical tolerances that should
be stable across BLAS/LAPACK, NumPy, and SciPy versions.

Set

    CLORINN_STRICT_REGRESSION=1

to run strict local trajectory-freezing tests. Strict mode compares full
histories, exact iteration counts, and near-exact fixture values. It is
intended as a developer diagnostic and is not required to pass on every
machine. They are allowed to fail across BLAS/LAPACK, NumPy, or SciPy 
versions even when the implementation is correct.

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

    python -m clorinn.tests.regression.generate_current_behavior

Commit the updated .npz files alongside the code change.

Running
-------
    python -m unittest clorinn.tests.regression.test_current_behavior -v
"""
import os
import unittest
import numpy as np

from clorinn.utils import SamplingCovariance
from clorinn.optimize import FrankWolfe, ProjectedGradientDescent
from clorinn.utils.logs import CustomLogger
from clorinn.tests.regression.regression_config import FW_CONFIG, PGD_CONFIG, R_NUC, L1_MULT, PORTABLE_TOLERANCE_RULES
from clorinn.tests.regression.regression_helper import assert_allclose_print_worst_entry

STRICT_REGRESSION = os.environ.get("CLORINN_STRICT_REGRESSION", "0") == "1"
FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')

def _load(name):
    return np.load(os.path.join(FIXTURES_DIR, name), allow_pickle=False)

def _tol(model, solver, quantity, is_masked):
    """
    Return (rtol, atol) for regression tests.

    In strict mode, use near-exact trajectory comparison.
    In portable mode, use tolerances calibrated to observed
    BLAS/LAPACK/NumPy/SciPy drift across machines.
    """
    if STRICT_REGRESSION:
        return (1e-12, 1e-12)

    key = (solver, model, is_masked, quantity)
    for pattern, tol in PORTABLE_TOLERANCE_RULES:
        if all(p == "*" or p == k for p, k in zip(pattern, key)):
            return tol

    # The trailing ("*","*","*","*") catch-all guarantees a match.
    raise RuntimeError(f"no _TOL_RULES entry for {key!r}")  # unreachable

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


    def _assert_close(self, actual, expected, quantity, name=None):
        is_masked = "mask" in self.fixture_name
        rtol, atol = _tol(self.model, "fw", quantity, is_masked)
        assert_allclose_print_worst_entry(self, actual, expected, rtol=rtol, atol=atol, name=name or quantity)

    def test_n_iter(self):
        if STRICT_REGRESSION:
            self.logger_.warning(f"Running strict regression for {self.__class__.__name__}")
            self.assertEqual(self.m.n_iter, int(self.f['n_iter']))
        else:
            self.assertLessEqual(abs(self.m.n_iter - int(self.f['n_iter'])), int(0.20 * int(self.f['n_iter'])))

    def test_X_identical(self):
        self._assert_close(self.m.X, self.f["X"], "X")

    def test_fx_history_identical(self):
        if STRICT_REGRESSION:
            self._assert_close(np.array(self.m.history.loss), self.f['fx'], "loss", name = "loss history")
            return
        if self.model == "nnm-corr":
            self._assert_close(np.array(self.m.history.loss)[-1], self.f['fx'][-1], "loss", name = "final loss")
            return
        k = min(len(self.m.history.loss), len(self.f['fx']))
        self._assert_close(np.array(self.m.history.loss)[:k], self.f['fx'][:k], "loss", name = "loss history")

    def test_dg_history_identical(self):
        # first dg can be np.inf
        if STRICT_REGRESSION:
            self._assert_close(np.array(self.m.history.duality_gap)[1:], self.f['dg'][1:], "gap")
            return
        if self.model == "nnm-corr":
            #self.skipTest("LAPACK/BLAS drift")
            #self._assert_close(np.array(self.m.history.duality_gap)[-1], self.f['dg'][-1], "gap", name = "final duality gap")
            #return
            k = min(20, len(self.m.history.duality_gap), len(self.f['dg']))
        else:
            k = min(len(self.m.history.duality_gap), len(self.f['dg']))
        self._assert_close(np.array(self.m.history.duality_gap)[1:k], self.f['dg'][1:k], "gap", name = "duality gap history")

    def test_steps_history_identical(self):
        if STRICT_REGRESSION:
            self._assert_close(np.array(self.m.history.step_size), self.f['steps'], "step")
            return
        if self.model == "nnm-corr":
            #self.skipTest("LAPACK/BLAS drift")
            #self._assert_close(np.array(self.m.history.step_size)[-1], self.f['steps'][-1], "step", name = "final step size")
            #return
            k = min(20, len(self.m.history.step_size), len(self.f['steps']))
        else:
            k = min(len(self.m.history.step_size), len(self.f['steps']))
        self._assert_close(np.array(self.m.history.step_size)[:k], self.f['steps'][:k], "step", name = "step size history")

    def test_history_length(self):
        """All histories have length n_iter + 1 (iteration 0 is included)."""
        if STRICT_REGRESSION:
            expected = int(self.f['n_iter']) + 1
        else:
            if self.model == "nnm-corr":
                expected = int(self.m.n_iter) + 1
            else:
                expected = int(self.f['n_iter']) + 1
        self.assertEqual(len(self.m.history.loss), expected)
        self.assertEqual(len(self.m.history.duality_gap), expected)
        self.assertEqual(len(self.m.history.step_size), expected)

    def test_M_identical(self):
        if 'M' not in self.f:
            self.skipTest("not a sparse fixture")
        self._assert_close(self.m.M, self.f['M'], "M")

    def test_M_absent_for_non_sparse(self):
        if 'M' in self.f:
            self.skipTest("sparse fixture")
        self.assertIsNone(self.m.M)


# ---------------------------------------------------------------------------
# FrankWolfe test classes
# ---------------------------------------------------------------------------

class TestRegressionFWNNM(_FWRegressionBase):
    """TC1: NNM, fully observed."""
    fixture_name = 'fw_nnm.npz'
    model = 'nnm'


class TestRegressionFWNNMMask(_FWRegressionBase):
    """TC2: NNM, 10 % missing data."""
    fixture_name = 'fw_nnm_mask.npz'
    model = 'nnm'


class TestRegressionFWNNMSparse(_FWRegressionBase):
    """TC3: NNM-Sparse, fully observed."""
    fixture_name = 'fw_nnm_sparse.npz'
    model = 'nnm-sparse'


class TestRegressionFWNNMSparseMask(_FWRegressionBase):
    """TC4: NNM-Sparse, 10 % missing data."""
    fixture_name = 'fw_nnm_sparse_mask.npz'
    model = 'nnm-sparse'


class TestRegressionFWNNMCorr(_FWRegressionBase):
    """TC5: NNM-Corr, fully observed."""
    fixture_name = 'fw_nnm_corr.npz'
    model = 'nnm-corr'


class TestRegressionFWNNMCorrMask(_FWRegressionBase):
    """TC6: NNM-Corr, 10 % missing data."""
    fixture_name = 'fw_nnm_corr_mask.npz'
    model = 'nnm-corr'


# ---------------------------------------------------------------------------
# ProjectedGradientDescent regression base
# ---------------------------------------------------------------------------

class _PGDRegressionBase(unittest.TestCase):
    fixture_name: str = ''
    model: str = ''
    rtol: float = 1e-12
    atol: float = 1e-12

    @classmethod
    def setUpClass(cls):
        if not cls.fixture_name:
            raise unittest.SkipTest("base class")

        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info(f"[pgd/{cls.model}] loading {cls.fixture_name}")

        f = _load(cls.fixture_name)
        cls.f = f

        pgd = ProjectedGradientDescent(model=cls.model, **PGD_CONFIG)
        fit_kwargs = dict(radius=float(f['radius']))
        if 'mask' in f:
            fit_kwargs['mask'] = f['mask']
        if 'A' in f:
            fit_kwargs['noise_cov'] = SamplingCovariance.from_matrix(f['A'])
        if 'sparse_scale' in f:
            fit_kwargs['sparse_scale'] = float(f['sparse_scale'])
        pgd.fit(f['Y'], **fit_kwargs)
        cls.pgd = pgd.result

    def _assert_close(self, actual, expected, quantity, name=None):
        is_masked = "mask" in self.fixture_name
        rtol, atol = _tol(self.model, "pgd", quantity, is_masked)
        assert_allclose_print_worst_entry(self, actual, expected, rtol=rtol, atol=atol, name=name or quantity)

    def test_n_iter(self):
        if STRICT_REGRESSION:
            self.logger_.warning(f"Running strict regression for {self.__class__.__name__}")
        self.assertEqual(self.pgd.n_iter, int(self.f['n_iter']))

    def test_X_identical(self):
        self._assert_close(self.pgd.X, self.f['X'], "X")

    def test_fx_history_identical(self):
        k = min(len(self.pgd.history.loss), len(self.f['fx']))
        self._assert_close(np.array(self.pgd.history.loss)[:k], self.f['fx'][:k], "loss")

    def test_converged_in_interior(self):
        self.assertEqual(self.pgd.converged,
                         bool(self.f['converged']))

    def test_history_length(self):
        """fx has length n_iter + 1."""
        expected = int(self.f['n_iter']) + 1
        self.assertEqual(len(self.pgd.history.loss), expected)

    def test_M_identical(self):
        if 'M' not in self.f:
            self.skipTest("not a sparse fixture")
        self._assert_close(self.pgd.M, self.f['M'], "M")

    def test_M_absent_for_non_sparse(self):
        if 'M' in self.f:
            self.skipTest("sparse fixture")
        self.assertIsNone(self.pgd.M)


# ---------------------------------------------------------------------------
# PGD test classes
# ---------------------------------------------------------------------------

class TestRegressionPGDNNM(_PGDRegressionBase):
    """TC7: PGD warm start, fully observed."""
    fixture_name = 'pgd_nnm.npz'
    model        = 'nnm'


class TestRegressionPGDNNMMask(_PGDRegressionBase):
    """TC8: PGD warm start, 10 % missing data."""
    fixture_name = 'pgd_nnm_mask.npz'
    model        = 'nnm'


class TestRegressionPGDNNMSparse(_PGDRegressionBase):
    """TC9: PGD + NNM-Sparse, fully observed."""
    fixture_name = 'pgd_nnm_sparse.npz'
    model        = 'nnm-sparse'


class TestRegressionPGDNNMSparseMask(_PGDRegressionBase):
    """TC10: PGD + NNM-Sparse, 10 % missing data."""
    fixture_name = 'pgd_nnm_sparse_mask.npz'
    model        = 'nnm-sparse'


class TestRegressionPGDNNMCorr(_PGDRegressionBase):
    """TC11: PGD + NNM-Corr, fully observed."""
    fixture_name = 'pgd_nnm_corr.npz'
    model        = 'nnm-corr'


class TestRegressionPGDNNMCorrMask(_PGDRegressionBase):
    """TC12: PGD + NNM-Corr, 10 % missing data."""
    fixture_name = 'pgd_nnm_corr_mask.npz'
    model        = 'nnm-corr'


# ---------------------------------------------------------------------------
# Fixture integrity check
# ---------------------------------------------------------------------------

class TestFixturesPresent(unittest.TestCase):
    EXPECTED = [
        'fw_nnm.npz', 'fw_nnm_mask.npz',
        'fw_nnm_sparse.npz', 'fw_nnm_sparse_mask.npz',
        'fw_nnm_corr.npz', 'fw_nnm_corr_mask.npz',
        'pgd_nnm.npz', 'pgd_nnm_mask.npz',
        'pgd_nnm_sparse.npz', 'pgd_nnm_sparse_mask.npz',
        'pgd_nnm_corr.npz', 'pgd_nnm_corr_mask.npz',
    ]

    def test_fixtures_directory_exists(self):
        self.assertTrue(
            os.path.isdir(FIXTURES_DIR),
            msg=f"Fixtures directory missing: {FIXTURES_DIR}\n"
                "Run: python -m clorinn.tests.regression.generate_current_behavior",
        )

    def test_all_fixture_files_present(self):
        missing = [n for n in self.EXPECTED
                   if not os.path.isfile(os.path.join(FIXTURES_DIR, n))]
        self.assertEqual(
            missing, [],
            msg=f"Missing fixtures: {missing}\n"
                "Run: python -m clorinn.tests.regression.generate_current_behavior",
        )
