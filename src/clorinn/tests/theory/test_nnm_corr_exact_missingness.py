"""
test_nnm_corr_exact_missingness.py
-----------------------------------
Correctness tests for the exact correlated-missingness formulation in
NNMCorrObjective.

These tests verify some mathematical properties of the per-pattern
submatrix Mahalanobis loss:

    1. value() matches a manual per-pattern sum
    2. An all-missing column contributes zero to the loss and does not crash
    3. gradient() matches finite differences of value()
    4. step_denom() matches a manual per-pattern sum 

All tests use a tiny fixed matrix (n=4 traits, p=5 SNPs) with known
covariance A and a hand-crafted mask that produces three distinct
missingness patterns plus one all-missing column.

Missingness layout (True = missing):

    SNP:      0     1     2     3     4
    Trait 0:  obs   obs   obs   mis   mis
    Trait 1:  obs   obs   mis   obs   mis
    Trait 2:  obs   obs   obs   mis   mis
    Trait 3:  obs   obs   mis   obs   mis

    Pattern A: {0,1,2,3}  -> SNPs [0, 1]   (fully observed)
    Pattern B: {0,2}      -> SNP  [2]       (traits 1,3 missing)
    Pattern C: {1,3}      -> SNP  [3]       (traits 0,2 missing)
    Pattern D: {}         -> SNP  [4]       (all missing)

"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import logging
import unittest
import numpy as np

from clorinn.optimize.objectives import NNMCorrObjective
from clorinn.utils.logs import CustomLogger
from clorinn.utils import SamplingCovariance


# ---------------------------------------------------------------------------
# Shared problem fixture
# ---------------------------------------------------------------------------

def _make_problem():
    """
    Build a tiny fixed problem with known missingness patterns.

    Returns
    -------
    dict with keys: Y, X, Z, A, noise_cov, mask, radius
    """
    rng = np.random.default_rng(42)
    n, p = 4, 5

    # PD covariance A
    B = rng.standard_normal((n, n))
    A = B @ B.T / n + np.eye(n) * 0.8
    noise_cov = SamplingCovariance.from_matrix(A)

    # Z-score matrix and iterate X
    Y = rng.standard_normal((n, p))
    X = rng.standard_normal((n, p)) * 0.3

    # Mask: True = missing
    # Layout documented in module docstring
    mask = np.zeros((n, p), dtype=bool)
    # SNP 2: traits 1,3 missing
    mask[[1, 3], 2] = True
    # SNP 3: traits 0,2 missing
    mask[[0, 2], 3] = True
    # SNP 4: all traits missing
    mask[:, 4] = True

    return dict(
        Y         = Y,
        X         = X,
        A         = A,
        noise_cov = noise_cov,
        mask      = mask,
        radius    = 10.0,
    )


# ---------------------------------------------------------------------------
# Helpers for manual reference computations
# ---------------------------------------------------------------------------

def _manual_value(Y, X, A, mask):
    """
    Reference implementation of the exact Mahalanobis loss.

    For each SNP i, restrict to observed traits O_i, compute
    (1/2) (z - x)_{O_i}^T A_{O_i}^{-1} (z - x)_{O_i}, and sum.
    """
    n, p = Y.shape
    observed = ~mask
    total = 0.0
    for i in range(p):
        obs = np.where(observed[:, i])[0]
        if len(obs) == 0:
            continue
        r = (Y - X)[np.ix_(obs, [i])].ravel()   # shape (|O|,)
        A_O = A[np.ix_(obs, obs)]
        total += 0.5 * float(r @ np.linalg.solve(A_O, r))
    return total


def _manual_value_over_cols(Y, X, A, mask, cols):
    total = 0.0
    for i in cols:
        obs = np.where(~mask[:, i])[0]
        if len(obs) == 0:
            continue
        r = (Y - X)[np.ix_(obs, [i])].ravel()
        A_O = A[np.ix_(obs, obs)]
        total += 0.5 * float(r @ np.linalg.solve(A_O, r))
    return total


def _numerical_gradient(obj, X, eps=1e-5):
    """
    Central-difference numerical gradient of obj.value at X.
    """
    n, p = X.shape
    G = np.zeros_like(X)
    for i in range(n):
        for j in range(p):
            Xp = X.copy(); Xp[i, j] += eps
            Xm = X.copy(); Xm[i, j] -= eps
            G[i, j] = (obj.value(Xp) - obj.value(Xm)) / (2.0 * eps)
    return G


def _manual_step_size_denom(Y, X, A, D, mask):
    """
    Reference implementation of the exact step size.
    """
    n, p = Y.shape
    observed = ~mask
    total = 0.0
    for i in range(p):
        obs = np.where(observed[:, i])[0]
        if len(obs) == 0:
            continue
        A_O = A[np.ix_(obs, obs)]
        L_O = np.linalg.cholesky(A_O)
        d_O = D[obs, i]
        Qt  = np.linalg.solve(L_O, d_O)
        total += float(np.dot(Qt, Qt))
    return total


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNNMCorrExactMissingness(unittest.TestCase):
    """
    Correctness tests for the exact per-pattern Mahalanobis formulation.
    """

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up NNM-Corr exact missingness tests")
        prob        = _make_problem()
        cls.prob    = prob
        cls.obj     = NNMCorrObjective(
            Y         = prob['Y'],
            radius    = prob['radius'],
            noise_cov = prob['noise_cov'],
            mask      = prob['mask'],
        )

    # ------------------------------------------------------------------
    # Test 1: value matches manual per-pattern sum
    # ------------------------------------------------------------------

    def test_value_matches_manual_pattern_sum(self):
        """
        value(X) equals the manual per-SNP Mahalanobis sum.

        This verifies that PatternBlock submatrix factors are correct
        and that the pattern grouping covers every SNP exactly once.
        """
        prob     = self.prob
        expected = _manual_value(prob['Y'], prob['X'], prob['A'], prob['mask'])
        actual   = self.obj.value(prob['X'])
        self.assertAlmostEqual(
            actual, expected, places=10,
            msg=(
                f"value() = {actual:.10g} does not match manual sum "
                f"{expected:.10g}. Difference: {abs(actual - expected):.3g}."
            ),
        )

    # ------------------------------------------------------------------
    # Test 2: all-missing column contributes zero and does not crash
    # ------------------------------------------------------------------

    def test_all_missing_column_zero_contribution(self):
        """
        A column with no observed traits contributes exactly zero to
        value() and does not raise.

        SNP 4 is all-missing in the test fixture.
        """
        prob = self.prob

        # SNP 4 must have all values missing
        obs_4  = np.where(~prob['mask'][:, 4])[0]  # should be empty
        self.assertEqual(
            len(obs_4), 0,
            msg="SNP 4 should be all-missing in the fixture.",
        )

        # Compare the full value against a manual sum 
        # that excludes column 4 explicitly.
        v_full = self.obj.value(prob['X'])
        v_excl4 = _manual_value_over_cols(
            prob['Y'], prob['X'], prob['A'], prob['mask'], cols=[0, 1, 2, 3]
        )
        self.assertAlmostEqual(
            v_full, v_excl4, places=10,
            msg = "All-missing column is contributing nonzero to value()."
            )



    def test_all_missing_column_no_crash(self):
        """
        gradient() and step_denom() do not crash when a column is all-missing.
        """
        prob = self.prob
        try:
            G = self.obj.gradient(prob['X'])
            D = np.ones_like(prob['X'])
            d = self.obj.step_denom(D)
        except Exception as e:
            self.fail(
                f"all-missing column caused an exception: {type(e).__name__}: {e}"
            )
        # all-missing column of gradient must be zero
        self.assertTrue(
            np.all(G[:, 4] == 0.0),
            msg="All-missing column of gradient() is nonzero.",
        )

        # step size should be finite
        self.assertTrue(
            np.isfinite(d),
            msg="Step size denominator is finite with all-missing column."
        )
        self.assertGreaterEqual(
            d, 0.0,
            msg="Step size denominator >= 0 with all-missing column."
        )

    # ------------------------------------------------------------------
    # Test 3: gradient matches finite differences of value
    # ------------------------------------------------------------------

    def test_gradient_matches_finite_differences(self):
        """
        gradient(X) matches the central-difference numerical gradient of
        value(X) to within floating-point tolerance.

        This is the most comprehensive correctness check: any error in
        the np.ix_ indexing, pattern grouping, or L_O_inv application
        will produce a mismatch here.
        """
        prob      = self.prob
        G_analytic  = self.obj.gradient(prob['X'])
        G_numerical = _numerical_gradient(self.obj, prob['X'], eps=1e-5)

        np.testing.assert_allclose(
            G_analytic, G_numerical,
            rtol = 1e-5,
            atol = 1e-7,
            err_msg = (
                "Analytic gradient does not match finite differences.\n"
                f"Max abs error: {np.abs(G_analytic - G_numerical).max():.3g}\n"
                f"Max rel error: "
                f"{(np.abs(G_analytic - G_numerical) / (np.abs(G_numerical) + 1e-15)).max():.3g}"
            ),
        )

    # ------------------------------------------------------------------
    # Test 4: step size denominator matches manual sum
    # ------------------------------------------------------------------

    def test_step_denom_matches_manual_pattern_sum(self):
        """
        step_denom(X) equals the manual per-SNP sum.
        """
        prob = self.prob
        rng = np.random.default_rng(99)
        D = rng.standard_normal(prob['Y'].shape)
        expected = _manual_step_size_denom(prob['Y'], prob['X'], prob['A'], D, prob['mask'])
        actual = self.obj.step_denom(D)

        self.assertAlmostEqual(
            actual, expected, places=10,
            msg=(
                f"step_denom() = {actual:.10g} does not match manual sum "
                f"{expected:.10g}. Difference: {abs(actual - expected):.3g}."
            ),
        )
