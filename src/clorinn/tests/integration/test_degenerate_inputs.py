import unittest
import numpy as np
import logging

from clorinn.optimize import FrankWolfe

class TestDegenerateInputs(unittest.TestCase):
    """
    Numerical robustness on degenerate inputs: zero matrix, rank-1 matrix,
    and a nearly-empty masked problem. 

    The goal is to catch:
      - NaNs/Infs entering the iterate
      - invalid step sizes
      - infeasible iterates
      - crashes on nearly-empty masked problems
    """

    def setUp(self):
        self.logger_ = logging.getLogger(__name__)

    def _make_solver(self):
        return FrankWolfe(
            model = 'nnm',
            max_iter = 10,
            svd_method = 'direct',
            stop_criteria = ['duality_gap', 'step_size'],
            verbose = 0,
        )

    def _assert_feasible(self, result, radius):
        self.assertIsNotNone(result.X)
        self.assertTrue(np.isfinite(result.X).all())
        self.assertTrue(
            np.isfinite(np.asarray(result.history.loss, dtype=float)).all()
        )
        if len(result.history.duality_gap) > 1:
            dg = np.asarray(result.history.duality_gap[1:], dtype=float)
            self.assertTrue(np.isfinite(dg).all())
        if len(result.history.step_size) > 1:
            ss = np.asarray(result.history.step_size[1:], dtype=float)
            self.assertTrue(np.isfinite(ss).all())
            self.assertTrue(np.all(ss >= 0.0))
            self.assertTrue(np.all(ss <= 1.0))
        nuc = np.linalg.norm(result.X, ord = 'nuc')
        self.assertLessEqual(nuc, radius + 1e-10)

    def test_zero_matrix(self):
        self.logger_.info("Testing zero matrix input")
        Y      = np.zeros((2, 3))
        radius = 1.0
        result = self._make_solver().fit(Y, radius).result
        self._assert_feasible(result, radius)
        np.testing.assert_allclose(result.X, np.zeros_like(Y), atol = 0.0)

    def test_rank_one_matrix(self):
        """Near-zero denominators can arise from rank-1 input."""
        self.logger_.info("Testing rank-1 matrix input")
        Y = np.array([
            [2.0, 4.0, 6.0],
            [1.0, 2.0, 3.0],
        ])
        radius = 5.0
        result = self._make_solver().fit(Y, radius).result
        self._assert_feasible(result, radius)

    def test_single_observed_entry(self):
        """
        Only one entry observed.  Loss must equal 0.5 * (Y[0,1] - X[0,1])^2.

        NOTE: earlier raised numpy.linalg.LinAlgError: SVD did not converge.
        This is a known edge-case bug which was fixed.
        """
        self.logger_.info("Testing single observed entry")
        Y = np.array([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        mask         = np.ones_like(Y, dtype=bool)
        mask[0, 1]   = False
        radius       = 1.0
        result       = self._make_solver().fit(Y, radius, mask=mask).result
        self._assert_feasible(result, radius)
        expected_loss = 0.5 * (Y[0, 1] - result.X[0, 1]) ** 2
        self.assertAlmostEqual(result.loss, expected_loss, places=12)
