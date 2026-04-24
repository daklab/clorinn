"""
test_pgd_nnm_sparse.py
----------------------
Invariant tests for the PGD solver with the NNM-Sparse objective.

Test classes
------------
TestPGDNNMSparseFull    PGD + NNM-Sparse, fully observed input.
TestPGDNNMSparseMask    PGD + NNM-Sparse, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_pgd_nnm_sparse -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest

from clorinn.optimize import ProjectedGradientDescent
from clorinn.utils.logs import CustomLogger
from clorinn.tests.invariants.invariant_base import PGDSparseInvariantBase
from clorinn.tests.invariants.invariant_utils import PGD_CONFIG, R_NUC, L1_MULT, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestPGDNNMSparseFull(PGDSparseInvariantBase):
    """PGD + NNM-Sparse, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up PGD / NNM-Sparse / full invariant tests")
        prob = _build_problem()
        pgd = ProjectedGradientDescent(model='nnm-sparse', **PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC, sparse_scale=L1_MULT)
        cls.result = pgd.result
        cls.radius = R_NUC
        cls.l1_threshold = pgd.objective.l1_threshold_


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestPGDNNMSparseMask(PGDSparseInvariantBase):
    """PGD + NNM-Sparse, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up PGD / NNM-Sparse / mask invariant tests")
        prob = _build_problem()
        pgd = ProjectedGradientDescent(model='nnm-sparse', **PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC, sparse_scale=L1_MULT, mask=prob['mask'])
        cls.result = pgd.result
        cls.radius = R_NUC
        cls.l1_threshold = pgd.objective.l1_threshold_


if __name__ == '__main__':
    unittest.main()
