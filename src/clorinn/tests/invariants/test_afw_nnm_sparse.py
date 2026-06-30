"""
test_afw_nnm_sparse.py
----------------------
Invariant tests for the Away-step Frank-Wolfe solver with the NNM-Sparse
objective.

Test classes
------------
TestAFWNNMSparseFull    AFW + NNM-Sparse, fully observed input.
TestAFWNNMSparseMask    AFW + NNM-Sparse, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_afw_nnm_sparse -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest
import logging

from clorinn.optimize import AwayStepFrankWolfe
from clorinn.tests.invariants.invariant_base import AFWSparseInvariantBase
from clorinn.tests.invariants.invariant_utils import (
    FW_CONFIG, R_NUC, L1_MULT, _build_problem,
)


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestAFWNNMSparseFull(AFWSparseInvariantBase):
    """AFW + NNM-Sparse, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up AFW / NNM-Sparse / full invariant tests")
        prob = _build_problem()
        m = AwayStepFrankWolfe(model='nnm-sparse', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, sparse_scale=L1_MULT)
        cls.result       = m.result
        cls.radius       = R_NUC
        cls.l1_threshold = m.objective.l1_threshold_


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestAFWNNMSparseMask(AFWSparseInvariantBase):
    """AFW + NNM-Sparse, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up AFW / NNM-Sparse / mask invariant tests")
        prob = _build_problem()
        m = AwayStepFrankWolfe(model='nnm-sparse', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC,
              sparse_scale=L1_MULT, mask=prob['mask'])
        cls.result       = m.result
        cls.radius       = R_NUC
        cls.l1_threshold = m.objective.l1_threshold_


if __name__ == '__main__':
    unittest.main()
