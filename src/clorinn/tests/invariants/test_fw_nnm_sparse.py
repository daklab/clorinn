"""
test_fw_nnm_sparse.py
---------------------
Invariant tests for the Frank-Wolfe solver with the NNM-Sparse objective.

Test classes
------------
TestFWNNMSparseFull    FW + NNM-Sparse, fully observed input.
TestFWNNMSparseMask    FW + NNM-Sparse, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_fw_nnm_sparse -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest

from clorinn.optimize import FrankWolfe
from clorinn.utils.logs import CustomLogger
from clorinn.tests.invariants.invariant_base import FWSparseInvariantBase
from clorinn.tests.invariants.invariant_utils import FW_CONFIG, R_NUC, L1_MULT, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestFWNNMSparseFull(FWSparseInvariantBase):
    """FW + NNM-Sparse, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up FW / NNM-Sparse / full invariant tests")
        prob = _build_problem()
        m = FrankWolfe(model='nnm-sparse', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, sparse_scale=L1_MULT)
        cls.result      = m.result
        cls.radius      = R_NUC
        cls.l1_threshold = m.objective.l1_threshold_


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestFWNNMSparseMask(FWSparseInvariantBase):
    """FW + NNM-Sparse, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up FW / NNM-Sparse / mask invariant tests")
        prob = _build_problem()
        m = FrankWolfe(model='nnm-sparse', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, sparse_scale=L1_MULT, mask=prob['mask'])
        cls.result       = m.result
        cls.radius       = R_NUC
        cls.l1_threshold = m.objective.l1_threshold_


if __name__ == '__main__':
    unittest.main()
