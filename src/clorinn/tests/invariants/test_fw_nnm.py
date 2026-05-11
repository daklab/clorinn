"""
test_fw_nnm.py
--------------
Invariant tests for the Frank-Wolfe solver with the NNM objective.

Test classes
------------
TestFWNNMFull      FW + NNM, fully observed input.
TestFWNNMMask      FW + NNM, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_fw_nnm -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest
import logging

from clorinn.optimize import FrankWolfe
from clorinn.tests.invariants.invariant_base import FWInvariantBase
from clorinn.tests.invariants.invariant_utils import FW_CONFIG, R_NUC, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestFWNNMFull(FWInvariantBase):
    """FW + NNM, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up FW / NNM / full invariant tests")
        prob = _build_problem()
        m = FrankWolfe(model='nnm', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC)
        cls.result = m.result
        cls.radius = R_NUC


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestFWNNMMask(FWInvariantBase):
    """FW + NNM, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up FW / NNM / mask invariant tests")
        prob = _build_problem()
        m = FrankWolfe(model='nnm', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, mask=prob['mask'])
        cls.result = m.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
