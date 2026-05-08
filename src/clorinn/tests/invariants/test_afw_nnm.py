"""
test_afw_nnm.py
---------------
Invariant tests for the Away-step Frank-Wolfe solver with the NNM objective.

Test classes
------------
TestAFWNNMFull      AFW + NNM, fully observed input.
TestAFWNNMMask      AFW + NNM, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_afw_nnm -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest

from clorinn.optimize import AwayStepFrankWolfe
from clorinn.utils.logs import CustomLogger
from clorinn.tests.invariants.invariant_base import AFWInvariantBase
from clorinn.tests.invariants.invariant_utils import FW_CONFIG, R_NUC, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestAFWNNMFull(AFWInvariantBase):
    """AFW + NNM, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up AFW / NNM / full invariant tests")
        prob = _build_problem()
        m = AwayStepFrankWolfe(model='nnm', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC)
        cls.result = m.result
        cls.radius = R_NUC


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestAFWNNMMask(AFWInvariantBase):
    """AFW + NNM, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up AFW / NNM / mask invariant tests")
        prob = _build_problem()
        m = AwayStepFrankWolfe(model='nnm', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, mask=prob['mask'])
        cls.result = m.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
