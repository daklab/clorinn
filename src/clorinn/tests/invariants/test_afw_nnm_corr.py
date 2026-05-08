"""
test_afw_nnm_corr.py
--------------------
Invariant tests for the Away-step Frank-Wolfe solver with the NNM-Corr
(correlated noise) objective.

Test classes
------------
TestAFWNNMCorrFull    AFW + NNM-Corr, fully observed input.
TestAFWNNMCorrMask    AFW + NNM-Corr, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_afw_nnm_corr -v
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

class TestAFWNNMCorrFull(AFWInvariantBase):
    """AFW + NNM-Corr, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up AFW / NNM-Corr / full invariant tests")
        prob = _build_problem()
        m = AwayStepFrankWolfe(model='nnm-corr', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, noise_cov=prob['noise_cov'])
        cls.result = m.result
        cls.radius = R_NUC


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestAFWNNMCorrMask(AFWInvariantBase):
    """AFW + NNM-Corr, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up AFW / NNM-Corr / mask invariant tests")
        prob = _build_problem()
        m = AwayStepFrankWolfe(model='nnm-corr', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC,
              noise_cov=prob['noise_cov'], mask=prob['mask'])
        cls.result = m.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
