"""
test_fw_nnm_corr.py
-------------------
Invariant tests for the Frank-Wolfe solver with the NNM-Corr objective
(Mahalanobis-type loss for correlated sampling errors).

Test classes
------------
TestFWNNMCorrFull    FW + NNM-Corr, fully observed input.
TestFWNNMCorrMask    FW + NNM-Corr, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_fw_nnm_corr -v
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

class TestFWNNMCorrFull(FWInvariantBase):
    """FW + NNM-Corr, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up FW / NNM-Corr / full invariant tests")
        prob = _build_problem()
        m = FrankWolfe(model='nnm-corr', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, noise_cov=prob['noise_cov'])
        cls.result = m.result
        cls.radius = R_NUC


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestFWNNMCorrMask(FWInvariantBase):
    """FW + NNM-Corr, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up FW / NNM-Corr / mask invariant tests")
        prob = _build_problem()
        m = FrankWolfe(model='nnm-corr', **FW_CONFIG)
        m.fit(prob['Y'], radius=R_NUC, mask=prob['mask'], noise_cov=prob['noise_cov'])
        cls.result = m.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
