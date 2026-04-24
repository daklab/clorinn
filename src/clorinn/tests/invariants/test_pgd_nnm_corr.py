"""
test_pgd_nnm_corr.py
--------------------
Invariant tests for the PGD solver with the NNM-Corr objective
(Mahalanobis-type loss for correlated sampling errors).

Test classes
------------
TestPGDNNMCorrFull    PGD + NNM-Corr, fully observed input.
TestPGDNNMCorrMask    PGD + NNM-Corr, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_pgd_nnm_corr -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest

from clorinn.optimize import ProjectedGradientDescent
from clorinn.utils.logs import CustomLogger
from clorinn.tests.invariants.invariant_base import PGDInvariantBase
from clorinn.tests.invariants.invariant_utils import PGD_CONFIG, R_NUC, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestPGDNNMCorrFull(PGDInvariantBase):
    """PGD + NNM-Corr, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up PGD / NNM-Corr / full invariant tests")
        prob = _build_problem()
        pgd = ProjectedGradientDescent(model='nnm-corr', **PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC, noise_cov=prob['noise_cov'])
        cls.result = pgd.result
        cls.radius = R_NUC


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestPGDNNMCorrMask(PGDInvariantBase):
    """PGD + NNM-Corr, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__)
        cls.logger_.info("Setting up PGD / NNM-Corr / mask invariant tests")
        prob = _build_problem()
        pgd = ProjectedGradientDescent(model='nnm-corr', **PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC, mask=prob['mask'], noise_cov=prob['noise_cov'])
        cls.result = pgd.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
