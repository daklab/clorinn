"""
test_pgd_nnm.py
---------------
Invariant tests for the PGD solver with the NNM objective.

Test classes
------------
TestPGDNNMFull    PGD + NNM, fully observed input.
TestPGDNNMMask    PGD + NNM, 10 % missing data.

Running
-------
    python -m unittest clorinn.tests.invariants.test_pgd_nnm -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest
import logging

from clorinn.optimize import ProjectedGradientDescent
from clorinn.tests.invariants.invariant_base import PGDInvariantBase
from clorinn.tests.invariants.invariant_utils import PGD_CONFIG, R_NUC, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestPGDNNMFull(PGDInvariantBase):
    """PGD + NNM, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up PGD / NNM / full invariant tests")
        prob = _build_problem()
        pgd = ProjectedGradientDescent(**PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC)
        cls.result = pgd.result
        cls.radius = R_NUC


# ---------------------------------------------------------------------------
# Missing data
# ---------------------------------------------------------------------------

class TestPGDNNMMask(PGDInvariantBase):
    """PGD + NNM, 10 % missing data."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = logging.getLogger(__name__)
        cls.logger_.info("Setting up PGD / NNM / mask invariant tests")
        prob = _build_problem()
        pgd = ProjectedGradientDescent(**PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC, mask=prob['mask'])
        cls.result = pgd.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
