"""
test_pgd_nnm.py
---------------
Invariant tests for the PGD warm-start solver with the NNM objective.

PGD is currently implemented for NNM only. When NNM-Sparse and NNM-Corr
support is added to PGDWarmStart, add TestPGDNNMSparseFull,
TestPGDNNMSparseMask, TestPGDNNMCorrFull, TestPGDNNMCorrMask here
following the same pattern, and extend PGDInvariantBase (or add
PGDSparseInvariantBase) in invariant_base.py as needed.

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

import logging
import unittest

from clorinn.optimize import PGDWarmStart
from clorinn.utils.logs import CustomLogger
from clorinn.tests.invariants.invariant_base import PGDInvariantBase
from clorinn.tests.invariants.invariant_utils import PGD_CONFIG, R_NUC, _build_problem


# ---------------------------------------------------------------------------
# Fully observed
# ---------------------------------------------------------------------------

class TestPGDNNMFull(PGDInvariantBase):
    """PGD + NNM, fully observed input."""

    @classmethod
    def setUpClass(cls):
        cls.logger_ = CustomLogger(__name__, level=logging.INFO)
        cls.logger_.info("Setting up PGD / NNM / full invariant tests")
        prob = _build_problem()
        pgd = PGDWarmStart(**PGD_CONFIG)
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
        cls.logger_ = CustomLogger(__name__, level=logging.INFO)
        cls.logger_.info("Setting up PGD / NNM / mask invariant tests")
        prob = _build_problem()
        pgd = PGDWarmStart(**PGD_CONFIG)
        pgd.fit(prob['Y'], radius=R_NUC, mask=prob['mask'])
        cls.result = pgd.result
        cls.radius = R_NUC


if __name__ == '__main__':
    unittest.main()
