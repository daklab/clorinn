
# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------
"""
Tests for individual classes / functions in isolation.
"""
from .unit.test_active_set import (
    TestActiveSetConstruct,
    TestActiveSetReconstruct,
    TestActiveSetUpdateFW,
    TestActiveSetUpdateAway,
    TestActiveSetOracleAway,
    TestActiveSetGammaMax,
    TestActiveSetPrune,
    TestActiveSetInvariants,
    TestActiveSetLongRunSequence,
)

# ---------------------------------------------------------------------------
# Invariant tests
# ---------------------------------------------------------------------------
"""
Properties the algorithm must satisfy by construction, 
independent of any specific numerical output.
Examples: converged nuclear norm, feasibility, monotonicity, 
          duality gap sign, step size bounds, etc.
But, they do not prove we implemented the right formula. 
They only check that some important property holds.
"""
from .invariants.test_fw_nnm import TestFWNNMFull, TestFWNNMMask
from .invariants.test_fw_nnm_sparse import TestFWNNMSparseFull, TestFWNNMSparseMask
from .invariants.test_fw_nnm_corr import TestFWNNMCorrFull, TestFWNNMCorrMask
from .invariants.test_afw_nnm import TestAFWNNMFull, TestAFWNNMMask
from .invariants.test_afw_nnm_sparse import TestAFWNNMSparseFull, TestAFWNNMSparseMask
from .invariants.test_afw_nnm_corr import TestAFWNNMCorrFull, TestAFWNNMCorrMask
from .invariants.test_pgd_nnm import TestPGDNNMFull, TestPGDNNMMask
from .invariants.test_pgd_nnm_sparse import TestPGDNNMSparseFull, TestPGDNNMSparseMask
from .invariants.test_pgd_nnm_corr import TestPGDNNMCorrFull, TestPGDNNMCorrMask

# ---------------------------------------------------------------------------
# Theory tests
# ---------------------------------------------------------------------------
"""
Tests that encode specific equations / update rules from the design document.
"""
from .theory.test_nnm_corr_exact_missingness import TestNNMCorrExactMissingness

# ---------------------------------------------------------------------------
# Regression tests
# ---------------------------------------------------------------------------
from .regression.test_current_behavior import (
    TestFixturesPresent,
    TestRegressionFWNNM, TestRegressionFWNNMMask,
    TestRegressionFWNNMSparse, TestRegressionFWNNMSparseMask,
    TestRegressionFWNNMCorr, TestRegressionFWNNMCorrMask,
    TestRegressionPGDNNM, TestRegressionPGDNNMMask,
    TestRegressionPGDNNMSparse, TestRegressionPGDNNMSparseMask,
    TestRegressionPGDNNMCorr, TestRegressionPGDNNMCorrMask,
)


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
from .integration.test_degenerate_inputs import TestDegenerateInputs
