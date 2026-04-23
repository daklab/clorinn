
#from .test_converged_nuclear_norm import TestConvergedNuclearNorm

from .test_current_behavior import (
    TestFixturesPresent,
    TestRegressionNNM, TestRegressionNNMMask,
    TestRegressionNNMSparse, TestRegressionNNMSparseMask,
    TestRegressionNNMCorr, TestRegressionNNMCorrMask,
    TestRegressionPGD, TestRegressionPGDMask,
)
from .test_degenerate_inputs import TestDegenerateInputs
