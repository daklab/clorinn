# test_converged_nuclear_norm

import unittest
import numpy as np

from ..tests import toy_data
from ..optimize import FrankWolfe
from ..utils.logs import CustomLogger

class TestConvergedNuclearNorm(unittest.TestCase):

    def test_converged_nuclear_norm(self):

        self.logger_ = CustomLogger(__name__)
        info_msg = "Testing if nuclear norm of converged solution is equal to the input constraint."
        self.logger_.info(info_msg)
        
        """
        Simulate data
        """
        n = 50
        p = 1000
        k = 10
        Q = 3
        h2 = 0.4
        h2_shared_frac = 0.5
        aq = 0.6
        a0 = 0.2
        sharing_proportion = 1.0
        nsample = 10000
        # nsample = np.random.uniform(nsample_min, nsample_max, n)
        g2 = h2 * h2_shared_frac
        Z, effect_size_obs, effect_size_true, L, F, M, C = \
            toy_data.effect_size(
                n, p, k, Q, h2, g2, aq, a0, nsample,
                sharing_proportion = sharing_proportion,
                cov_design = 'blockdiag', shuffle = False,
                seed = 1210)

        Z_cent = toy_data.do_standardize(Z, scale = False)
        Z_true_cent = toy_data.do_standardize(effect_size_true)

        """
        Run optimization and check converged nuclear norm
        """
        target_nucnorm = 100.0
        err_msg = f"Nuclear Norm after convergence is not equal to the constrained value."
        nnm = FrankWolfe(model = 'nnm', svd_max_iter = 50)
        nnm.fit(Z_cent, target_nucnorm, Ytrue = Z_true_cent)
        converged_nucnorm = np.linalg.norm(nnm.X, ord='nuc')
        np.testing.assert_almost_equal(converged_nucnorm, target_nucnorm, err_msg = err_msg)
