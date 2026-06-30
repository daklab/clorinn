"""
noise_cov_op.py
----------------------
Runtime operator for the sampling covariance matrix used by NNM-Corr.

NoiseCovarianceOperator is a solver-internal class.  It is constructed from
a validated SamplingCovariance instance and caches the decompositions needed
by NNMCorrObjective: the Cholesky factor L, its inverse L_inv, the full
inverse Sigma_inv = A^{-1}, and the eigenvalues of A.

Users never interact with this class directly.  It is constructed once
inside NNMCorrObjective.__init__ and lives for the lifetime of the
objective.
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import logging
import numpy as np

_logger = logging.getLogger(__name__)


class NoiseCovarianceOperator:
    """
    Cached decompositions of the sampling covariance matrix A.

    Parameters
    ----------
    noise_cov : SamplingCovariance
        Validated positive-definite sampling covariance.  Must satisfy
        noise_cov.A is symmetric PD.

    Attributes
    ----------
    A_ : np.ndarray [shape (n, n)]
        Sampling covariance matrix.

    L_ : np.ndarray [shape (n, n)]
        Lower Cholesky factor of A  (A = L L^T).

    L_inv_ : np.ndarray [shape (n, n)]
        Inverse of L  (L_inv = L^{-1}).

    Sigma_inv_ : np.ndarray [shape (n, n)]
        Inverse of A  (Sigma_inv = A^{-1} = L_inv^T @ L_inv).

    eigvals_ : np.ndarray [shape (n,)]
        Eigenvalues of A in ascending order.

    pgd_step_size_ : float
        PGD step size  eta = lambda_min(A) = 1 / lambda_max(A^{-1}).

    """

    def __init__(self, noise_cov):
        A = noise_cov.A

        # Cholesky and its inverse
        self.A_         = A
        self.L_         = np.linalg.cholesky(A)
        self.L_inv_     = np.linalg.solve(self.L_, np.eye(A.shape[0]))

        # Full inverse and eigenvalues
        self.Sigma_inv_ = self.L_inv_.T @ self.L_inv_
        self.eigvals_   = np.linalg.eigvalsh(A)   # ascending order

        # PGD step size: eta = lambda_min(A) = 1 / lambda_max(A^{-1})
        self.pgd_step_size_ = float(self.eigvals_.min())

        _logger.debug(
            f"CovarianceOperator: n={A.shape[0]}, "
            f"lambda_min(A)={self.eigvals_.min():.4g}, "
            f"lambda_max(A)={self.eigvals_.max():.4g}, "
            f"cond(A)={self.eigvals_.max() / self.eigvals_.min():.4g}, "
            f"pgd_step_size={self.pgd_step_size_:.4g}"
        )


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n(self):
        """Number of traits."""
        return self.A_.shape[0]

    @property
    def pgd_step_size(self):
        """PGD step size eta = lambda_min(A)."""
        return self.pgd_step_size_
