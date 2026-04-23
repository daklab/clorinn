"""
covariance_operator.py
----------------------
Runtime operator for the sampling covariance matrix used by NNM-Corr.

NoiseCovarianceOperator is a solver-internal class.  It is constructed from
a validated SamplingCovariance instance and caches the decompositions needed
by NNMCorrObjective: the Cholesky factor L, its inverse L_inv, the full
inverse Sigma_inv = A^{-1}, and the eigenvalues of A.

It also provides lazy per-pattern submatrix Cholesky factors for the
exact correlated-missingness formulation (Section 4.2 of [1]), populated
on first request and reused thereafter.

Users never interact with this class directly.  It is constructed once
inside NNMCorrObjective.__init__ and lives for the lifetime of the
objective.

References
----------
.. [1] Banerjee et al. (2025).  Convex approaches to isolate the shared and
       distinct genetic structures of subphenotypes.  medRxiv 2025.04.15.
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

    Notes
    -----
    Submatrix Cholesky factors for exact correlated-missingness patterns
    are computed lazily via submatrix_chol() and
    cached in _submatrix_cache_.
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

        # Lazy cache for per-pattern submatrix Cholesky factors
        # key: frozenset of observed trait indices
        # value: (L_O, L_O_inv) for the submatrix A[O, O]
        self._submatrix_cache_ = {}

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


    # ------------------------------------------------------------------
    # Submatrix Cholesky (lazy, for exact correlated missingness)
    # ------------------------------------------------------------------

    def submatrix_chol(self, pattern):
        """
        Return the Cholesky factor and its inverse for the submatrix of A
        indexed by the observed-trait pattern.

        The result is cached on first call and reused on subsequent calls
        for the same pattern.

        Parameters
        ----------
        pattern : frozenset of int
            Set of observed trait indices  O ⊆ {0, ..., n-1}.

        Returns
        -------
        L_O : np.ndarray [shape (|O|, |O|)]
            Lower Cholesky factor of A[O, O].

        L_O_inv : np.ndarray [shape (|O|, |O|)]
            Inverse of L_O.

        Raises
        ------
        np.linalg.LinAlgError
            If A[O, O] is not positive definite.  This signals a QC
            issue: the observed-trait submatrix is ill-conditioned,
            typically because a trait is observed at very few SNPs
            (Section 4.2 of [1]).
        """
        if pattern in self._submatrix_cache_:
            return self._submatrix_cache_[pattern]

        idx   = sorted(pattern)
        A_O   = self.A_[np.ix_(idx, idx)]
        L_O   = np.linalg.cholesky(A_O)
        L_O_inv = np.linalg.solve(L_O, np.eye(len(idx)))

        _logger.debug(
            f"CovarianceOperator.submatrix_chol: "
            f"|O|={len(idx)}, "
            f"lambda_min(A_O)={np.linalg.eigvalsh(A_O).min():.4g}"
        )

        self._submatrix_cache_[pattern] = (L_O, L_O_inv)
        return L_O, L_O_inv
