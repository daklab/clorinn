"""
Objective functions and gradients for the nuclear norm minimisation problems.

Three problem formulations are provided:

  NNMObjective        Isotropic Frobenius loss (NNM and NNM-Sparse share
                      the same loss; the sparse model adds an l1 constraint
                      handled by NNMSparseObjective).

  NNMSparseObjective  Extends NNMObjective with the l1-constrained sparse
                      component and its exact projection update.

  NNMCorrObjective    Replaces the isotropic loss with a Mahalanobis-type
                      loss that accounts for correlated sampling errors
                      across traits (NNM-Corr).

Each class owns the problem data (Y, mask, weights, constraint radii) and
exposes a uniform interface consumed by all solvers:

    gradient(X)      Gradient of the loss at X.
    step_denom(D)    Line-search denominator  ||Q||_F^2  for direction D.
    value(X)         Scalar loss at X.
    pgd_step_size    Step size  eta = 1 / L_f  for projected gradient descent.
    project_sparse   Exact sparse-component update (NNMSparse only).

For NNM-Sparse the solver passes X + M to gradient() and value(); the
objective does not need to distinguish X and M, only project_sparse() does.

References
----------
.. [1] Banerjee et al. (2025).  Convex approaches to isolate the shared and
       distinct genetic structures of subphenotypes.  medRxiv 2025.04.15.
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import logging
from .projections import EuclideanProjection

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Base objective  (isotropic Frobenius loss)
# ---------------------------------------------------------------------------

class NNMObjective():
    """
    Isotropic Frobenius loss for the nuclear norm minimisation problem (NNM).

    Defines the loss

        f(X) = (1/2) * || R ⊙ W ⊙ (Z - X) ||_F^2,

    where R is the observation mask (0 at missing entries, 1 elsewhere) and
    W is an optional per-entry weight matrix (W = 1 when omitted).

    NNMSparseObjective inherits this class unchanged: the joint (X, M) loss
    is f(X + M), so the solver passes X + M to gradient() and value().

    Parameters
    ----------
    Y : np.ndarray [size (n, p); dtype: float]
        Observed Z-score matrix.  NaN values are added to the mask
        automatically.

    r : float
        Nuclear norm constraint radius.

    mask : np.ndarray [size (n, p); dtype: bool] or None, default=None
        Entries where mask is True are excluded from the loss and gradient.
        Combined with any NaN positions in Y.

    weight : np.ndarray [size (n, p); dtype: float] or None, default=None
        Per-entry weight matrix W.

    Attributes
    ----------
    Y_ : np.ndarray [size (n, p)]
        Input matrix with NaN replaced by 0.

    mask_ : np.ndarray [size (n, p); dtype: bool] or None
        Combined observation mask.  None when all entries are observed.

    radius_ : float
        Nuclear norm constraint radius.
    """

    def __init__(self, Y, radius, mask = None, weight = None):
        n, p = Y.shape

        # Replace NaN with 0 and record their positions.
        nan_mask    = np.isnan(Y)
        self.Y_     = np.nan_to_num(Y, copy = True, nan = 0.0)

        # Merge NaN mask with any explicit mask.
        input_mask  = np.zeros((n, p), dtype = bool) if mask is None else mask
        combined    = np.logical_or(nan_mask, input_mask)
        self.mask_  = combined if np.any(combined) else None

        self.radius_ = radius

        # Pre-apply mask to weights so that weight_mask_ is ready to use
        # directly in gradient and step_denom without re-masking each time.
        self.weight_mask_ = self._masked(weight)

        _logger.debug(
            f"{self.__class__.__name__}: shape={Y.shape}, radius={radius}, "
            f"masked_entries={np.sum(combined) if self.mask_ is not None else 0}, "
            f"weighted={'yes' if weight is not None else 'no'}"
        )
        return


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def radius(self):
        """Nuclear norm constraint radius."""
        return self.radius_


    # ------------------------------------------------------------------
    # Core solver interface
    # ------------------------------------------------------------------

    def gradient(self, X):
        """
        Gradient  grad f(X) = R ⊙ W^2 ⊙ (X - Z).

        For NNM-Sparse, pass X + M.  Masked entries are zero.
        """
        G = self._masked(X - self.Y_)
        if self.weight_mask_ is None:
            return G
        return np.square(self.weight_mask_) * G


    def step_denom(self, D):
        """
        Line-search denominator  ||R ⊙ W ⊙ D||_F^2.

        Used in the Frank-Wolfe exact line search:
            gamma = duality_gap / step_denom(D).
        """
        Dm = self._masked(D)
        Qt = Dm if self.weight_mask_ is None else self.weight_mask_ * Dm
        return float(np.linalg.norm(Qt, 'fro') ** 2)


    def value(self, X):
        """
        Loss  f(X) = (1/2) * ||R ⊙ W ⊙ (Z - X)||_F^2.

        For NNM-Sparse, pass X + M.
        """
        R = self._masked(self.Y_ - X)
        if self.weight_mask_ is not None:
            R = self.weight_mask_ * R
        return float(0.5 * np.linalg.norm(R, 'fro') ** 2)


    @property
    def pgd_step_size(self):
        """
        PGD step size  eta = 1 / L_f.

        Unweighted: L_f = 1, eta = 1.
        Weighted: L_f = max(W_ij^2), eta = 1 / max(W_ij^2).
        """
        if self.weight_mask_ is None:
            return 1.0
        return float(1.0 / np.max(np.square(self.weight_mask_)))


    def project_sparse(self, X):
        """Not applicable to NNM.  Overridden by NNMSparseObjective."""
        raise NotImplementedError(
            "project_sparse is only available for NNMSparseObjective."
        )


    # ------------------------------------------------------------------
    # Private helper
    # ------------------------------------------------------------------

    def _masked(self, X):
        """Return R ⊙ X  (zero at missing entries)."""
        if self.mask_ is None or X is None:
            return X
        return X * ~self.mask_


# ---------------------------------------------------------------------------
# NNM-Sparse objective
# ---------------------------------------------------------------------------

class NNMSparseObjective(NNMObjective):
    """
    Isotropic Frobenius loss with a joint nuclear-norm and l1 constraint.

    Extends NNMObjective with the l1-ball sparse component.  gradient() and
    value() are inherited unchanged.  The only addition is project_sparse(),
    which implements the exact sparse-component update

        M = P_{l1-ball}[ R ⊙ (Z - X) ]

    from the zero-gap projection theorem (Appendix B of [1]).

    Parameters
    ----------
    l1_multiplier : float
        Multiplier for the l1 constraint threshold.  The actual threshold is

            l1_threshold = l1_multiplier * ||Y||_1 / sqrt(max(n, p)),

        where ||Y||_1 is the matrix operator 1-norm (max absolute column
        sum) of the preprocessed input.

    simplex_method : string, default='sort'
        Algorithm for the simplex sub-projection inside project_sparse.
        See EuclideanProjection for available options.

    Attributes
    ----------
    l1_threshold_ : float
        Scaled l1 constraint radius used by project_sparse.
    """

    def __init__(self, Y, radius, l1_multiplier,
                 mask = None, weight = None, simplex_method = 'sort'):
        super().__init__(Y, radius, mask = mask, weight = weight)
        self.simplex_method_ = simplex_method

        # Scale the l1 multiplier by the data-dependent factor.
        # np.linalg.norm(Y_, ord=1) for a 2-D array returns the max absolute
        # column sum (matrix 1-norm), matching the original implementation.
        self.l1_threshold_ = (
            l1_multiplier
            * np.linalg.norm(self.Y_, ord = 1)
            / np.sqrt(np.max(self.Y_.shape))
        )

        _logger.debug(
            f"NNMSparseObjective: l1_multiplier={l1_multiplier}, "
            f"l1_threshold={self.l1_threshold_:.4g}"
        )
        return


    @property
    def l1_threshold(self):
        """Scaled l1 constraint radius."""
        return self.l1_threshold_


    def project_sparse(self, X):
        """
        Exact sparse-component update  M = P_{l1-ball}[ R ⊙ (Z - X) ].

        Unique minimiser of f(X, M) over the l1 ball for fixed X.
        Missing entries of M are zero.
        """
        n, p = self.Y_.shape
        B = self._masked(self.Y_ - X)
        ep = EuclideanProjection(method = self.simplex_method_, target = 'l1')
        ep.fit(B.ravel(), a = self.l1_threshold_)
        return ep.proj.reshape(n, p)


# ---------------------------------------------------------------------------
# NNM-Corr objective  (Mahalanobis loss)
# ---------------------------------------------------------------------------

class NNMCorrObjective(NNMObjective):
    """
    Mahalanobis-type loss for the correlated-error NNM problem (NNM-Corr).

    Replaces the isotropic Frobenius loss with

        f(X) = (1/2) * || L^{-1} (R ⊙ (Z - X)) ||_F^2,

    where A = L L^T is the sampling covariance matrix induced by sample
    overlap across GWAS traits (Section 2 of [1]).  The nuclear norm oracle
    and stopping criteria are unchanged.

    Parameters
    ----------
    L_inv : np.ndarray [size (n, n); dtype: float]
        Inverse of the lower Cholesky factor of A  (L_inv = L^{-1}).

    Sigma_inv : np.ndarray [size (n, n); dtype: float]
        Inverse of the sampling covariance  (Sigma_inv = A^{-1} = L_inv^T @ L_inv).

    Attributes
    ----------
    L_inv_ : np.ndarray [size (n, n)]
        Stored Cholesky inverse.

    Sigma_inv_ : np.ndarray [size (n, n)]
        Stored covariance inverse.

    Notes
    -----
    The mask is applied before the A^{-1} product and again afterwards.
    The second application zeros out leakage at missing positions introduced
    by the off-diagonal structure of A^{-1}.
    """

    def __init__(self, Y, radius, L_inv, Sigma_inv, mask = None, weight = None):
        if L_inv is None or Sigma_inv is None:
            raise ValueError(
                "NNMCorrObjective requires both L_inv and Sigma_inv."
            )
        super().__init__(Y, radius, mask = mask, weight = weight)

        self.L_inv_     = L_inv
        self.Sigma_inv_ = Sigma_inv

        # Cache the PGD step size: eta = lambda_min(A) = 1 / lambda_max(A^{-1}).
        # eigvalsh is used because Sigma_inv is symmetric positive definite.
        lambda_max_Ainv     = float(np.linalg.eigvalsh(Sigma_inv).max())
        self._pgd_step_size = 1.0 / lambda_max_Ainv

        _logger.debug(
            f"NNMCorrObjective: lambda_max(A^{{-1}})={lambda_max_Ainv:.4g}, "
            f"pgd_step_size={self._pgd_step_size:.4g}"
        )
        return


    def gradient(self, X):
        """
        Gradient of the Mahalanobis loss at X.

        Unweighted:  grad f(X) = R ⊙ [ A^{-1} (R ⊙ (X - Z)) ].
        Weighted:    grad f(X) = R ⊙ [ L^{-T} (W^2 ⊙ (L^{-1} (R ⊙ (X - Z)))) ].

        The mask is applied before and after A^{-1} to exclude missing entries
        and suppress leakage from the off-diagonal structure of A^{-1}.
        """
        G = self._masked(X - self.Y_)
        if self.weight_mask_ is None:
            gx = self.Sigma_inv_ @ G
        else:
            gx = self.L_inv_.T @ (np.square(self.weight_mask_) * (self.L_inv_ @ G))
        return self._masked(gx)


    def step_denom(self, D):
        """
        Line-search denominator  || W ⊙ L^{-1} (R ⊙ D) ||_F^2.
        """
        Dm = self._masked(D)
        Qt = self.L_inv_ @ Dm
        if self.weight_mask_ is not None:
            Qt = self.weight_mask_ * Qt
        return float(np.linalg.norm(Qt, 'fro') ** 2)


    def value(self, X):
        """
        Mahalanobis loss  f(X) = (1/2) * || L^{-1} (R ⊙ (Z - X)) ||_F^2.
        """
        R = self._masked(self.Y_ - X)
        R = self.L_inv_ @ R
        if self.weight_mask_ is not None:
            R = self.weight_mask_ * R
        return float(0.5 * np.linalg.norm(R, 'fro') ** 2)


    @property
    def pgd_step_size(self):
        """
        PGD step size  eta = lambda_min(A) = 1 / lambda_max(A^{-1}).

        When A = I this recovers eta = 1.
        """
        return self._pgd_step_size
