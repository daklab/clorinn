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
from dataclasses import dataclass
from .projections import EuclideanProjection
from .noise_cov_op import NoiseCovarianceOperator

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
        #return self.squared_frobenius_norm(Qt)


    def value(self, X):
        """
        Loss  f(X) = (1/2) * ||R ⊙ W ⊙ (Z - X)||_F^2.

        For NNM-Sparse, pass X + M.
        """
        R = self._masked(self.Y_ - X)
        if self.weight_mask_ is not None:
            R = self.weight_mask_ * R
        #return float(0.5 * np.linalg.norm(R, 'fro') ** 2)
        return 0.5 * self.squared_frobenius_norm(R)


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


    def squared_frobenius_norm(self, X):
        """Slightly cheaper Frobenius norm / small gain, but in inner loop."""
        #return float(np.einsum('ij,ij->', X, X))
        return float(np.sum(X * X))


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

@dataclass
class PatternBlock:
    obs_idx   : np.ndarray   # sorted observed trait indices, shape (|O|,)
    col_idx   : np.ndarray   # SNP column indices, shape (|I_O|,)
    L_inv     : np.ndarray   # L_O^{-1}, shape (|O|, |O|)
    Sigma_inv : np.ndarray   # A_O^{-1} = L_O^{-T} L_O^{-1}, shape (|O|, |O|)
    lam_min   : float


class NNMCorrObjective(NNMObjective):
    """
    Mahalanobis-type loss for the correlated-error NNM problem (NNM-Corr).

    Fully observed case
    -------------------
    Replaces the isotropic Frobenius loss with the Mahalanobis-type loss


        f(X) = (1/2) * || L^{-1} (Z - X) ||_F^2,

    where A = L L^T is the sampling covariance matrix induced by sample
    overlap across GWAS traits.  The nuclear norm oracle and stopping 
    criteria are unchanged.

    Missing data (exact formulation)
    ---------------------------------
    When the input Z has missing entries, the loss is restricted to the
    observed coordinates of each SNP,
    
        f(X) = (1/2) * sum_i (z_i - x_i)_{O_i}^T A_{O_i}^{-1} (z_i - x_i)_{O_i},
    
    where O_i is the set of observed traits for SNP i and A_{O_i} is the
    principal submatrix of A indexed by O_i.  Applying the full A^{-1} and
    re-masking is an approximation that can bias the loss toward directions
    of high noise correlation.  The exact formulation avoids this by using
    only the submatrix A_{O_i}^{-1} for each SNP.
    
    At construction, all distinct missingness patterns {O_i} are identified,
    their submatrix Cholesky factors L_{O_i} are precomputed once, and the
    results are stored as PatternBlock objects.  gradient(), step_denom(),
    value(), and pgd_step_size all dispatch to the per-pattern path when
    missing data is present, and to the fast global path otherwise.


    Parameters
    ----------
    Y, radius, mask, weight : see NNMObjective

    noise_cov : SamplingCovariance
        See utils/sampling_covariance.py for this object.

    """

    def __init__(self, Y, radius, noise_cov, mask = None, weight = None):
        if noise_cov is None:
            raise ValueError("NNMCorrObjective requires noise_cov.")
        super().__init__(Y, radius, mask = mask, weight = weight)

        # handle noise covariance 
        # # cov_op is stored on self so that the submatrix Cholesky cache
        # (used by exact correlated-missingness gradient)
        # persists across gradient() calls.
        self.cov_op_ = NoiseCovarianceOperator(noise_cov)
        self.L_inv_         = self.cov_op_.L_inv_
        self.Sigma_inv_     = self.cov_op_.Sigma_inv_

        # build missingness pattern
        self.patterns_ = self._build_patterns(self.mask_)

        # Cache the PGD step size
        self.pgd_step_size_ = min(pat.lam_min for pat in self.patterns_)

        _logger.debug(
            f"NNMCorrObjective: pgd_step_size={self.pgd_step_size:.4g}, "
            f"n_patterns={len(self.patterns_)}"
        )

        return


    def gradient(self, X):
        """
        Gradient of the Mahalanobis loss at X.

        Fully observed:  grad f(X) = A^{-1} (X - Z).
        Missing data:    per-pattern A_O^{-1} applied to observed rows,
                         zero-lifted at unobserved rows (Eq. 28 of [1]).
        """
        G = self._masked(X - self.Y_)
        pat0 = self.patterns_[0]
    
        # Fast path: fully observed
        if pat0.obs_idx is None:
            gx = pat0.Sigma_inv @ G
            if self.weight_mask_ is not None:
                gx = pat0.L_inv.T @ (np.square(self.weight_mask_) * (pat0.L_inv @ G))
            return self._masked(gx)
    
        # Exact path: per-pattern submatrix application
        Gout = np.zeros_like(G)
        for pat in self.patterns_:
            block = G[np.ix_(pat.obs_idx, pat.col_idx)]
            if self.weight_mask_ is not None:
                W  = self.weight_mask_[np.ix_(pat.obs_idx, pat.col_idx)]
                Qt = pat.L_inv.T @ (np.square(W) * (pat.L_inv @ block))
            else:
                Qt = pat.Sigma_inv @ block # one matmul, not two (L_inv @ L_inv.T @ block)
            Gout[np.ix_(pat.obs_idx, pat.col_idx)] = Qt
        return Gout
    
    
    def step_denom(self, D):
        """
        Line-search denominator.
    
        Fully observed:  || L^{-1} D ||_F^2.
        Missing data:    sum over patterns || L_O^{-1} D_{O, I_O} ||_F^2
        """
        Dm   = self._masked(D)
        pat0 = self.patterns_[0]
    
        # Fast path: fully observed
        if pat0.obs_idx is None:
            Qt = pat0.L_inv @ Dm
            if self.weight_mask_ is not None:
                Qt = self.weight_mask_ * Qt
            return float(np.linalg.norm(Qt, 'fro') ** 2)
            #return self.squared_frobenius_norm(Qt)
    
        # Exact path: per-pattern submatrix application
        total = 0.0
        for pat in self.patterns_:
            block = Dm[np.ix_(pat.obs_idx, pat.col_idx)]
            Qt    = pat.L_inv @ block
            if self.weight_mask_ is not None:
                Qt = self.weight_mask_[np.ix_(pat.obs_idx, pat.col_idx)] * Qt
            total += float(np.linalg.norm(Qt, 'fro') ** 2)
            #total += self.squared_frobenius_norm(Qt)
        return total
    
    
    def value(self, X):
        """
        Mahalanobis loss.
    
        Fully observed:  f(X) = (1/2) || L^{-1} (Z - X) ||_F^2.
        Missing data:    f(X) = (1/2) sum_O || L_O^{-1} (Z - X)_{O, I_O} ||_F^2
        """
        R    = self._masked(self.Y_ - X)
        pat0 = self.patterns_[0]
    
        # Fast path: fully observed
        if pat0.obs_idx is None:
            Qt = pat0.L_inv @ R
            if self.weight_mask_ is not None:
                Qt = self.weight_mask_ * Qt
            return float(0.5 * np.linalg.norm(Qt, 'fro') ** 2)
            #return 0.5 * self.squared_frobenius_norm(Qt)
    
        # Exact path: per-pattern submatrix application
        total = 0.0
        for pat in self.patterns_:
            block = R[np.ix_(pat.obs_idx, pat.col_idx)]
            Qt    = pat.L_inv @ block
            if self.weight_mask_ is not None:
                Qt = self.weight_mask_[np.ix_(pat.obs_idx, pat.col_idx)] * Qt
            total += float(0.5 * np.linalg.norm(Qt, 'fro') ** 2)
            #total += 0.5 * self.squared_frobenius_norm(Qt)
        return total
    
    
    @property
    def pgd_step_size(self):
        return self.pgd_step_size_


    def _build_patterns(self, mask):
        """
        Precompute all missingness patterns and their submatrix factors.
    
        Returns a list of PatternBlock objects, one per distinct pattern.
        When mask is None, returns a single sentinel with obs_idx=None
        signalling the fast global path.
        """
        if mask is None:
            return [PatternBlock(
                obs_idx = None,
                col_idx = None,
                L_inv   = self.cov_op_.L_inv_,
                Sigma_inv = self.cov_op_.Sigma_inv_,
                lam_min = self.cov_op_.pgd_step_size_,
            )]
    
        observed = ~mask                              # True at observed entries
        n, p     = observed.shape
    
        pattern_map = {}                              # frozenset -> list of col indices
        for i in range(p):
            key = frozenset(np.where(observed[:, i])[0])
            if key not in pattern_map:
                pattern_map[key] = []
            pattern_map[key].append(i)
    
        blocks = []
        for key, cols in pattern_map.items():
            if len(key) == 0:          # all-missing column group — skip entirely
                continue
            obs = np.array(sorted(key), dtype=int)
            A_O     = self.cov_op_.A_[np.ix_(obs, obs)]
            L_O     = np.linalg.cholesky(A_O)
            L_O_inv = np.linalg.solve(L_O, np.eye(len(obs)))
            Sigma_inv = L_O_inv.T @ L_O_inv
            lam_min = float(np.linalg.eigvalsh(A_O).min())
            blocks.append(PatternBlock(
                obs_idx = obs,
                col_idx = np.array(cols, dtype=int),
                L_inv   = L_O_inv,
                Sigma_inv = Sigma_inv,
                lam_min = lam_min,
            ))
        return blocks
