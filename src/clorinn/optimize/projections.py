"""
Euclidean projections onto convex sets used by Frank-Wolfe and PGD solvers.
Implements two classes:
 
  EuclideanProjection    Euclidean projection onto the simplex or l1 ball,
                         via three algorithms (sort, Michelot, Condat).
  NuclearNormProjection  Singular-value soft-thresholding for the nuclear-
                         norm ball, using EuclideanProjection internally.
 
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import logging

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Implementation classes
# ---------------------------------------------------------------------------

class EuclideanProjection():
    """
    Euclidean projection onto the unit simplex or the l1-norm ball.

    The unit simplex is defined as

        Delta = { x in R^n | x >= 0, sum(x) = a },

    and the l1-norm ball as

        B1 = { x in R^n | ||x||_1 <= a }.

    Projection onto the simplex solves

        min_{x in Delta}  0.5 * ||y - x||_2^2,

    which has a unique closed-form solution computable in O(n log n) time.
    Projection onto the l1 ball reduces to the simplex case after
    taking absolute values and restoring signs.

    Laurent Condat [1] presents a review and comparison
    of existing algorithms with a new proposal for
    projection onto the unit simplex.
    This paper lists the worst-case complexity
    and empirical complexity of those algorithms,
    and presents concise pseudo-code for many algorithms.
    Condat has included C and MATLAB implementations
    of all the algorithms mentioned in his paper
    here: https://lcondat.github.io/software.html

    Parameters
    ----------
    method : string, default='sort'
        Algorithm for the simplex sub-projection.  Options:

            'sort'      Sort-based algorithm.  O(n log n).  Numerically
                        robust; the default for all solvers.
            'condat'    Condat's algorithm [1].  O(n) average case.
            'michelot'  Michelot's iterative algorithm.  Slower in
                        practice; included for reference.

    target : string, default='simplex'
        Feasible set.  Either 'simplex' (non-negative entries summing to a)
        or 'l1' (l1-norm ball of radius a).

    Attributes
    ----------
    proj : np.ndarray [size (n,)]
        Projected vector.  Available after fit().

    References
    ----------
    .. [1] Condat, L. Fast projection onto the simplex and the l1 ball.
           Math. Program. 158, 575-585 (2016).
           https://doi.org/10.1007/s10107-015-0946-6
    """

    def __init__(self, method = 'condat', target = 'simplex'):
        self.method_ = method
        self.target_ = target
        return


    @property
    def proj(self):
        return self.yproj_


    def fit(self, y, a = 1.0):
        """
        Project vector y onto the simplex or l1 ball of radius a.

        Parameters
        ----------
        y : np.ndarray [size (n,); dtype: float]
            Input vector.  Must be non-negative when target='simplex'.
 
        a : float, default=1.0
            Radius of the target set.
        """

        if self.target_ == 'simplex':
            if not np.all(y >= 0):
                raise ValueError("Input vector must be non-negative for simplex projection.")
            ypos = y
        elif self.target_ == 'l1':
            ypos = np.abs(y)

        if np.sum(ypos) <= a:
            # Already feasible: projection is the input itself.
            yproj = ypos
        else:
            if self.method_ == 'sort':
                yproj = self._simplex_proj_sort(ypos, a)
            elif self.method_ == 'michelot':
                yproj = self._simplex_proj_michelot(ypos, a)
            elif self.method_ == 'condat':
                yproj = self._simplex_proj_condat(ypos, a)

        self.yproj_ = yproj

        # Restore signs for l1 projection
        if self.target_ == 'l1':
            self.yproj_ *= np.sign(y)
        return


    @staticmethod
    def l1_norm(y):
        return np.sum(np.abs(y))


    @staticmethod
    def _simplex_proj_sort(y, a):
        """
        Sort-based simplex projection.  O(n log n).
 
        Finds the unique threshold tau such that
            x_i = max(y_i - tau, 0)  and  sum(x) = a.
        """
        n = y.shape[0]
        u = np.sort(y)[::-1]
        ukvals = (np.cumsum(u) - a) / np.arange(1, n + 1)
        k = np.nonzero(ukvals < u)[0][-1]
        x = np.clip(y - ukvals[k], a_min=0, a_max=None)
        return x


    @staticmethod
    def _simplex_proj_michelot(y, a):
        """
        Michelot's iterative thresholding algorithm.
 
        Iteratively removes elements below the current threshold and
        recomputes until convergence.  Correct but slower than 'sort'
        in practice; included for reference.
        """
        auxv = y.copy()
        n = y.shape[0]
        rho = (np.sum(y) - a) / n
        vnorm_last = np.sum(np.abs(auxv))
        while True:
            allowed = auxv > rho
            auxv = auxv[allowed]
            nv = np.sum(allowed)
            vnorm = l1_norm(auxv)
            if vnorm == vnorm_last:
                break
            rho = (np.sum(auxv) - a) / nv
            vnorm_last = vnorm
        x = np.clip(y - rho, a_min = 0, a_max = None)
        return x


    @staticmethod
    def _simplex_proj_condat(y, a):
        """
        Condat's O(n) average-case simplex projection algorithm [1].
 
        A single-pass algorithm that maintains a running threshold
        estimate and corrects it in a final cleanup step.
        """
        auxv = np.array([y[0]])
        vtilde = np.array([])
        rho = y[0] - a
        # Step 2
        for yn in y[1:]:
            if yn > rho:
                rho += (yn - rho) / (auxv.shape[0] + 1)
                if rho > (yn - a):
                    auxv = np.append(auxv, yn)
                else:
                    vtilde = np.append(vtilde, auxv)
                    auxv = np.array([yn])
                    rho = yn - a
        # Step 3
        if vtilde.shape[0] > 0:
            for v in vtilde:
                if v > rho:
                    auxv = np.append(auxv, v)
                    rho += (v - rho) / (auxv.shape[0])                
        # Step 4
        nv_last = auxv.shape[0]
        while True:
            to_remove = list()
            nv_ = auxv.shape[0]
            for i, v in enumerate(auxv):
                if v <= rho:
                    to_remove.append(i)
                    nv_ = nv_ - 1
                    rho += (rho - v) / nv_
            auxv = np.delete(auxv, to_remove)
            nv = auxv.shape[0]
            assert nv == nv_
            if nv == nv_last:
                break
            nv_last = nv
        # Step 5
        x = np.clip(y - rho, a_min=0, a_max=None)
        return x


class NuclearNormProjection():
    """
    Euclidean projection onto the nuclear norm ball.

    Projects a matrix X onto the set

        { X ∈ R^{n×p}  |  ||X||_* <= r }

    by soft-thresholding its singular values.  Concretely, the thin SVD
    X = U diag(s) V^T is computed, the singular value vector s is projected
    onto the scaled simplex { q >= 0, sum(q) <= r } via EuclideanProjection,
    and the result is reconstructed as U diag(s_proj) V^T.
 
 
    Parameters
    ----------
    simplex_method : string, default='sort'
        Algorithm for the simplex sub-projection.
        See EuclideanProjection for available options.


    Attributes
    ----------
    proj : np.ndarray [size (n, p)]
        Projected matrix.  Available after fit().
 
    U : np.ndarray [size (n, k)]
        Left singular vectors from the thin SVD of the input.
 
    s : np.ndarray [size (k,)]
        Projected (soft-thresholded) singular values.
 
    Vt : np.ndarray [size (k, p)]
        Right singular vectors from the thin SVD of the input.
 
    is_clipped : bool
        True if the nuclear norm constraint was active, i.e. the input
        lay outside the ball and the singular values were clipped.
        False if the input was already feasible.
 
    Notes
    -----
    nuclear_norm_before_ and nuclear_norm_after_ are set after fit() and
    give the nuclear norms of the input and the projected output.
    """
 
    def __init__(self, simplex_method = 'sort'):
        self.simplex_method_ = simplex_method
        return
 
 
    @property
    def proj(self):
        return self.Xproj_
 
 
    @property
    def U(self):
        return self.U_
 
 
    @property
    def s(self):
        return self.s_
 
 
    @property
    def Vt(self):
        return self.Vt_
 
 
    @property
    def is_clipped(self):
        """Whether the projection had to clip (constraint was active)."""
        return self.is_clipped_
 
 
    def fit(self, X, r):
        """
        Project X onto the nuclear norm ball of radius r.
 
        Parameters
        ----------
        X : np.ndarray [size (n, p); dtype: float]
            Input matrix.
        r : float
            Nuclear norm radius.
        """
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        nuc = np.sum(s)
 
        if nuc <= r:
            self.is_clipped_ = False
            self.s_ = s
        else:
            self.is_clipped_ = True
            eucp = EuclideanProjection(method = self.simplex_method_, target = 'simplex')
            eucp.fit(s, a = r)
            self.s_ = eucp.proj
 
        self.U_  = U
        self.Vt_ = Vt
        self.Xproj_ = (U * self.s_) @ Vt
        self.nuclear_norm_before_ = nuc
        self.nuclear_norm_after_  = np.sum(self.s_)
        return
