"""
The orthogonal projection onto the unit simplex
is a convex optimization problem.
The unit simplex is defined by,
    S = {x ∈ R^n ∣ x ⪰ 0, ∑x = 1}
Orthogonal projection onto the unit simplex is given by:
    argmin_x 0.5 || y - x ||_2^2
    subject to x ∈ S.
Several algorithms exist for this optimization problem.

Laurent Condat [1] presents a review and comparison 
of existing algorithms with a new proposal for 
projection onto the unit simplex. 
This paper lists the worst-case complexity 
and empirical complexity of those algorithms, 
and presents concise pseudo-code for many algorithms.
Condat has included C and MATLAB implementations 
of all the algorithms mentioned in his paper
here: https://lcondat.github.io/software.html

[1] Condat, L. Fast projection onto the simplex and the 𝑙1 ball. 
    Math. Program. 158, 575–585 (2016). 
    https://doi.org/10.1007/s10107-015-0946-6
"""
# Author: Saikat Banerjee

import numpy as np
import logging
from ..utils.logs import CustomLogger

class EuclideanProjection():

    def __init__(self, method = 'condat', target = 'simplex'):
        self.method_ = method
        self.target_ = target
        return


    @property
    def proj(self):
        return self.yproj_


    def fit(self, y, a = 1.0):
        #
        if self.target_ == 'simplex':
            assert np.all(y >= 0)
            ypos = y
        elif self.target_ == 'l1':
            ypos = np.abs(y)
        """
        Main methods factory
        """
        if np.sum(ypos) == a:
            # best projection: itself
            yproj = ypos
        else:
            if self.method_ == 'sort':
                yproj = self._simplex_proj_sort(ypos, a)
            elif self.method_ == 'michelot':
                yproj = self._simplex_proj_michelot(ypos, a)
            elif self.method_ == 'condat':
                yproj = self._simplex_proj_condat(ypos, a)
        """
        For l1 projection, put the signs back
        """
        self.yproj_ = yproj
        if self.target_ == 'l1':
            self.yproj_ *= np.sign(y)
        return


    @staticmethod
    def l1_norm(y):
        return np.sum(np.abs(y))


    @staticmethod
    def _simplex_proj_sort(y, a):
        n = y.shape[0]
        u = np.sort(y)[::-1]
        ukvals = (np.cumsum(u) - a) / np.arange(1, n + 1)
        k = np.nonzero(ukvals < u)[0][-1]
        x = np.clip(y - ukvals[k], a_min=0, a_max=None)
        return x


    @staticmethod
    def _simplex_proj_michelot(y, a):
        auxv = y.copy()
        n = y.shape[0]
        rho = (np.sum(y) - a) / n
        vnorm_last = l1_norm(auxv)
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
        istep = 0
        while True:
            istep += 1
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
    Orthogonal projection onto the nuclear norm ball:
        { X ∈ R^{n×p}  |  ||X||_* <= r }
 
    The projection is obtained by taking the thin SVD of X,
    projecting the singular value vector onto the scaled simplex
    { s >= 0, sum(s) <= r }, and reconstructing.
 
    Parameters
    ----------
    method : string, default='sort'
        Method for the simplex sub-projection. See EuclideanProjection.
    """
 
    def __init__(self, method = 'sort'):
        self.method_ = method
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
            eucp = EuclideanProjection(method = self.method_, target = 'simplex')
            eucp.fit(s, a = r)
            self.s_ = eucp.proj
            #self.s_ = EuclideanProjection._simplex_proj_sort(s, r)
 
        self.U_  = U
        self.Vt_ = Vt
        self.Xproj_ = (U * self.s_) @ Vt
        self.nuclear_norm_before_ = nuc
        self.nuclear_norm_after_  = np.sum(self.s_)
        return
 
 
class PGDWarmStart():
    """
    Adaptive projected gradient descent for warm-starting Frank-Wolfe
    on the nuclear-norm-constrained matrix completion problem:
 
        min   0.5 * || P_Omega (X - Y) ||_F^2
        s.t.  ||X||_* <= r
 
    PGD iterates until either
        (a) the nuclear norm constraint becomes active (projection clips),
            at which point Frank-Wolfe's rank-1 updates are more natural, or
        (b) the objective stalls, meaning the solution lies in the interior
            and FW will confirm convergence via the duality gap.
 
    Parameters
    ----------
    max_iter : integer, default=50
        Maximum number of PGD iterations.
 
    rel_tol : float, default=1e-6
        Relative tolerance on the objective for detecting stall.
 
    simplex_method : string, default='sort'
        Method for the simplex sub-projection.
        See EuclideanProjection for available options.
 
    show_progress : boolean, default=False
        Print iteration progress.
 
    print_skip : integer, default=10
        Number of steps skipped between each printed step
        if `show_progress = True`.
    """
 
    def __init__(self, max_iter = 50, rel_tol = 1e-6,
                 simplex_method = 'sort',
                 show_progress = False,
                 print_skip = 1,
                 debug = False,
                 suppress_warnings = True):
        self.max_iter_ = max_iter
        self.rel_tol_  = rel_tol
        self.simplex_method_ = simplex_method
        self.show_progress_  = show_progress
        self.print_skip_     = print_skip
        # set logger for this class
        loglevel = None
        if debug: 
            loglevel = logging.DEBUG
        elif show_progress:
            loglevel = logging.INFO
        elif suppress_warnings:
            loglevel = logging.ERROR
        self.logger_ = CustomLogger(__name__, level = loglevel)
        self.suppress_warnings_ = suppress_warnings
        return
 
 
    @property
    def X(self):
        return self.X_
 
 
    @property
    def fx(self):
        return self.fx_list_
 
 
    @property
    def n_iter(self):
        return self.n_iter_
 
 
    @property
    def converged_in_interior(self):
        return self.converged_interior_
 
 
    def fit(self, Y, r, mask = None, weight = None):
        """
        Run PGD warm start.
 
        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix (NaN values should already be replaced by 0).
 
        r : float
            Nuclear norm radius.
 
        mask : np.ndarray [size (n, p); dtype: bool] or None
            True for entries to ignore (NaN or held out).
 
        weight : np.ndarray [size (n, p); dtype: float] or None
            Per-element weights.
 
        Returns
        -------
        X : np.ndarray [size (n, p); dtype: float]
            Feasible warm-start iterate with ||X||_* <= r.
        """
        n, p = Y.shape
        obs = np.ones((n, p), dtype=bool) if mask is None else ~mask
        wobs = None
        if weight is not None:
            wobs = np.where(obs, weight, 0.0)
 
        X = np.zeros_like(Y)
        projector = NuclearNormProjection(method = self.simplex_method_)
        fx_old = np.inf
        self.fx_list_ = []
        self.converged_interior_ = False
 
        # Step size = 1/L where L is the Lipschitz constant of the gradient.
        # Unweighted: L = 1. Weighted: L = max(w_ij^2).
        if wobs is None:
            eta = 1.0
        else:
            eta = 1.0 / np.max(np.square(wobs))
 
        for t in range(self.max_iter_):

            self.n_iter_ = t + 1

            # Objective: 0.5 * ||W * P_Omega(X - Y)||^2
            R = np.where(obs, X - Y, 0.0)
            if wobs is not None:
                R = wobs * R
            fx = 0.5 * np.linalg.norm(R, 'fro')**2
            self.fx_list_.append(fx)
 
            # Gradient (accounting for weights)
            G = np.where(obs, X - Y, 0.0)
            if wobs is not None:
                G = np.square(wobs) * G
 
            # Gradient step
            X_candidate = X - eta * G
 
            # Project onto nuclear norm ball
            projector.fit(X_candidate, r)
            X = projector.proj
 
            if self.show_progress_:
                if (self.n_iter_ == 1) or (self.n_iter_ % self.print_skip_ == 0):
                    nuc = projector.nuclear_norm_after_
                    tag = "clipped" if projector.is_clipped else "interior"
                    self.logger_.info(
                        f"PGD iter {self.n_iter_:4d}  f = {fx:.4f}  "
                        f"||X||_* = {nuc:.1f}  ({tag})"
                    )

            # Stop if projection clipped: constraint is active, hand off to FW
            # if projector.is_clipped:
            #     if self.show_progress_:
            #         self.logger_.info(
            #             f"PGD iter {self.n_iter_:4d}  Nuclear norm constraint active "
            #             f"({projector.nuclear_norm_before_:.1f} -> {r:.1f}). "
            #             f"Handing off to FW."
            #         )
            #     break
 
            # Stop if objective stalled: converged in interior
            if t > 0 and np.isfinite(fx_old):
                rel = abs(fx - fx_old) / max(1.0, abs(fx_old))
                if rel < self.rel_tol_:
                    if self.show_progress_:
                        nuc = projector.nuclear_norm_after_
                        self.logger_.info(
                            f"PGD iter {self.n_iter_:4d}  Converged in interior "
                            f"(||X||_* = {nuc:.1f} < r = {r:.1f})"
                        )
                    self.converged_interior_ = True
                    break
 
            fx_old = fx
 
        else:
            self.n_iter_ = self.max_iter_
 
        self.X_ = X
        return
