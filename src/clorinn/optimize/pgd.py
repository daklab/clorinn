# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import logging
from .objectives import NNMObjective
from .projections import NuclearNormProjection
from ..utils.logs import CustomLogger

class PGDWarmStart():
    """
    Adaptive projected gradient descent for warm-starting Frank-Wolfe
    on the nuclear-norm-constrained matrix completion problem.

    Solves
 
        min   f(X)
        s.t.  ||X||_* <= r
 
    where f(X) is the objective defined by an NNMObjective. PGD iterates until 
    either
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
        Algorithm for the simplex sub-projection inside the nuclear norm
        ball projection. See EuclideanProjection for available options.
 
    show_progress : boolean, default=False
        Print iteration progress at each step (or every print_skip steps).
 
    print_skip : integer, default=10
        Number of steps skipped between each printed step
        if `show_progress = True`.

    debug : boolean, default=False
        Set log level to DEBUG for this instance.
 
    suppress_warnings : boolean, default=True
        Suppress WARNING-level messages.  Set to False to see convergence
        warnings.
    """
 
    def __init__(self, max_iter = 50, rel_tol = 1e-6,
                 simplex_method = 'sort',
                 show_progress = False,
                 print_skip = 1,
                 debug = False,
                 suppress_warnings = True):
        self.max_iter_       = max_iter
        self.rel_tol_        = rel_tol
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
        self.logger_.override_global_default_loglevel(loglevel)
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
        self: PGDWarmStart
            Fitted instance. Access the result via properties.
        """
        obj = NNMObjective(Y, r, mask = mask, weight = weight)
        eta = obj.pgd_step_size

        projector = NuclearNormProjection(simplex_method = self.simplex_method_)
        X = np.zeros_like(obj.Y_)
        fx_old = np.inf
        self.fx_list_ = []
        self.converged_interior_ = False
 
        for t in range(self.max_iter_):

            self.n_iter_ = t + 1

            fx = obj.value(X)
            self.fx_list_.append(fx)

            G = obj.gradient(X)
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
        return self
