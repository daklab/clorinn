# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
from .objectives import NNMObjective
from .projections import NuclearNormProjection
from .result import History, FitResult
from .state import StopReason
from ..utils.logs import get_loglevel, CustomLogger

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
 
    print_skip : integer, default=None
        Number of steps skipped between each printed step
        if `verbose = 1`.

    verbose : int, default=1
        Controls the verbosity of solver output.  Three levels are
        recognised:
            0  Silent.  Only warnings and errors are reported.
               Equivalent to logging.WARN.
            1  Progress.  Equivalent to logging.INFO.
               Logs iteration count, step size, and duality gap at convergence.
            2  Debug.  Equivalent to logging.DEBUG.
               Logs all debugging outputs.
        Higher values are treated as 2.
    """
 
    def __init__(self, max_iter = 50, rel_tol = 1e-6,
                 simplex_method = 'sort',
                 print_skip = None,
                 verbose = 1):
        self.max_iter_       = max_iter
        self.rel_tol_        = rel_tol
        self.simplex_method_ = simplex_method
        self.prog_step_skip_     = print_skip

        self.prog_step_skip_ = print_skip
        if verbose > 0 and self.prog_step_skip_ is None:
            if verbose == 1:
                self.prog_step_skip_ = max(1, int(self.max_iter_ / 10))
            elif verbose > 1:
                self.prog_step_skip_ = 1

        # set logger for this class
        loglevel = get_loglevel(verbose)
        self.logger_ = CustomLogger(__name__, level = loglevel)
        self.logger_.override_subsystem_loglevel(loglevel)
        return
 
 
    @property
    def result(self):
        return self.result_
 
 
    def fit(self, Y, radius, mask = None, weight = None, noise_cov = None):
        """
        Run PGD warm start.
 
        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix (NaN values should already be replaced by 0).
 
        radius : float
            Nuclear norm radius.
 
        mask : np.ndarray [size (n, p); dtype: bool] or None
            True for entries to ignore (NaN or held out).
 
        weight : np.ndarray [size (n, p); dtype: float] or None
            Per-element weights.

        noise_cov : np.ndarray [size (n, n); dtype: float], default=None
            Sampling covariance matrix A. Required for model='nnm-corr'.
            Must be created / validated from SamplingCovariance, e.g.:
                A = SamplingCovariance.from_matrix(raw_cov_from_ldsc)
                A = SamplingCovariance.from_ldsc(...)
 
        Returns
        -------
        self: PGDWarmStart
            Fitted instance. Access the result via properties.
        """
        obj = NNMObjective(Y, radius, mask = mask, weight = weight)
        eta = obj.pgd_step_size

        projector = NuclearNormProjection(simplex_method = self.simplex_method_)
        X = np.zeros_like(obj.Y_)
        fx_old = np.inf
        fx_hist = []
        converged_interior = False
        stop_reason = StopReason.MAX_ITER
 
        for t in range(self.max_iter_):

            n_iter = t + 1

            fx = obj.value(X)
            fx_hist.append(fx)

            G = obj.gradient(X)
            X_candidate = X - eta * G

            # Project onto nuclear norm ball
            projector.fit(X_candidate, radius)
            X = projector.proj
 
            if self.prog_step_skip_ is not None:
                if (n_iter == 1) or (n_iter % self.prog_step_skip_ == 0):
                    nuc = projector.nuclear_norm_after_
                    tag = "clipped" if projector.is_clipped else "interior"
                    self.logger_.info(
                        f"PGD iter {n_iter:4d}  f = {fx:.4f}  "
                        f"||X||_* = {nuc:.1f}  ({tag})"
                    )

            # Stop if projection clipped: constraint is active, hand off to FW
            # if projector.is_clipped:
            #     if self.prog_step_skip_ is not None:
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
                    if self.prog_step_skip_ is not None:
                        nuc = projector.nuclear_norm_after_
                        self.logger_.info(
                            f"PGD iter {n_iter:4d}  Converged in interior "
                            f"(||X||_* = {nuc:.1f} < r = {radius:.1f})"
                        )
                    stop_reason = StopReason.RELATIVE_LOSS
                    converged_interior = True
                    break
 
            fx_old = fx

        self.result_  = FitResult(
            X         = X,
            history   = History(loss = fx_hist),
            n_iter    = n_iter,
            stop_reason = stop_reason,
            converged = converged_interior,
            message   = 'Converged in interior' if converged_interior else 'Max iterations',
            metrics   = {
                'nuclear_norm': float(np.linalg.norm(X, 'nuc')),
                'radius': radius,
                },
        )

        return self
