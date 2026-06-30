# Author: Saikat Banerjee
# License: BSD 3 clause

import logging
import numpy as np

from .objectives import make_objective
from .projections import NuclearNormProjection
from .result import History, FitResult
from .state import StopReason
from ..utils.logs import configure_module_logger
from ..utils.sampling_covariance import SamplingCovariance 

class ProjectedGradientDescent():
    """
    Projected gradient descent algorithm for low rank matrix 
    approximation problem / matrix completion problem
    using constraints on nuclear norm and/or l1 norm of the matrix.

    Solves three problem variants depending on ``model'':
    'nnm', 'nnm-sparse' and 'nnm-corr'.

    PGD can be used standalone or as a warm start for Frank-Wolfe / AFW.
    As a warm start, include ``'boundary_active'`` in ``stop_criteria`` so
    that PGD halts as soon as the nuclear norm constraint becomes active and
    returns control to the caller.
 
    Parameters
    ----------
    model : str, default='nnm'
        Problem variant.  One of ``'nnm'``, ``'nnm-sparse'``,
        ``'nnm-corr'``.

    max_iter : integer, default=50
        Maximum number of PGD iterations.
 
    rel_tol : float, default=1e-6
        Relative tolerance on the objective for detecting stall.
 
    simplex_method : string, default='sort'
        Algorithm for the simplex sub-projection inside the nuclear norm
        ball projection. See EuclideanProjection for available options.

    stop_criteria : tuple of str, default=('relative_loss',)
        Stopping criteria to check at each iteration.  The solver halts
        at the first criterion that is satisfied.  Recognised values:
    
            'relative_loss'    Stop when the relative change in the
                               objective falls below rel_tol.  Indicates
                               convergence in the interior of the nuclear
                               norm ball.
            'boundary_active'  Stop when the nuclear norm projection clips
                               the singular values, i.e. ||X||_* = r.
                               Signals that the constraint is active and
                               the iterate is ready to be handed off to a
                               Frank-Wolfe solver.
    
        Pass an empty tuple to run for exactly max_iter iterations
        regardless of convergence.  For use as a warm start, include
        'boundary_active' so that PGD stops as soon as the boundary is
        reached and control is returned to the caller.

    print_skip : integer, default=None
        Number of steps skipped between each printed step
        if `verbose = 1`.

    verbose : int or None, default=None
        Controls the verbosity of solver output. If None, logging is handled by
        caller. If not None, override logging handlers of caller / installs new
        log handler only if no clorinn handler exists.
        Three levels are recognised:
            0     Silent.  Only warnings and errors are reported.
                  Equivalent to logging.WARN.
            1     Progress.  Equivalent to logging.INFO.
                  Logs iteration count, step size, and duality gap at convergence.
            2     Debug.  Equivalent to logging.DEBUG.
                  Logs all debugging outputs.
        Higher values are treated as 2.
    """
 
    def __init__(self,
                 model = 'nnm',
                 max_iter = 50, rel_tol = 1e-6,
                 simplex_method = 'sort',
                 stop_criteria=('relative_loss',),
                 print_skip = None,
                 verbose = None):

        self.logger_ = configure_module_logger(__name__, verbosity=verbose)

        self.model_          = model
        self.max_iter_       = max_iter
        self.rel_tol_        = rel_tol
        self.simplex_method_ = simplex_method
        self.stop_criteria_  = tuple(stop_criteria)
        self.prog_step_skip_ = print_skip

        if self.prog_step_skip_ is None and self.logger_.isEnabledFor(logging.INFO):
            if self.logger_.isEnabledFor(logging.DEBUG):
                self.prog_step_skip_ = 1
            else:
                self.prog_step_skip_ = max(1, int(self.max_iter_ / 100)) * 10

        return
 
 
    @property
    def result(self):
        """
        FitResult from the most recent fit() call.
        """
        return self.result_

    @property
    def objective(self):
        """
        Objective instance constructed during fit().
        """
        return self.obj_
 
 
    def fit(self, Y, radius,
        sparse_scale = None,
        mask = None, weight = None, 
        X0 = None,
        noise_cov = None):
        """
        Run projected gradient descent.
 
        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix. NaN values are replaced by zero internally.
 
        radius : float
            Nuclear norm constraint radius r.

        sparse_scale : float or None
            Multiplier to determine the constraint on l1 norm. 
            Used for model='nnm-sparse'. Defaults to 1.0 if None.
 
        mask : np.ndarray [size (n, p); dtype: bool] or None
            True for entries to exlcude from the loss (missing or held out).
 
        weight : np.ndarray [size (n, p); dtype: float] or None
            Per-element weights. None means unweighted.

        X0 : np.ndarray [size (n, p); dtype: float] or None
            Optional initial guess for the low rank matrix in the PGD algorithm.
            Defaults to zeros if None.

        noise_cov : SamplingCovariance 
                    or np.ndarray [size (n, n); dtype: float] or None 
            SamplingCovariance object. Required for model='nnm-corr'.
            If input is np.ndarray, it is automatically converted to
            SamplingCovariance object
                A = SamplingCovariance.from_matrix(noise_cov)
            See `utils/sampling_covariance.py`.
 
        Returns
        -------
        self: ProjectedGradientDescent
            Fitted instance. Access the result via ``result''.

        Note
        ----
        Any NaN value in the input matrix is automatically added to `mask`.


        Usage
        -----
            model = ProjectedGradientDescent(<params>)
            model.fit(Y, r, ...)
            result = model.result
        """

        self.logger_.debug(f"Model = {self.model_}")
        self.logger_.debug(f"Shape of input data = {Y.shape}")

        # ---------------------
        # Validate
        # ---------------------
        if self.model_ == 'nnm-corr' and noise_cov is None:
            raise ValueError(
                "model='nnm-corr' requires noise_cov. "
                "Pass a covariance matrix or a SamplingCovariance instance."
            )
        if isinstance(noise_cov, np.ndarray):
            noise_cov = SamplingCovariance.from_matrix(noise_cov)

        # ---------------------
        # Build objective
        # ---------------------
        self.obj_ = make_objective(self.model_, Y, radius, 
            sparse_scale = sparse_scale, 
            mask = mask, weight = weight, noise_cov = noise_cov,
            simplex_method = self.simplex_method_)

        self.logger_.debug(
            f"Masked entries = {np.sum(self.obj_.mask_) if self.obj_.mask_ is not None else 0}"
        )

        # ---------------------
        # Initialize iterates
        # ---------------------
        eta = self.obj_.pgd_step_size
        projector = NuclearNormProjection(simplex_method = self.simplex_method_)

        X = np.zeros_like(self.obj_.Y_) if X0 is None else X0.copy()
        M = self.obj_.project_sparse(X) if self.model_ == 'nnm-sparse' else None

        # ---------------------
        # Initialize history
        # ---------------------
        fx_old = self.obj_.value(X) if M is None else self.obj_.value(X + M)
        fx_hist = [fx_old]
        stop_reason = StopReason.MAX_ITER
        converged = False
        n_iter = 0
 
        # ---------------------
        # Steps 1, ..., max_iter
        # ---------------------
        for i in range(self.max_iter_):

            n_iter += 1

            # ---- Gradient step on X ----
            G = self.obj_.gradient(X) if M  is None else self.obj_.gradient(X+M)
            X_candidate = X - eta * G

            # ---- Project X onto nuclear norm ball ----
            projector.fit(X_candidate, radius)
            X = projector.proj

            # ---- Exact sparse update for NNM-Sparse (Algorithm 7) ----
            if M is not None:
                M = self.obj_.project_sparse(X)

            # ---- Evaluate objective ----
            fx = self.obj_.value(X) if M is None else self.obj_.value(X + M)
            fx_hist.append(fx)

            # ---- Progress logging ----
            if self.prog_step_skip_ is not None:
                if (n_iter == 1) or (n_iter % self.prog_step_skip_ == 0):
                    nuc = projector.nuclear_norm_after_
                    tag = "clipped" if projector.is_clipped else "interior"
                    self.logger_.info(
                        f"PGD iter {n_iter:4d}  f = {fx:.4f}  "
                        f"||X||_* = {nuc:.1f}  ({tag})"
                    )

            # ---- Stopping: boundary active ----
            if 'boundary_active' in self.stop_criteria_ and projector.is_clipped:
                self.logger_.info(
                    f"PGD iter {n_iter:4d}  Nuclear norm constraint active "
                    f"||X_candidate||_* = ({projector.nuclear_norm_before_:.1f} > r = {radius:.1f}). "
                    f"Handing off to FW."
                )
                stop_reason = StopReason.BOUNDARY_ACTIVE
                break
 
            # ---- Stopping: relative loss ----
            if 'relative_loss' in self.stop_criteria_ and n_iter > 1 and np.isfinite(fx_old):
                rel = abs(fx - fx_old) / max(1.0, abs(fx_old))
                if rel < self.rel_tol_:
                    if self.prog_step_skip_ is not None:
                        nuc = projector.nuclear_norm_after_
                        self.logger_.info(
                            f"PGD iter {n_iter:4d}  Converged in interior "
                            f"(||X||_* = {nuc:.1f} < r = {radius:.1f})"
                        )
                    stop_reason = StopReason.RELATIVE_LOSS
                    converged = True
                    break
 
            fx_old = fx

        # ---------------------
        # Assemble result
        # ---------------------
        history = History(
            loss = fx_hist,
        )

        metrics = {
            'nuclear_norm': float(np.linalg.norm(X, 'nuc')),
            'radius' : radius,
        }

        if self.model_ == 'nnm-sparse':
            metrics['l1_norm'] = float(np.sum(np.abs(M)))
            metrics['l1_threshold'] = float(self.obj_.l1_threshold_)
            metrics['sparse_scale'] = sparse_scale

        self.result_  = FitResult(
            X           = X,
            M           = M,
            history     = history,
            n_iter      = n_iter,
            converged   = converged,
            stop_reason = stop_reason,
            message     = stop_reason.message,
            metrics     = metrics,
        )

        return self
