# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import time
import logging
from .objectives import NNMObjective, NNMSparseObjective, NNMCorrObjective
from .svd import nuclear_norm_oracle
from .projections import EuclideanProjection
from .result import History, FitResult
from .config import SolverConfig
from .state import IterState, StopReason
from ..utils.logs import get_loglevel, CustomLogger
from ..utils.sampling_covariance import SamplingCovariance


class FrankWolfe():

    """
    Frank-Wolfe algorithms for low rank matrix approximation
    using constraints on nuclear norm and/or l1 norm of the matrix.

    Parameters
    ----------
    max_iter : integer, default=1000
        Maximum number of iterations allowed for the FW algorithms.

    svd_method : string, default='power'
        The method for finding the top SVD component at every iteration 
        of the FW algorithm. See TopCompSVD for available options.

    svd_max_iter : integer, default=None
        Maximum number of iterations allowed for the SVD method (within each 
        iteration of the FW method). If `None`, it is set adaptively at every step,
        estimated from the iteration number.

    stop_criteria : list<string>, default=['duality_gap', 'step_size', 'relative_loss']
        A list of stop criteria to be used for the FW algorithm. Options include:
            'duality_gap': Stop the algorithm if the duality gap becomes
                less than the threshold tolerance 'tol'. A low duality gap
                indicates that the current solution is close to the global minimum.
            'step_size': Stop the algorithm if the step size becomes less
                than the threshold 'step_tol'. A low step size indicates that
                the update would be negligible.
            'relative_loss': Stop the algorithm if the change in objective
                function in successive iterations is less than 'rel_tol'. A low value
                indicates that the updates are negligible.
            'relative_dg': Stop the algorithm if the change in duality gap
                in successive iterations is less then 'rel_tol'.

    model : string, default='nnm'
        The model used for the FW algorithm. Options include:
            'nnm'        : Nuclear Norm constraint
            'nnm-sparse' : Nuclear Norm + L1 constraint
            'nnm-corr'   : Nuclear Norm with correlated error

    simplex_method : string, default='sort'
        The method used for the simplex projection of the l1 ball. See 
        EuclideanProjection for available options.

    tol : float, default=1e-3
        Threshold tolerance for duality gap, see 'stop_criteria'.

    step_tol : float, default=1e-3
        Threshold tolerance for the step size, see 'stop_criteria'.

    rel_tol : float, default=1e-8
        Threshold for the change in objective function in successive iterations,
        see 'stop_criteria'.

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

    print_skip : integer, default=None
        Number of steps skipped between each printed step if `verbose = 1`.
        If set to None, it is automatically estimated from `max_iter`.

    """
    def __init__(self, max_iter = 1000,
            svd_method = 'power', svd_max_iter = None,
            stop_criteria = ['duality_gap', 'step_size', 'relative_loss', 'relative_dg'],
            model = 'nnm', simplex_method = 'sort',
            tol = 1e-3, step_tol = 1e-3, rel_tol = 1e-8,
            verbose = 1, print_skip = None):

        self.model_ = model

        self.cfg_ = SolverConfig(
            max_iter       = max_iter,
            svd_method     = svd_method,
            svd_max_iter   = svd_max_iter,
            stop_criteria  = tuple(stop_criteria),
            tol            = tol,
            step_tol       = step_tol,
            rel_tol        = rel_tol,
            simplex_method = simplex_method,
        )
                
        self.prog_step_skip_ = print_skip
        if verbose > 0 and self.prog_step_skip_ is None:
            if verbose == 1:
                self.prog_step_skip_ = max(1, int(self.cfg_.max_iter / 100)) * 10
            elif verbose > 1:
                self.prog_step_skip_ = 1

        # Logger
        loglevel = get_loglevel(verbose)
        self.logger_ = CustomLogger(__name__, level = loglevel)
        self.logger_.override_subsystem_loglevel(loglevel)

        return


    @property
    def result(self):
        return self.result_


    @property
    def solver_config(self):
        return self.cfg_

    @property
    def objective(self):
        return self.obj_


    # ------------------------------------------------------------------
    # Objective factory
    # ------------------------------------------------------------------
 
    def _make_objective(self, Y, radius, sparse_scale, mask, weight, noise_cov):
        """
        Construct and return the objective instance for the requested model.
 
        For 'nnm-sparse', r may be a (r_nuc, l1_multiplier) tuple or a
        scalar (l1_multiplier defaults to 1.0).
        """
        if self.model_ == 'nnm':
            return NNMObjective(Y, radius, mask = mask, weight = weight)
 
        elif self.model_ == 'nnm-sparse':
            l1_mult = 1.0 if sparse_scale is None else sparse_scale
            return NNMSparseObjective(
                Y, radius, l1_mult,
                mask = mask, weight = weight,
                simplex_method = self.cfg_.simplex_method,
            )
 
        elif self.model_ == 'nnm-corr':
            return NNMCorrObjective(
                Y, radius, noise_cov,
                mask = mask, weight = weight,
            )
 
        else:
            raise ValueError(
                f"Unknown model '{self.model_}'. "
                "Choose from 'nnm', 'nnm-sparse', 'nnm-corr'."
            )
 
 
    # ------------------------------------------------------------------
    # SVD budget
    # ------------------------------------------------------------------
 
    def _get_svd_max_iter(self, svd_max_iter, iter_state):
        """
        Resolve the SVD iteration budget for the current call.
 
        Priority: explicit argument > class-level setting > adaptive schedule.
        The adaptive schedule grows slowly with iteration count.
        """
        if svd_max_iter is not None:
            return svd_max_iter
        if self.cfg_.svd_max_iter is not None:
            return self.cfg_.svd_max_iter
        adaptive = 10 + int(iter_state.istep / 20)
        return min(adaptive, 100)


    # ------------------------------------------------------------------
    # Oracles + duality gap computation
    # ------------------------------------------------------------------

    def _oracle_l1norm(self, G):
        '''
        Linear optimization oracle,
        where the feasible region is a l1 norm ball for some r
        '''
        idx = np.unravel_index(np.argmax(np.abs(G)), G.shape)
        S = np.zeros_like(G)
        sgn = np.sign(G[idx])
        if sgn != 0:
            S[idx] = - self.obj_.l1_threshold_ * sgn
        return S


    def _oracle_nnm(self, G, *, iter_state, warm_start_uv = False, svd_max_iter = None):
        '''
        Compute the FW linear oracle, descent direction D, and duality gap
        for the 'nnm' and 'nnm-corr' models.
 
        Uses the convention D = X - S so that the update is:
            X_new = X - step * D  =  (1 - step) * X + step * S
 
        Parameters
        ----------
        G : np.ndarray
            Current gradient.
        iter_state: IterState
            Current iterate.
        warm_start_uv : bool
            Whether to warm-start the SVD power iteration.
        svd_max_iter : int or None
            Maximum iterations for the SVD solver.
 
        Returns
        -------
        S : np.ndarray  - FW vertex (nuclear-norm oracle output)
        D : np.ndarray  - descent direction  (X - S)
        dg : float      - duality gap
        '''

        max_iter = self._get_svd_max_iter(svd_max_iter, iter_state)

        S, u1, v1_t, n_iter = nuclear_norm_oracle(
            G, self.obj_.radius_,
            method = self.cfg_.svd_method,
            max_iter = max_iter,
            warm_start = warm_start_uv,
            u0 = iter_state.svd_u, 
            v0 = iter_state.svd_vt
        )

        self.logger_.debug(f"SVD: max_iter = {max_iter}, performed = {n_iter}")
        if iter_state.svd_u is not None:
            self.logger_.debug(
                f"     u1 norm change = {np.linalg.norm(u1 - iter_state.svd_u):g}, "
                f"v1t norm change = {np.linalg.norm(v1_t - iter_state.svd_vt):g}"
            )

        iter_state.svd_u = u1
        iter_state.svd_vt = v1_t
        iter_state.svd_n_iter = n_iter

        D  = iter_state.X - S
        dg = np.sum(D * G)   # equivalent to trace(D.T @ G), avoids full matmul
        return S, D, dg
 
 
    def _oracle_nnm_sparse(self, G, *, iter_state, warm_start_uv = False, svd_max_iter = None):
        '''
        Compute the FW linear oracle, descent directions DL / DM, and duality gap
        for the 'nnm-sparse' model.
 
        Uses the convention DL = X - SL, DM = M - SM so that the updates are:
            X_new = X - step * DL  =  (1 - step) * X + step * SL
            M_new = M - step * DM  =  (1 - step) * M + step * SM
 
        Parameters
        ----------
        G : np.ndarray
            Current gradient.
        iter_state: IterState
            Current iterate.
        warm_start_uv : bool
            Whether to warm-start the SVD power iteration.
        svd_max_iter : int or None
            Maximum iterations for the SVD solver.
 
        Returns
        -------
        SL : np.ndarray  - nuclear-norm oracle output
        SM : np.ndarray  - l1-norm oracle output
        DL : np.ndarray  - low-rank descent direction  (X - SL)
        DM : np.ndarray  - sparse descent direction    (M - SM)
        dg : float       - duality gap
        '''

        SL = self._oracle_nnm(G,
            iter_state = iter_state,
            warm_start_uv = warm_start_uv, 
            svd_max_iter = svd_max_iter)[0]
        SM = self._oracle_l1norm(G)
        DL = iter_state.X - SL
        DM = iter_state.M - SM
        dg = np.sum(DL * G) + np.sum(DM * G)
        return SL, SM, DL, DM, dg
 
 
    # ------------------------------------------------------------------
    # Negative duality-gap guard
    # ------------------------------------------------------------------
 
    def _try_positive_dg(self, oracle_fn, G, *, iter_state):
        '''
        Call ``oracle_fn(G, *args, warm_start_uv=False, svd_max_iter=...)``
        and, if the returned duality gap is negative, retry with progressively
        larger SVD iteration budgets (power method only).

        A negative duality gap indicates inaccurate singular vectors.
        Retrying with more power iterations (for method='power') typically
        resolves the issue.
 
        ``oracle_fn`` must return a tuple whose **last element is dg**.
 
        Parameters
        ----------
        oracle_fn : callable
            One of ``_oracle_nnm`` or ``_oracle_nnm_sparse``.
        G : np.ndarray
            Current gradient passed as the first positional argument to oracle_fn.
        iter_state : IterState
            Current state of the iteration, forwarded to oracle_fn.
 
        Returns
        -------
        result : tuple
            The return value of oracle_fn (possibly from a retry call).
        '''
        result = oracle_fn(G,
            iter_state = iter_state,
            warm_start_uv = False,
            svd_max_iter  = self.cfg_.svd_max_iter)
        dg = result[-1]
        if dg >= 0:
            return result
 
        self.logger_.warn(f"Iteration {iter_state.istep}. Duality gap < 0 ({dg:g}).")
        if self.cfg_.svd_method != 'power':
            return result
 
        self.logger_.warn("Retrying SVD power iteration with larger budget.")
        svd_max_iter = self.cfg_.svd_max_iter * 2 if self.cfg_.svd_max_iter is not None else 100
        for n_rep in range(1, 6):

            # Use fresh random initialization (warm_start_uv=False) so we don't
            # get trapped in the basin of a non-dominant singular vector.
            result = oracle_fn(G,
                iter_state = iter_state,
                warm_start_uv = False,
                svd_max_iter  = svd_max_iter)
            dg = result[-1]
            self.logger_.warn(
                f"Power iteration trial {n_rep}. "
                f"dg = {dg:g}, n_iter = {iter_state.svd_n_iter}, max_iter = {svd_max_iter}"
            )
            if dg > 0:
                break
            svd_max_iter *= 2
 
        if dg < 0:
            self.logger_.warn(
                f"Power iteration could not recover positive dg ({dg:g})."
            )
        return result

    # ------------------------------------------------------------------
    # Step size
    # ------------------------------------------------------------------
 
    def _compute_step_size(self, dg, D, iter_state):
        """
        Exact line-search step size  gamma = dg / step_denom(D), clamped to [0, 1].
 
        The denominator is model-specific and computed by the objective.
 
        Parameters
        ----------
        dg : float
            Duality gap (numerator of the line search).
        D : np.ndarray
            Descent direction.
        iter_state: IterState
            Required for last step size.
 
        Returns
        -------
        ss : float   step size in (0, 1].
        """
        denom = self.obj_.step_denom(D)

        if not np.isfinite(denom) or denom == 0.0:
            self.logger_.warn(f"Step size denominator is {denom:g} at iteration {iter_state.istep}. Returning zero step.")
            return 0.0

        ss    = min(dg / denom, 1.0)
        if ss < 0:
            self.logger_.warn(f"Step size < 0 ({ss:g}). Using last valid step size.")
            ss = iter_state.last_step_size
        return ss
 
 
    # ------------------------------------------------------------------
    # Frank-Wolfe step functions
    # ------------------------------------------------------------------
 
    def _fw_one_step_nnm(self, iter_state):
        '''
        Single Frank-Wolfe step for the 'nnm' and 'nnm-corr' models.
 
        Parameters
        ----------
        X : np.ndarray - current iterate.

        Updates
        -------
        iter_state : IterState
 
        Returns
        -------
        G    : np.ndarray - gradient at X
        dg   : float      - duality gap
        step : float      - step size used
        '''
        # 1. Gradient
        G = self.obj_.gradient(iter_state.X)
        # 2. Linear oracle + descent direction + duality gap (with negative-dg guard)
        S, D, dg = self._try_positive_dg(self._oracle_nnm, G, iter_state = iter_state)
        # 3. Step size
        step_size = self._compute_step_size(dg, D, iter_state)
        # 4. Update:  X - step*(X - S) = (1-step)*X + step*S
        iter_state.X = iter_state.X - step_size * D
        iter_state.last_step_size = step_size
        return G, dg, step_size
 
 
    def _fw_one_step_nnm_sparse(self, iter_state):
        '''
        Single Frank-Wolfe step for the 'nnm-sparse' model.
 
        Parameters
        ----------
        X : np.ndarray - current low-rank iterate.
        M : np.ndarray - current sparse iterate.
 
        Updates
        -------
        iter_state : IterState
 
        Returns
        -------
        G    : np.ndarray - gradient at X + M
        dg   : float      - duality gap
        step : float      - step size used
        '''
        G = self.obj_.gradient(iter_state.X + iter_state.M)
        SL, SM, DL, DM, dg = self._try_positive_dg(self._oracle_nnm_sparse, G, iter_state = iter_state)

        # v1.0
        #step_size    = self._compute_step_size(dg, DL, iter_state)
        #iter_state.X = iter_state.X - step_size * DL
        #Mnew         = iter_state.M - step_size * DM
        #G_half       = self.obj_.gradient(iter_state.X + Mnew)
        #iter_state.M = self._proj_l1ball(Mnew - G_half)

        # v2.0
        step_size = self._compute_step_size(np.sum(DL * G), DL, iter_state)
        iter_state.X = iter_state.X - step_size * DL
        iter_state.M = self.obj_.project_sparse(iter_state.X)

        return G, dg, step_size


    #def _proj_l1ball(self, X):
    #    n, p = X.shape
    #    ep = EuclideanProjection(method = self.cfg_.simplex_method, target = 'l1')
    #    ep.fit(self.obj_._masked(X).ravel(), a = self.obj_.l1_threshold_)
    #    return ep.proj.reshape(n, p)


    # ------------------------------------------------------------------
    # Stopping criterion
    # ------------------------------------------------------------------

    def _do_stop(self, fx_hist, dg_hist, st_hist, iter_state):

        def _safe_relative_change(x1, x0, eps=1e-8):
            '''
            Relative change when |x0| >= 1,
            absolute change when |x0| < 1.
            When |x0| < 1, it becomes |x1 - x0|
            '''
            if not np.isfinite(x1) or not np.isfinite(x0):
                return np.inf
            # Hybrid relative / absolute
            scale = max(1.0, np.abs(x0))
            # Symmetric relative change for more stable stopping near zero
            # scale = max(0.5 * (abs(x1) + abs(x0)), eps)
            xr = np.abs((x1 - x0) / scale)
            return xr

        if 'duality_gap' in self.cfg_.stop_criteria:
            dg = dg_hist[-1]
            if np.abs(dg) <= self.cfg_.tol:
                iter_state.stop_reason = StopReason.DUALITY_GAP
                return True

        if 'step_size' in self.cfg_.stop_criteria:
            ss = st_hist[-1]
            if ss <= self.cfg_.step_tol:
                iter_state.stop_reason = StopReason.STEP_SIZE
                return True

        if 'relative_loss' in self.cfg_.stop_criteria:
            fx = fx_hist[-1]
            fx0 = fx_hist[-2]
            # fx_rel = np.abs((fx - fx0) / fx0)
            fx_rel = _safe_relative_change(fx, fx0)
            if fx_rel <= self.cfg_.rel_tol:
                iter_state.stop_reason = StopReason.RELATIVE_LOSS
                return True

        if 'relative_dg' in self.cfg_.stop_criteria:
            dg = dg_hist[-1]
            dg0 = dg_hist[-2]
            # dg_rel = np.abs((dg - dg0) / dg0)
            dg_rel = _safe_relative_change(dg, dg0)
            if dg_rel <= self.cfg_.rel_tol:
                iter_state.stop_reason = StopReason.RELATIVE_DG
                return True

        return False


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def fit(self, Y, radius,
            sparse_scale = None,
            weight = None, mask = None, X0 = None,
            noise_cov = None):

        '''
        Wrapper function for the low rank matrix approximation (LRMA)
        using the Frank-Wolfe (FW) algorithm.

        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix. May have missing values as np.nan.

        radius : float
            Nuclear norm constraint radius.

        sparse_scale: float, default=1.0
            Multiplier to determine the threshold sparse radius (constraint on 
            l1 norm). Used for model='nnm-sparse'.

        weight : (optional) np.ndarray [size (n, p); dtype: float], default=None
            An array of weights for each element in the input matrix Y.

        mask : (optional) np.ndarray [size (n, p); dtype: boolean], default=None
            An array of masks to hide specific elements in the input matrix Y.
            True for entries to exclude. Internally, it is combined with 
            all NaN entries.

        X0 : (optional) np.ndarray [size (n, p); dtype: float], default=None
            Optional initial guess for the low rank matrix in the FW algorithm. 
            Defaults to zeros.

        noise_cov : np.ndarray [size (n, n); dtype: float], default=None
            Sampling covariance matrix A. Required for model='nnm-corr'.
            Must be created / validated from SamplingCovariance, e.g.:
                A = SamplingCovariance.from_matrix(raw_cov_from_ldsc)
                A = SamplingCovariance.from_ldsc(...)


        Note
        ----
        Any NaN value in the input matrix is automatically added to `mask`.


        Usage
        -----
            model = FrankWolfe(<params>)
            model.fit(Y, r, ...)
        '''

        self.logger_.debug(f"Model = {self.model_}")
        self.logger_.debug(f"Shape of input data = {Y.shape}")

        # ---------------------
        # Validations
        # ---------------------
        if self.model_ == 'nnm-corr' and noise_cov is None:
            raise ValueError(
                "model='nnm-corr' requires a covariance argument. "
                "Construct one via SamplingCovariance.from_matrix(A)."
            )
            if isinstance(noise_cov, np.ndarray): noise_cov = SamplingCovariance.from_matrix(noise_cov)

        # ---------------------
        # Build objective
        # ---------------------
        self.obj_ = self._make_objective(Y, radius, sparse_scale, mask, weight, noise_cov)

        self.logger_.debug(
            f"Masked entries = {np.sum(self.obj_.mask_) if self.obj_.mask_ is not None else 0}"
        )

        # ---------------------
        # Initialize iterates
        # ---------------------
        X = np.zeros_like(self.obj_.Y_) if X0 is None else X0.copy()
        #M = np.zeros_like(self.obj_.Y_) if self.model_ == 'nnm-sparse' else None
        M = self.obj_.project_sparse(X) if self.model_ == 'nnm-sparse' else None
        iter_state = IterState(X=X, M=M)

        # ---------------------
        # Initialize history
        # ---------------------
        fx_hist = [self.obj_.value(iter_state.X)]
        dg_hist = [np.inf]
        st_hist = [1.0]
        cpu_time_hist = [1e-8] # do not use 0 to avoid log10 error.

        iter_state.istep = 0

        if self.model_ == 'nnm-sparse':
            fx_hist = [self.obj_.value(iter_state.X + iter_state.M)]
            fm_hist = [self.obj_.value(iter_state.M)]
            fl_hist = [self.obj_.value(iter_state.X)]

        cpu_time_old = time.process_time()

        # ---------------------
        # Steps 1, ..., max_iter
        # ---------------------
        for i in range(self.cfg_.max_iter):

            iter_state.istep += 1
            self.logger_.debug(f"Iteration {iter_state.istep}.")

            if self.model_ in ('nnm', 'nnm-corr'):
                G, dg, step_size = self._fw_one_step_nnm(iter_state)
                fx = self.obj_.value(iter_state.X)

            elif self.model_ == 'nnm-sparse':
                G, dg, step_size = self._fw_one_step_nnm_sparse(iter_state)
                fx = self.obj_.value(iter_state.X + iter_state.M)
                fm = self.obj_.value(iter_state.M)
                fl = self.obj_.value(iter_state.X)

            self.logger_.debug(f"Step size {step_size:.3f}. Duality Gap {dg:g}.")

            # Append iteration history
            cpu_time = time.process_time()
            cpu_time_hist.append(cpu_time - cpu_time_old)
            cpu_time_old = cpu_time
            fx_hist.append(fx)
            dg_hist.append(dg)
            st_hist.append(step_size)

            if self.model_ == 'nnm-sparse':
                fm_hist.append(fm)
                fl_hist.append(fl)

            if self.prog_step_skip_ is not None:
                if (iter_state.istep == 1) or (iter_state.istep % self.prog_step_skip_ == 0):
                    self.logger_.info(f"Iteration {iter_state.istep}. Step size {step_size:.3f}. Duality Gap {dg:g}")

            if self._do_stop(fx_hist, dg_hist, st_hist, iter_state):
                break


        self.logger_.info(
            f"Stopping at iteration {iter_state.istep}. "
            f"Step size {step_size:.3f}. Duality Gap {dg:g}")
        self.logger_.info(iter_state.stop_reason.message)

        history = History(
            loss          = fx_hist,
            duality_gap   = dg_hist,
            step_size     = st_hist,
            cpu_time      = cpu_time_hist,
            loss_sparse   = fm_hist if self.model_ == 'nnm-sparse' else None,
            loss_low_rank = fl_hist if self.model_ == 'nnm-sparse' else None,
        )

        self.result_  = FitResult(
            X           = iter_state.X,
            M           = iter_state.M if self.model_ == 'nnm-sparse' else None,
            history     = history,
            n_iter      = iter_state.istep,
            converged   = iter_state.stop_reason != StopReason.MAX_ITER,
            stop_reason = iter_state.stop_reason,
            message     = iter_state.stop_reason.message,
            metrics     = {
                'nuclear_norm': float(np.linalg.norm(iter_state.X, 'nuc')),
                #'l1_norm': float(np.linalg.norm(iter_state.M, 1)) if self.model_ == 'nnm-sparse' else None,
                'l1_norm': float(np.sum(np.abs(iter_state.M))) if self.model_ == 'nnm-sparse' else None,
                'radius' : radius,
                'sparse_scale' : sparse_scale,
                'l1_threshold' : self.obj_.l1_threshold_ if self.model_ == 'nnm-sparse' else None,
            },
        )

        return self
