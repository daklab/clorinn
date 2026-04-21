# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import time
import logging
from .objectives import NNMObjective, NNMSparseObjective, NNMCorrObjective
from .svd import nuclear_norm_oracle
from .projections import EuclideanProjection
from ..utils.logs import CustomLogger

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

    stop_criteria : list<string>, default=['duality_gap', 'step_size', 'relative_objective']
        A list of stop criteria to be used for the FW algorithm. Options include:
            'duality_gap': Stop the algorithm if the duality gap becomes
                less than the threshold tolerance 'tol'. A low duality gap
                indicates that the current solution is close to the global minimum.
            'step_size': Stop the algorithm if the step size becomes less
                than the threshold 'step_tol'. A low step size indicates that
                the update would be negligible.
            'relative_objective': Stop the algorithm if the change in objective
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

    show_progress : boolean, default=False
        Display progress of the FW algorithm. If true, then the step size and duality gap 
        is printed on the console after every few steps. The number of steps skipped 
        between each print is specified by 'print_skip'.

    print_skip : integer, default=None
        Number of steps skipped between each printed step if `show_progress = True`.
        If set to None, it is automatically estimated from `max_iter`.

    debug : boolean, default=False
        Whether to provide a verbose output for debugging the algorithm.

    suppress_warnings : boolean, default=False
        Whether to suppress the warnings. If True, the loglevel is set to ERROR.
        It can be used for a silent run.


    Debugging Notes
    ---------------
        The default project logging level is WARN. The logging level can be modified 
        using the following flags:
            - `debug = True` sets the logging level to DEBUG.
            - `show_progress = True` sets the logging level to INFO. 
            - `suppress_warnings = True` sets the logging level to ERROR.
        The lowest setting always takes precedence. For example, `debug = True, 
        show_progress = False` sets logging level to DEBUG. `debug = False, 
        show_progress = True, suppress_warnings = True` sets logging level to INFO.

    """
    def __init__(self, max_iter = 1000,
            svd_method = 'power', svd_max_iter = None,
            stop_criteria = ['duality_gap', 'step_size', 'relative_objective', 'relative_dg'],
            model = 'nnm', simplex_method = 'sort',
            tol = 1e-3, step_tol = 1e-3, rel_tol = 1e-8,
            show_progress = False, print_skip = None,
            debug = False, suppress_warnings = False):
                
        self.max_iter_ = max_iter
        self.model_ = model
        self.svd_method_ = svd_method
        self.svd_max_iter_ = svd_max_iter
        self.simplex_method_ = simplex_method
        self.stop_criteria_ = stop_criteria
        self.tol_ = tol
        self.step_size_tol_ = step_tol
        self.fxrel_tol_ = rel_tol
        self.show_progress_ = show_progress
        self.prog_step_skip_ = print_skip
        if self.show_progress_ and self.prog_step_skip_ is None:
            self.prog_step_skip_ = max(1, int(self.max_iter_ / 100)) * 10

        # Logger
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

        # Warm-start state for power-iteration SVD (tracked for logging)
        self.svd_u_prev_ = None
        self.svd_vt_prev_ = None
        self.svd_n_iter_ = None
        return


    @property
    def X(self):
        return self.X_


    @property
    def M(self):
        if self.model_ != 'nnm-sparse':
            raise AttributeError("M is only defined for model='nnm-sparse'.")
        return self.M_


    @property
    def duality_gaps(self):
        return self.dg_list_


    @property
    def fx(self):
        return self.fx_list_


    @property
    def steps(self):
        return self.st_list_

    
    @property
    def n_iter(self):
        return self.iter_num_


    # ------------------------------------------------------------------
    # Objective factory
    # ------------------------------------------------------------------
 
    def _make_objective(self, Y, r, mask, weight, L_inv, Sigma_inv):
        """
        Construct and return the objective instance for the requested model.
 
        For 'nnm-sparse', r may be a (r_nuc, l1_multiplier) tuple or a
        scalar (l1_multiplier defaults to 1.0).
        """
        if self.model_ == 'nnm':
            return NNMObjective(Y, r, mask = mask, weight = weight)
 
        elif self.model_ == 'nnm-sparse':
            if isinstance(r, tuple):
                r_nuc, l1_mult = r
            else:
                r_nuc, l1_mult = r, 1.0
            return NNMSparseObjective(
                Y, r_nuc, l1_mult,
                mask = mask, weight = weight,
                simplex_method = self.simplex_method_,
            )
 
        elif self.model_ == 'nnm-corr':
            return NNMCorrObjective(
                Y, r, L_inv, Sigma_inv,
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
 
    def _get_svd_max_iter(self, svd_max_iter):
        """
        Resolve the SVD iteration budget for the current call.
 
        Priority: explicit argument > class-level setting > adaptive schedule.
        The adaptive schedule grows slowly with iteration count.
        """
        if svd_max_iter is not None:
            return svd_max_iter
        if self.svd_max_iter_ is not None:
            return self.svd_max_iter_
        adaptive = 10 + int(self.iter_num_ / 20)
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


    def _oracle_nnm(self, G, X, warm_start_uv = False, svd_max_iter = None):
        '''
        Compute the FW linear oracle, descent direction D, and duality gap
        for the 'nnm' and 'nnm-corr' models.
 
        Uses the convention D = X - S so that the update is:
            X_new = X - step * D  =  (1 - step) * X + step * S
 
        Parameters
        ----------
        G : np.ndarray
            Current gradient.
        X : np.ndarray
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

        max_iter = self._get_svd_max_iter(svd_max_iter)

        S, u1, v1_t, n_iter = nuclear_norm_oracle(
            G, self.obj_.r_,
            method = self.svd_method_,
            max_iter = max_iter,
            warm_start = warm_start_uv,
            u0 = self.svd_u_prev_, 
            v0 = self.svd_vt_prev_
        )

        self.logger_.debug(f"SVD: max_iter = {max_iter}, performed = {n_iter}")
        if self.svd_u_prev_ is not None:
            self.logger_.debug(
                f"     u1 norm change = {np.linalg.norm(u1 - self.svd_u_prev_):g}, "
                f"v1t norm change = {np.linalg.norm(v1_t - self.svd_vt_prev_):g}"
            )

        self.svd_u_prev_ = u1
        self.svd_vt_prev_ = v1_t
        self.svd_n_iter_ = n_iter

        D  = X - S
        dg = np.sum(D * G)   # equivalent to trace(D.T @ G), avoids full matmul
        return S, D, dg
 
 
    def _oracle_nnm_sparse(self, G, X, M, warm_start_uv = False, svd_max_iter = None):
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
        X : np.ndarray
            Current low-rank iterate.
        M : np.ndarray
            Current sparse iterate.
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

        SL = self._oracle_nnm(G, X, 
            warm_start_uv = warm_start_uv, 
            svd_max_iter = svd_max_iter)[0]
        SM = self._oracle_l1norm(G)
        DL = X - SL
        DM = M - SM
        dg = np.sum(DL * G) + np.sum(DM * G)
        return SL, SM, DL, DM, dg
 
 
    # ------------------------------------------------------------------
    # Negative duality-gap guard
    # ------------------------------------------------------------------
 
    def _try_positive_dg(self, oracle_fn, G, *args):
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
        *args
            Additional positional arguments forwarded to oracle_fn (e.g. X, M).
 
        Returns
        -------
        result : tuple
            The return value of oracle_fn (possibly from a retry call).
        '''
        result = oracle_fn(G, *args,
                           warm_start_uv = False,
                           svd_max_iter  = self.svd_max_iter_)
        dg = result[-1]
        if dg >= 0:
            return result
 
        self.logger_.warn(f"Iteration {self.iter_num_}. Duality gap < 0 ({dg:g}).")
        if self.svd_method_ != 'power':
            return result
 
        self.logger_.warn("Retrying SVD power iteration with larger budget.")
        svd_max_iter = self.svd_max_iter_ * 2 if self.svd_max_iter_ is not None else 100
        for n_rep in range(1, 6):

            # Use fresh random initialization (warm_start_uv=False) so we don't
            # get trapped in the basin of a non-dominant singular vector.
            result = oracle_fn(G, *args,
                               warm_start_uv = False,
                               svd_max_iter  = svd_max_iter)
            dg = result[-1]
            self.logger_.warn(
                f"Power iteration trial {n_rep}. "
                f"dg = {dg:g}, n_iter = {self.svd_n_iter_}, max_iter = {svd_max_iter}"
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
 
    def _compute_step_size(self, dg, D):
        """
        Exact line-search step size  gamma = dg / step_denom(D), clamped to [0, 1].
 
        The denominator is model-specific and computed by the objective.
 
        Parameters
        ----------
        dg : float
            Duality gap (numerator of the line search).
        D : np.ndarray
            Descent direction.
 
        Returns
        -------
        ss : float   step size in (0, 1].
        """
        denom = self.obj_.step_denom(D)
        ss    = min(dg / denom, 1.0)
        if ss < 0:
            self.logger_.warn(
                f"Step size < 0 ({ss:g}). Using last valid step size."
            )
            ss = self.st_list_[-1]
        return ss
 
 
    # ------------------------------------------------------------------
    # Frank-Wolfe step functions
    # ------------------------------------------------------------------
 
    def _fw_one_step_nnm(self, X):
        '''
        Single Frank-Wolfe step for the 'nnm' and 'nnm-corr' models.
 
        Parameters
        ----------
        X : np.ndarray - current iterate.
 
        Returns
        -------
        Xnew : np.ndarray
        G    : np.ndarray - gradient at X
        dg   : float      - duality gap
        step : float      - step size used
        '''
        # 1. Gradient
        G = self.obj_.gradient(X)
        # 2. Linear oracle + descent direction + duality gap (with negative-dg guard)
        S, D, dg = self._try_positive_dg(self._oracle_nnm, G, X)
        # 3. Step size
        step = self._compute_step_size(dg, D)
        # 4. Update:  X - step*(X - S) = (1-step)*X + step*S
        Xnew = X - step * D
        return Xnew, G, dg, step
 
 
    def _fw_one_step_nnm_sparse(self, X, M):
        '''
        Single Frank-Wolfe step for the 'nnm-sparse' model.
 
        Parameters
        ----------
        X : np.ndarray - current low-rank iterate.
        M : np.ndarray - current sparse iterate.
 
        Returns
        -------
        Xnew : np.ndarray
        Mnew : np.ndarray
        G    : np.ndarray - gradient at X + M
        dg   : float      - duality gap
        step : float      - step size used
        '''
        # 1. Gradient
        G = self.obj_.gradient(X + M)
        # 2. Linear oracles + descent directions + duality gap (with negative-dg guard)
        SL, SM, DL, DM, dg = self._try_positive_dg(self._oracle_nnm_sparse, G, X, M)
        # 3. Step size (uses combined descent direction)
        step = self._compute_step_size(dg, DL + DM)
        # 4. Update low-rank and sparse components
        Xnew = X - step * DL
        Mnew = M - step * DM
        # 5. l1-ball projection half-step on M
        G_half = self.obj_.gradient(Xnew + Mnew)
        Mnew = self._proj_l1ball(Mnew - G_half)
        return Xnew, Mnew, G, dg, step


    def _proj_l1ball(self, X):
        n, p = X.shape
        ep = EuclideanProjection(method = self.simplex_method_, target = 'l1')
        ep.fit(X.ravel(), a = self.obj_.l1_threshold_)
        return ep.proj.reshape(n, p)


    # ------------------------------------------------------------------
    # Stopping criterion
    # ------------------------------------------------------------------

    def _do_stop(self):

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

        if 'duality_gap' in self.stop_criteria_:
            dg = self.dg_list_[-1]
            if np.abs(dg) <= self.tol_:
                self.convergence_msg_ = "Duality gap converged below tolerance."
                return True

        if 'step_size' in self.stop_criteria_:
            ss = self.st_list_[-1]
            if ss <= self.step_size_tol_:
                self.convergence_msg_ = "Step size converged below tolerance."
                return True

        if 'relative_objective' in self.stop_criteria_:
            fx = self.fx_list_[-1]
            fx0 = self.fx_list_[-2]
            # fx_rel = np.abs((fx - fx0) / fx0)
            fx_rel = _safe_relative_change(fx, fx0)
            if fx_rel <= self.fxrel_tol_:
                self.convergence_msg_ = "Relative difference in objective function converged below tolerance."
                return True

        if 'relative_dg' in self.stop_criteria_:
            dg = self.dg_list_[-1]
            dg0 = self.dg_list_[-2]
            # dg_rel = np.abs((dg - dg0) / dg0)
            dg_rel = _safe_relative_change(dg, dg0)
            if dg_rel <= self.fxrel_tol_:
                self.convergence_msg_ = "Relative difference in duality gap converged below tolerance."
                return True

        return False


    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------
    def fit(self, Y, r, 
            weight = None, mask = None, X0 = None,
            L_inv = None, Sigma_inv = None):

        '''
        Wrapper function for the low rank matrix approximation (LRMA)
        using the Frank-Wolfe (FW) algorithm.

        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix. May have missing values as np.nan.

        r : float [if model = 'nnm'] or tuple(float, float) [if model = 'nnm-sparse']
            Nuclear norm constraint radius. For 'nnm-sparse', a tuple 
            (r_nuc, l1_multiplier) may be passed, where the second element
            is a multiplier to determine the threshold l1 norm.
            l1_multiplier defaults to 1.0 when r is scalar.

        weight : (optional) np.ndarray [size (n, p); dtype: float], default=None
            An array of weights for each element in the input matrix Y.

        mask : (optional) np.ndarray [size (n, p); dtype: boolean], default=None
            An array of masks to hide specific elements in the input matrix Y.
            True for entries to exclude. Internally, it is combined with 
            all NaN entries.

        X0 : (optional) np.ndarray [size (n, p); dtype: float], default=None
            Optional initial guess for the low rank matrix in the FW algorithm. 
            Defaults to zeros.

        L_inv : np.ndarray [size (n, n); dtype: float], default=None
            Inverse Cholesky factor of the sampling covariance A.
            Required for model='nnm-corr'.
 
        Sigma_inv : np.ndarray [size (n, n); dtype: float], default=None
            Inverse of the sampling covariance matrix A = L L^T.
            Required for model='nnm-corr'.


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

        # ------------------------------------------------------------------
        # Build objective
        # ------------------------------------------------------------------
        self.obj_ = self._make_objective(Y, r, mask, weight, L_inv, Sigma_inv)

        self.logger_.debug(
            f"Masked entries = {np.sum(self.obj_.mask_) if self.obj_.mask_ is not None else 0}"
        )

        # Compatibility attributes for external inspection and tests
        self.Y_    = self.obj_.Y_
        self.mask_ = self.obj_.mask_
        self.rank_ = self.obj_.r_
        if self.model_ == 'nnm-sparse':
            self.l1_thres_ = self.obj_.l1_threshold_
        if self.model_ == 'nnm-corr':
            self.L_inv_     = self.obj_.L_inv_
            self.Sigma_inv_ = self.obj_.Sigma_inv_

        # ---------------------
        # Initialize iterates
        # ---------------------
        X = np.zeros_like(self.obj_.Y_) if X0 is None else X0.copy()
        if self.model_ == 'nnm-sparse':
            M = np.zeros_like(self.obj_.Y_)

        # ---------------------
        # Initialize history
        # ---------------------
        dg = np.inf
        step = 1.0
        fx = self.obj_.value(X)

        self.fx_list_  = [fx]
        self.dg_list_  = [dg]
        self.st_list_  = [step]
        self.cpu_time_ = [1e-8] # do not use 0 to avoid log10 error.
        self.iter_num_ = 0
        self.convergence_msg_ = "Maximum number of iterations reached."

        if self.model_ == 'nnm-sparse':
            self.fm_list_ = [self.obj_.value(M)]
            self.fl_list_ = [fx]

        cpu_time_old = time.process_time()

        # ---------------------
        # Steps 1, ..., max_iter
        # ---------------------
        for i in range(self.max_iter_):

            self.iter_num_ += 1
            self.logger_.debug(f"Iteration {self.iter_num_}.")

            if self.model_ in ('nnm', 'nnm-corr'):
                X, G, dg, step = self._fw_one_step_nnm(X)
                fx = self.obj_.value(X)

            elif self.model_ == 'nnm-sparse':
                X, M, G, dg, step = self._fw_one_step_nnm_sparse(X, M)
                fx = self.obj_.value(X + M)
                fm = self.obj_.value(M)
                fl = self.obj_.value(X)

            self.logger_.debug(f"Step size {step:.3f}. Duality Gap {dg:g}.")

            # Append iteration history
            cpu_time = time.process_time()
            self.cpu_time_.append(cpu_time - cpu_time_old)
            cpu_time_old = cpu_time
            self.fx_list_.append(fx)
            self.dg_list_.append(dg)
            self.st_list_.append(step)

            if self.model_ == 'nnm-sparse':
                self.fm_list_.append(fm)
                self.fl_list_.append(fl)

            # The logger would not print anything if logging level is higher than INFO.
            # The outer if loop reduces some workload of the logger class.
            # However, this introduces a bug: if debug = True and show_progress = False,
            # then the logging level is DEBUG but the steps are not printed. 
            # I can live with that bug.
            if self.show_progress_:
                if (self.iter_num_ == 1) or (self.iter_num_ % self.prog_step_skip_ == 0):
                    self.logger_.info(f"Iteration {i+1}. Step size {step:.3f}. Duality Gap {dg:g}")

            if self._do_stop():
                break

            # If max_iter is reached, then terminate.
            if i == self.max_iter_ - 1:
                self.convergence_msg_ = f"Maxinum number of iterations reached."


        self.logger_.info(
            f"Stopping at iteration {self.iter_num_}. "
            f"Step size {step:.3f}. Duality Gap {dg:g}")
        self.logger_.info(self.convergence_msg_)
        self.X_ = X
        if self.model_ == 'nnm-sparse': 
            self.M_ = M

        return
