# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import time
import logging
from .objectives import make_objective
from .svd import nuclear_norm_oracle
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
        Linear optimization oracle on the l1 norm ball.
        '''
        idx = np.unravel_index(np.argmax(np.abs(G)), G.shape)
        S = np.zeros_like(G)
        sgn = np.sign(G[idx])
        if sgn != 0:
            S[idx] = - self.obj_.l1_threshold_ * sgn
        return S


    def _oracle_nucnorm(self, G, *, iter_state, warm_start_uv = False, svd_max_iter = None):
        '''
        Linear optimization oracle on the nuclear norm ball.
        This is already returned by the `nuclear_norm_oracle`.
        We only take the results, log the current atoms in iter_state
        and return the clean vertex.
 
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
        S    : np.ndarray  - FW vertex (nuclear-norm oracle output)
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

        return S


    # ------------------------------------------------------------------
    # Negative duality-gap guard
    # ------------------------------------------------------------------
 
    def _try_positive_dg(self, G, *, iter_state):
        '''
        Call ``oracle_fn(G, *args, warm_start_uv=False, svd_max_iter=...)``
        Call `_oracle_nucnorm`, build the descent direction D = anchor - S,
        compute dg = <D, G>, and retry with larger SVD iteration budgets 
        if dg < 0 (power iteration only).

        A negative duality gap indicates inaccurate singular vectors.
        Retrying with more power iterations (for method='power') typically
        resolves the issue.

        After this returns, iter_state.svd_u / svd_vt hold the singular
        vectors of the accepted oracle call.

        AFW note: this helper is for the *FW direction* only. The AFW
        away direction has D_aw = S_aw - X with S_aw drawn from the active
        set (not the SVD), so it does not need this helper.
 
        Parameters
        ----------
        G : np.ndarray
            Current gradient passed as the first positional argument to oracle_fn.
        iter_state : IterState
            Current state of the iteration, forwarded to oracle_fn.
 
        Returns
        -------
        S : np.ndarray  - FW vertex (nuclear-norm oracle output)
        D : np.ndarray  - descent direction  (X - S)
        dg : float      - duality gap

        '''
        S = self._oracle_nucnorm(
            G,
            iter_state = iter_state,
            warm_start_uv = False,
            svd_max_iter  = self.cfg_.svd_max_iter)
        D = iter_state.X - S
        dg = float(np.sum(D * G))   # equivalent to trace(D.T @ G), avoids full matmul

        # Numerical-noise guard. dg = <X, G> - <S, G> where -<S, G> = r * sigma_1(G).
        # When X is on or near the FW vertex (e.g. after a gamma=1 step), these
        # two terms cancel and dg is dominated by floating-point roundoff.
        # Clamp tiny negatives to 0; only retry / warn if the gap is genuinely
        # negative at a magnitude that exceeds floating-point noise.
        #
        # What should be the noise tolerance?
        # The standard floating-point dot-product bound gives
        #
        #   |fl(dg) - dg| <= gamma_N * |vec(X-S)|^T |vec(G)|,
        #
        # with gamma_N = N*u/(1 - N*u), where N = X.size.
        # By Cauchy-Schwarz,
        #
        #   |vec(X-S)|^T |vec(G)| <= ||X-S||_F * ||G||_F.
        #
        # For a practical tolerance, we use a smaller pairwise-summation-style
        # factor k_round = 10 * log2(N), rather than the pessimistic sequential
        # worst-case factor N. The forward roundoff bound on dg = <X - S, G>:
        #
        #   |fl(dg) - dg|  ≤  k_round · eps · (||X||_F + ||S||_F) · ||G||_F
        #
        # because ||X-S||_F <= ||X||_F + ||S||_F.
        #
        # Use feasibility bounds:
        #   ||X||_F ≤ r, ||S||_F = r  =>  ||X||_F + ||S||_F ≤ 2r
        #   ||G||_F bounded by self.obj_.gradient_norm_bound  (cached)
        # k_round = 10 · log2(N) absorbs pairwise-summation and X - S subtraction.
        k_round    = 10.0 * np.log2(max(D.size, 2))
        noise_tol  = (
            k_round 
            * np.finfo(float).eps 
            * 2.0 * self.obj_.radius_
            * self.obj_.gradient_norm_bound
        )
        if dg >= -noise_tol:
            return S, D, max(dg, 0.0)

        self.logger_.warn(f"Iteration {iter_state.istep}. Duality gap < 0 ({dg:g}, noise tolerance = {noise_tol:g})")
        if self.cfg_.svd_method != 'power':
            return S, D, dg
 
        self.logger_.warn("Retrying SVD power iteration with larger budget.")
        svd_max_iter = self.cfg_.svd_max_iter * 2 if self.cfg_.svd_max_iter is not None else 100
        for n_rep in range(1, 6):

            # Use fresh random initialization (warm_start_uv=False) so we don't
            # get trapped in the basin of a non-dominant singular vector.
            S = self._oracle_nucnorm(
                G,
                iter_state = iter_state,
                warm_start_uv = False,
                svd_max_iter  = svd_max_iter)
            D = iter_state.X - S
            dg = np.sum(D * G)
            self.logger_.warn(
                f"Power iteration trial {n_rep}. "
                f"dg = {dg:g}, n_iter = {iter_state.svd_n_iter}, max_iter = {svd_max_iter}"
            )
            if dg >= -noise_tol:
                return S, D, max(dg, 0.0)
            svd_max_iter *= 2
 
        self.logger_.warn(
            f"Power iteration could not recover non-negative dg "
            f"({dg:g}, noise tol = {noise_tol:g})."
        )
        return S, D, dg

    # ------------------------------------------------------------------
    # Step size
    # ------------------------------------------------------------------
 
    def _compute_step_size(self, dg, D, iter_state, gamma_max = 1.0):
        """
        Exact line-search step size  gamma = dg / step_denom(D), 
        clamped to [0, gamma_max].
 
        The denominator is model-specific and computed by the objective.
        gamma_max sets the upper bound on the step:

            FW direction (any model):  gamma_max = 1.0  (default)
            AFW away direction:        gamma_max = alpha_aw / (1 - alpha_aw),
                                       to keep the active-set weight of the
                                       worst atom non-negative after the update.


        Parameters
        ----------
        dg : float
            Duality gap (numerator of the line search).
        D : np.ndarray
            Descent direction.
        iter_state: IterState
            Required for last step size.
        gamma_max : float, default=1.0
            Upper clamp on the step size.
 
        Returns
        -------
        ss : float   step size in (0, gamma_max].
        """
        denom = self.obj_.step_denom(D)

        if not np.isfinite(denom) or denom == 0.0:
            self.logger_.warn(f"Step size denominator is {denom:g} at iteration {iter_state.istep}. Returning zero step.")
            return 0.0

        ss  = min(dg / denom, gamma_max)
        if ss < 0:
            self.logger_.warn(
                f"Step size < 0 ({ss:g}) at iteration {iter_state.istep}. "
                "Returning zero step; the algorithm will stop on this "
                "iteration if 'step_size' is in stop_criteria."
            )
            ss = 0.0
            #ss = iter_state.last_step_size
        return ss
 
 
    # ------------------------------------------------------------------
    # Frank-Wolfe step functions
    # ------------------------------------------------------------------
 
    def _fw_one_step_nnm(self, iter_state):
        '''
        Single Frank-Wolfe step for the 'nnm' and 'nnm-corr' models.

        Uses the convention D = X - S so that the update is:
            X_new = X - step * D  =  (1 - step) * X + step * S
 
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
        S, D, dg = self._try_positive_dg(G, iter_state = iter_state)

        # 3. Step size
        step_size = self._compute_step_size(dg, D, iter_state)

        # 4. Update
        iter_state.X = iter_state.X - step_size * D
        iter_state.last_step_size = step_size

        return G, dg, step_size
 
 
    def _fw_one_step_nnm_sparse(self, iter_state):
        '''
        Single Frank-Wolfe step for the 'nnm-sparse' model.
 
        Uses the convention D = X - S so that the update is:
            X_new = X - step * D  =  (1 - step) * X + step * S
 
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
        dgX  : float      - X-block duality gap (drives stopping)
        dgM  : float      - M-block duality gap (diagnostic, ~0)
        step : float      - step size used
        '''
        # 1. Gradient at the joint iterate
        G = self.obj_.gradient(iter_state.X + iter_state.M)

        # 2. X-block linear oracle + descent direction + duality gap (with negative-dg guard)
        SX, DX, dgX = self._try_positive_dg(G, iter_state = iter_state)

        # 3. M-block: closed-form l1 oracle, gM diagnostic only (no retry)
        SM = self._oracle_l1norm(G)
        DM = iter_state.M - SM
        dgM = float(np.sum(DM * G))

        # 4. Step size from the X-block gap
        step_size = self._compute_step_size(dgX, DX, iter_state)

        # 5. Update X, then exact projection for M
        iter_state.X = iter_state.X - step_size * DX
        iter_state.M = self.obj_.project_sparse(iter_state.X)
        iter_state.last_step_size = step_size

        return G, dgX, dgM, step_size


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

        sparse_scale: float or None
            Multiplier to determine the threshold sparse radius (constraint on 
            l1 norm). Used for model='nnm-sparse'. Defaults to 1.0 if None.

        weight : np.ndarray [size (n, p); dtype: float] or None
            An array of weights for each element in the input matrix.
            None means unweighted.

        mask : np.ndarray [size (n, p); dtype: boolean] or None
            An array of masks to hide specific elements in the input matrix Y.
            True for entries to exclude. Internally, it is combined with 
            all NaN entries.

        X0 : np.ndarray [size (n, p); dtype: float] or None
            Optional initial guess for the low rank matrix in the FW algorithm. 
            Defaults to zeros if None.

        noise_cov : SamplingCovariance
                    or np.ndarray [size (n, n); dtype: float] or None
            SamplingCovariance object. Required for model='nnm-corr'.
            If input is np.ndarray, it is automatically passed through
                A = SamplingCovariance.from_matrix(noise_cov)
            See `utils/sampling_covariance.py`.

        Returns
        -------
        self: FrankWolfe
            Fitted instance. Access the result via ``result''.

        Note
        ----
        Any NaN value in the input matrix is automatically added to `mask`.


        Usage
        -----
            model = FrankWolfe(<params>)
            model.fit(Y, r, ...)
            result = model.result
        '''

        self.logger_.debug(f"Model = {self.model_}")
        self.logger_.debug(f"Shape of input data = {Y.shape}")

        # ---------------------
        # Validate
        # ---------------------
        if self.model_ == 'nnm-corr' and noise_cov is None:
            raise ValueError(
                "model='nnm-corr' requires a covariance argument. "
                "Construct one via SamplingCovariance.from_matrix(A)."
            )
        if isinstance(noise_cov, np.ndarray): 
            noise_cov = SamplingCovariance.from_matrix(noise_cov)

        # ---------------------
        # Build objective
        # ---------------------
        self.obj_ = make_objective(self.model_, Y, radius,
            sparse_scale = sparse_scale,
            mask = mask, weight = weight, noise_cov = noise_cov,
            simplex_method = self.cfg_.simplex_method)

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
            fx_hist  = [self.obj_.value(iter_state.X + iter_state.M)]
            fm_hist  = [self.obj_.value(iter_state.M)]
            fl_hist  = [self.obj_.value(iter_state.X)]
            dgm_hist = [np.inf] 

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
                G, dg, dgM, step_size = self._fw_one_step_nnm_sparse(iter_state)
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
                dgm_hist.append(dgM)

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
            duality_gap_sparse   = dgm_hist if self.model_ == 'nnm-sparse' else None,
            duality_gap_low_rank = dg_hist  if self.model_ == 'nnm-sparse' else None,
        )

        metrics = {
            'nuclear_norm': float(np.linalg.norm(iter_state.X, 'nuc')),
            'radius' : radius,
        }

        if self.model_ == 'nnm-sparse':
            metrics['l1_norm'] = float(np.sum(np.abs(iter_state.M)))
            metrics['l1_threshold'] = float(self.obj_.l1_threshold_)
            metrics['sparse_scale'] = sparse_scale

        self.result_  = FitResult(
            X           = iter_state.X,
            M           = iter_state.M if self.model_ == 'nnm-sparse' else None,
            history     = history,
            n_iter      = iter_state.istep,
            converged   = iter_state.stop_reason != StopReason.MAX_ITER,
            stop_reason = iter_state.stop_reason,
            message     = iter_state.stop_reason.message,
            metrics     = metrics,
        )

        return self
