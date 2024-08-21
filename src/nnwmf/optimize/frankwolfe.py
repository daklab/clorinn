# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import time
import logging
from .top_comp_svd import TopCompSVD
from .simplex_projection import EuclideanProjection
from ..utils.logs import CustomLogger
from ..utils import model_errors as merr

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
            'relative_objective': stop the algorithm if the change in objective
                function in successive iterations is less than 'rel_tol'. A low value
                indicates that the updates are negligible.

    model : string, default='nnm'
        The model used for the FW algorithm. Options include:
            'nnm' : Nuclear Norm constraint
            'nnm-sparse' : Nuclear Norm + L1 constraint

    simplex_method : string, default='sort'
        The method used for the simplex projection of the l1 ball. See 
        EuclideanProjection for available options.

    benchmark : boolean, default=False
        Whether to calculate the error statistic for benchmarking. If set to `True`,
        it requires 'Ytrue' in the `fit()` function.

    benchmark_method : string, default='rmse'
        The error statistic to be calculated for benchmarking. See utils.model_errors 
        for available options.

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


    TO-DO.
    -----
        - [ ] Allow multiple error statistic calculation during benchmarking.
    """
    def __init__(self, max_iter = 1000,
            svd_method = 'power', svd_max_iter = None,
            stop_criteria = ['duality_gap', 'step_size', 'relative_objective'],
            model = 'nnm', simplex_method = 'sort',
            benchmark_method = 'rmse',
            tol = 1e-3, step_tol = 1e-3, rel_tol = 1e-8,
            show_progress = False, print_skip = None,
            debug = False, suppress_warnings = False, benchmark = False):
                
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
        # flag the benchmarking, requires Ytrue for calculating errors
        self.is_benchmark_ = benchmark
        self.benchmark_method_ = benchmark_method
        return


    @property
    def X(self):
        return self.X_


    @property
    def duality_gaps(self):
        return self.dg_list_


    @property
    def fx(self):
        return self.fx_list_


    @property
    def steps(self):
        return self.st_list_


    def _f_objective(self, X):
        '''
        Objective function
        Y is observed, X is estimated
        W is the weight of each observation.
        '''
        Xmask = self._get_masked(X)
        # The * operator can be used as a shorthand for np.multiply on ndarrays.
        if self.weight_mask_ is None:
            fx = 0.5 * np.linalg.norm(self.Y_ - Xmask, 'fro')**2
        else:
            fx = 0.5 * np.linalg.norm(self.weight_mask_ * (self.Y_ - Xmask), 'fro')**2
        return fx


    def _f_gradient(self, X):
        '''
        Gradient of the objective function.
        '''
        Xmask = self._get_masked(X)
        if self.weight_mask_ is None:
            gx = Xmask - self.Y_
        else:
            gx = np.square(self.weight_mask_) * (Xmask - self.Y_)
        return gx


    def _fw_step_size(self, dg, D):
        if self.weight_mask_ is None:
            denom = np.linalg.norm(D, 'fro')**2
        else:
            denom = np.linalg.norm(self.weight_mask_ * D, 'fro')**2
        ss = dg / denom
        ss = min(ss, 1.0)
        if ss < 0:
            self.logger_.warn("Step Size is less than 0. Using last valid step size.")
            ss = self.st_list_[-1]
        return ss


    def _get_masked(self, X):
        if self.mask_ is None or X is None:
            return X
        else:
            return X * ~self.mask_


    def _linopt_oracle_l1norm(self, X):
        '''
        Linear optimization oracle,
        where the feasible region is a l1 norm ball for some r
        '''
        maxidx = np.unravel_index(np.argmax(X), X.shape)
        S = np.zeros_like(X)
        S[maxidx] = - self.l1_thres_
        return S


    def _linopt_oracle_nucnorm(self, X):
        '''
        Linear optimization oracle,
        where the feasible region is a nuclear norm ball for some r
        '''
        U1, V1_T = self._get_singular_vectors(X)
        S = - self.rank_ * U1 @ V1_T
        return S


    def _get_singular_vectors(self, X):
        max_iter = self.svd_max_iter_
        if max_iter is None:
            nstep = len(self.st_list_) + 1
            max_iter = 10 + int(nstep / 20)
            max_iter = min(max_iter, 25)
        svd = TopCompSVD(method = self.svd_method_, max_iter = max_iter)
        svd.fit(X)
        return svd.u1, svd.v1_t


    def _proj_l1ball(self, X):
        n, p = X.shape
        xflat = X.flatten()
        eucp = EuclideanProjection(method = self.simplex_method_, target = 'l1')
        eucp.fit(xflat, a = self.l1_thres_)
        return eucp.proj.reshape(n, p)


    def fit(self, Y, r, weight = None, mask = None, X0 = None, Ytrue = None):

        '''
        Wrapper function for the low rank matrix approximation (LRMA)
        using the Frank-Wolfe (FW) algorithm.

        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix with n rows and p columns. May have missing values as np.nan.

        r : float [if model = 'nnm'] or tuple(float, float) [if model = 'nnm-sparse']
            Input constraint for the FW algorithm. If the model is 'nnm', the constraint 
            is the threshold nuclear norm of the matrix. If the model is 'nnm-sparse', 
            the constraint is a tuple of two elements -- the first element is the threshold 
            nuclear norm and the second element is the threshold l_1 norm of the matrix.

        weight : (optional) np.ndarray [size (n, p): dtype: float], default=None
            An array of weights for each element in the input matrix Y.

        mask : (optional) np.ndarray [size (n, p); dtype: boolean], default=None
            An array of masks to hide specific elements in the input matrix Y.
            To design the mask array, use 'True' for hiding / masking the element 
            and 'False' for using the element in calculations.

        X0 : (optional) np.ndarray [size (n, p); dtype: float], default=None
            Optional initial guess for the low rank matrix in the FW algorithm. 
            If set to None, the optimization is initiated from a matrix of zeros.

        Ytrue : (optional) np.ndarray [size (n, p); dtype: float], default=None
            Optional 'ground truth' for the input data matrix. This can be used
            for benchmarking, where the error statistic (see utils.model_error)
            is calculated between the estimated matrices (X, M and X+M) and Ytrue.
            If set to None, then `Ytrue` is assumed equal to `Y`.

        Note
        ----
        Any NaN value in the input matrix is added to `mask`.

        Usage
        -----
            model = FrankWolfe(<params>)
            model.fit(Y, r, ...)
        '''

        n, p = Y.shape

        # Set class attributes
        # to avoid nan values, set nan to zero for calculation.
        # but remember the indices of nan values.
        nan_mask = np.isnan(Y) # the input may contain NaN entries
        self.Y_ = np.nan_to_num(Y, copy = True, nan = 0.0)
        # Combine the NaN mask and the input mask, if provided.
        # Prefer to keep it None if there is no mask for slightly faster calculations.
        input_mask = np.zeros((n, p), dtype=bool) if mask is None else mask
        self.mask_ = np.logical_or(nan_mask, input_mask)
        if not np.any(self.mask_): self.mask_ = None

        self.weight_ = weight
        self.weight_mask_ = self._get_masked(self.weight_)
        if self.model_ == 'nnm':
            self.rank_ = r
        elif self.model_ == 'nnm-sparse':
            if isinstance(r, tuple):
                self.rank_, self.l1_thres_ = r
            else:
                self.rank_ = r
                self.l1_thres_ = 1.0
            self.l1_thres_ *= np.linalg.norm(self.Y_, ord = 1) / np.sqrt(np.max(self.Y_.shape))

        # Step 0
        X = np.zeros_like(Y) if X0 is None else X0.copy()
        dg = np.inf
        step = 1.0
        fx = self._f_objective(X)
        # used only for 'nnm-sparse'
        if self.model_ == 'nnm-sparse':
            M = np.zeros_like(Y)
            fm = self._f_objective(M)
            self.fm_list_ = [fm]
            self.fl_list_ = [fx]

        # Save relevant variables in list
        self.fx_list_ = [fx]
        self.dg_list_ = [dg]
        self.st_list_ = [step]
        self.cpu_time_ = [1e-8] # do not use 0 to avoid log10 error.

        # Benchmarking
        if self.is_benchmark_:
            if Ytrue is None:
                self.logger_.warn("True input not provided. Using observed input matrix for RMSE calculation.")
                Ytrue = self.Y_.copy()
            assert Ytrue.shape == Y.shape
            if self.model_ == 'nnm':
                self.rmse_ = [merr.get(Ytrue, X, method = self.benchmark_method_)]
            elif self.model_ == 'nnm-sparse':
                self.rmse_          = [merr.get(Ytrue, X + M, method = self.benchmark_method_)]
                self.rmse_low_rank_ = [merr.get(Ytrue, X, method = self.benchmark_method_)]
                self.rmse_sparse_   = [merr.get(Ytrue, M, method = self.benchmark_method_)]

        cpu_time_old = time.process_time()
        # Steps 1, ..., max_iter
        for i in range(self.max_iter_):

            if self.model_ == 'nnm':
                X, G, dg, step = self._fw_one_step_nnm(X)
                fx = self._f_objective(X)
            elif self.model_ == 'nnm-sparse':
                X, M, G, dg, step = self._fw_one_step_nnm_sparse(X, M)
                fx = self._f_objective(X + M)
                fm = self._f_objective(M)
                fl = self._f_objective(X)

            # Time 
            cpu_time = time.process_time()
            self.cpu_time_.append(cpu_time - cpu_time_old)
            cpu_time_old = cpu_time


            self.fx_list_.append(fx)
            self.dg_list_.append(dg)
            self.st_list_.append(step)

            if self.model_ == 'nnm-sparse':
                self.fm_list_.append(fm)
                self.fl_list_.append(fl)

            if self.is_benchmark_:
                if self.model_ == 'nnm':
                    self.rmse_.append(merr.get(Ytrue, X, method = self.benchmark_method_))
                elif self.model_ == 'nnm-sparse':
                    self.rmse_.append(merr.get(Ytrue, X + M, method = self.benchmark_method_))
                    self.rmse_low_rank_.append(merr.get(Ytrue, X, method = self.benchmark_method_))
                    self.rmse_sparse_.append(merr.get(Ytrue, M, method = self.benchmark_method_))

            # The logger would not print anything if logging level is higher than INFO.
            # The outer if loop reduces some workload of the logger class.
            # However, this introduces a bug: if debug = True and show_progress = False,
            # then the logging level is DEBUG but the steps are not printed. 
            # I can live with that bug.
            if self.show_progress_:
                if (i % self.prog_step_skip_ == 0):
                    self.logger_.info(f"Iteration {i}. Step size {step:.3f}. Duality Gap {dg:g}")

            if self._do_stop():
                break

            # If max_iter is reached, then terminate.
            if i == self.max_iter_ - 1:
                self.convergence_msg_ = f"Maxinum number of iterations reached."


        self.logger_.info(f"Iteration {i}. Step size {step:.3f}. Duality Gap {dg:g}")
        self.logger_.info(self.convergence_msg_)
        self.X_ = X
        if self.model_ == 'nnm-sparse': self.M_ = M

        return

    def _fw_one_step_nnm(self, X):
        # 1. Gradient for X_(t-1)
        G = self._f_gradient(X)
        # 2. Linear optimization subproblem
        S = self._linopt_oracle_nucnorm(G)
        # 3. Define D
        D = X - S
        # 4. Duality gap
        dg = np.trace(D.T @ G)
        # 5. Step size
        step = self._fw_step_size(dg, D)
        # 6. Update
        Xnew = X - step * D
        return Xnew, G, dg, step


    def _fw_one_step_nnm_sparse(self, L, M):
        # 1. Gradient for X_(t-1)
        G = self._f_gradient(L + M)
        # 2. Linear optimization subproblem
        SL = self._linopt_oracle_nucnorm(G)
        SM = self._linopt_oracle_l1norm(G)
        # 3. Define D
        DL = L - SL
        DM = M - SM
        # 4. Duality gap
        dg = np.trace(DL.T @ G) + np.trace(DM.T @ G)
        # 5. Step size
        #step = self._fw_step_size(dg, DL + DM)
        step = self._fw_step_size(dg, DL)
        # 6. Update
        Lnew = L - step * DL
        Mnew = M - step * DM
        # 7. l1 ball projection
        G_half = self._f_gradient(Lnew + Mnew)
        Mnew = self._proj_l1ball(Mnew - G_half)
        return Lnew, Mnew, G, dg, step


    def _do_stop(self):
        # self._stop_criteria = ['duality_gap', 'step_size', 'relative_objective']
        #
        if 'duality_gap' in self.stop_criteria_:
            dg = self.dg_list_[-1]
            if np.abs(dg) <= self.tol_:
                self.convergence_msg_ = "Duality gap converged below tolerance."
                return True
        #
        if 'step_size' in self.stop_criteria_:
            ss = self.st_list_[-1]
            if ss <= self.step_size_tol_:
                self.convergence_msg_ = "Step size converged below tolerance."
                return True
        #
        if 'relative_objective' in self.stop_criteria_:
            fx = self.fx_list_[-1]
            fx0 = self.fx_list_[-2]
            fx_rel = np.abs((fx - fx0) / fx0)
            if fx_rel <= self.fxrel_tol_:
                self.convergence_msg_ = "Relative difference in objective function converged below tolerance."
                return True
        #
            if len(self.st_list_) > 2:
                dg = self.dg_list_[-1]
                dg0 = self.dg_list_[-2]
                dg_rel = np.abs((dg - dg0) / dg0)
                if dg_rel <= self.fxrel_tol_:
                    self.convergence_msg_ = "Relative difference in duality gap converged below tolerance."
                    return True
        #
        return False
