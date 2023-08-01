"""
Nuclear Norm Rank Minimization using Frank-Wolfe algorithm
"""
# Author: Saikat Banerjee

import numpy as np
import time
from .top_comp_svd import TopCompSVD
from .simplex_projection import EuclideanProjection
from ..utils.logs import CustomLogger
from ..utils import model_errors as merr

class FrankWolfe():

    def __init__(self, max_iter = 1000,
            svd_method = 'power', svd_max_iter = None,
            stop_criteria = ['duality_gap', 'step_size', 'relative_objective'],
            model = 'nnm', simplex_method = 'condat',
            benchmark_method = 'rmse',
            tol = 1e-3, step_tol = 1e-3, rel_tol = 1e-8,
            show_progress = False, print_skip = None,
            debug = True, suppress_warnings = False, benchmark = False):
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
        self.is_debug_ = debug
        self.logger_   = CustomLogger(__name__, is_debug = self.is_debug_)
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
            if not self.suppress_warnings_:
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
        Wrapper function for the minimization
        mask: array of mask, True for indices which are masked, False for indices used in calculation
        '''

        n, p = Y.shape

        # Make some variables available for the class
        self.weight_ = weight
        self.mask_ = mask
        self.weight_mask_ = self._get_masked(self.weight_)
        self.Y_ = Y
        if self.model_ == 'nnm':
            self.rank_ = r
        elif self.model_ == 'nnm-sparse':
            self.rank_, self.l1_thres_ = r

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
        self.cpu_time_ = [0]

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

            if self.show_progress_:
                if (i % self.prog_step_skip_ == 0):
                    self.logger_.info(f"Iteration {i}. Step size {step:.3f}. Duality Gap {dg:g}")

            if self._do_stop():
                break

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
        step = self._fw_step_size(dg, DL + DM)
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
        return False
