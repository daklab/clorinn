"""
Nuclear Norm Rank Minimization using Frank-Wolfe algorithm
"""
# Author: Saikat Banerjee

import numpy as np
from .top_comp_svd import TopCompSVD
from ..utils.logs import CustomLogger

class NNMFW():

    def __init__(self, max_iter = 1000,
            svd_method = 'power', svd_max_iter = None,
            stop_criteria = ['duality_gap', 'step_size', 'relative_objective'],
            tol = 1e-3, step_tol = 1e-3, rel_tol = 1e-8,
            show_progress = False, print_skip = None,
            debug = True, suppress_warnings = False):
        self._max_iter = max_iter
        self._svd_method = svd_method
        self._svd_max_iter = svd_max_iter
        self._stop_criteria = stop_criteria
        self._tol = tol
        self._step_size_tol = step_tol
        self._fxrel_tol = rel_tol
        self._show_progress = show_progress
        self._prog_step_skip = print_skip
        if self._show_progress and self._prog_step_skip is None:
            self._prog_step_skip = max(1, int(self._max_iter / 100)) * 10
        # set logger for this class
        self._is_debug = debug
        self.logger    = CustomLogger(__name__, is_debug = self._is_debug)
        self._suppress_warnings = suppress_warnings
        return


    @property
    def X(self):
        return self._X


    @property
    def duality_gaps(self):
        return self._dg_list


    @property
    def fx(self):
        return self._fx_list


    @property
    def steps(self):
        return self._st_list


    def f_objective(self, X):
        '''
        Objective function
        Y is observed, X is estimated
        W is the weight of each observation.
        '''
        Xmask = self.get_masked(X)
        # The * operator can be used as a shorthand for np.multiply on ndarrays.
        if self._weight_mask is None:
            fx = 0.5 * np.linalg.norm(self._Y - Xmask, 'fro')**2
        else:
            fx = 0.5 * np.linalg.norm(self._weight_mask * (self._Y - Xmask), 'fro')**2
        return fx


    def f_gradient(self, X):
        '''
        Gradient of the objective function.
        '''
        Xmask = self.get_masked(X)
        if self._weight_mask is None:
            gx = Xmask - self._Y
        else:
            gx = np.square(self._weight_mask) * (Xmask - self._Y)
        return gx


    def fw_step_size(self, dg, D):
        if self._weight_mask is None:
            denom = np.linalg.norm(D, 'fro')**2
        else:
            denom = np.linalg.norm(self._weight_mask * D, 'fro')**2
        ss = dg / denom
        ss = min(ss, 1.0)
        if ss < 0:
            if not self._suppress_warnings:
                self.logger.warn("Step Size is less than 0. Using last valid step size.")
            ss = self._st_list[-1]
        return ss


    def get_masked(self, X):
        if self._mask is None or X is None:
            return X
        else:
            return X * ~self._mask


    def linopt_oracle(self, X):
        '''
        Linear optimization oracle,
        where the feasible region is a nuclear norm ball for some r
        '''
        U1, V1_T = self.get_singular_vectors(X)
        S = - self._rank * U1 @ V1_T
        return S


    def get_singular_vectors(self, X):
        max_iter = self._svd_max_iter
        if max_iter is None:
            nstep = len(self._st_list) + 1
            max_iter = 10 + int(nstep / 20)
            max_iter = min(max_iter, 25)
        svd = TopCompSVD(method = self._svd_method, max_iter = max_iter)
        svd.fit(X)
        return svd.u1, svd.v1_t


    def fit(self, Y, r, weight = None, mask = None, X0 = None):

        '''
        Wrapper function for the minimization
        mask: array of mask, True for indices which are masked, False for indices used in calculation
        '''

        n, p = Y.shape

        # Make some variables available for the class
        self._weight = weight
        self._mask = mask
        self._weight_mask = self.get_masked(self._weight)
        self._Y = Y
        self._rank = r

        # Step 0
        X = np.zeros_like(Y) if X0 is None else X0.copy()
        dg = np.inf
        step = 1.0
        fx = self.f_objective(X)

        # Save relevant variables in list
        self._dg_list = [dg]
        self._fx_list = [fx]
        self._st_list = [step]

        # Steps 1, ..., max_iter
        for i in range(self._max_iter):

            X, G, dg, step = self.fw_one_step(X)
            fx = self.f_objective(X)

            self._fx_list.append(fx)
            self._dg_list.append(dg)
            self._st_list.append(step)

            if self._show_progress:
                if (i % self._prog_step_skip == 0):
                    self.logger.info(f"Iteration {i}. Step size {step:.3f}. Duality Gap {dg:g}")

            if self.do_stop():
                break

        self._X = X

        return

    def fw_one_step(self, X):
        # 1. Gradient for X_(t-1)
        G = self.f_gradient(X)
        # 2. Linear optimization subproblem
        S = self.linopt_oracle(G)
        # 3. Define D
        D = X - S
        # 4. Duality gap
        dg = np.trace(D.T @ G)
        # 5. Step size
        step = self.fw_step_size(dg, D)
        # 6. Update
        Xnew = X - step * D
        return Xnew, G, dg, step

    def do_stop(self):
        # self._stop_criteria = ['duality_gap', 'step_size', 'relative_objective']
        #
        if 'duality_gap' in self._stop_criteria:
            dg = self._dg_list[-1]
            if np.abs(dg) <= self._tol:
                self._convergence_msg = "Duality gap converged below tolerance."
                return True
        #
        if 'step_size' in self._stop_criteria:
            ss = self._st_list[-1]
            if ss <= self._step_size_tol:
                self._convergence_msg = "Step size converged below tolerance."
                return True
        #
        if 'relative_objective' in self._stop_criteria:
            fx = self._fx_list[-1]
            fx0 = self._fx_list[-2]
            fx_rel = np.abs((fx - fx0) / fx0)
            if fx_rel <= self._fxrel_tol:
                self._convergence_msg = "Relative difference in objective function converged below tolerance."
                return True
        #
        return False
