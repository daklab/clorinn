#!/usr/bin/env python

import numpy as np
import time

from ..utils import model_errors as merr
from ..utils.logs import CustomLogger

def frobenius_norm(X):
    return np.linalg.norm(X, ord='fro')


def l1_norm(X):
    return np.sum(np.abs(X))


def l2_norm(X):
    return np.linalg.norm(X, ord=2)

def soft_thresholding(y, mu):
    """
    Soft thresholding operator as explained in Section 6.5.2 of https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    Solves the following problem:
    argmin_x (1/2)*||x-y||_F^2 + mu*||x||_1

    Parameters
    ----------
        y : np.ndarray
            Target vector/matrix
        mu : float
            Penalty parameter
    Returns
    -------
        x : np.ndarray
            argmin solution
    """
    return np.sign(y) * np.clip(np.abs(y) - mu, a_min=0, a_max=None)


def singular_value_thresholding(y, tau):
    """
    SVD shrinakge operator as explained in Theorem 2.1 of https://statweb.stanford.edu/~candes/papers/SVT.pdf
    Solves the following problem:
    argmin_x (1/2)*||x-y||_F^2 + tau*||x||_*

    Parameters
    ----------
        y : np.ndarray
            Target vector/matrix
        tau : float
            Penalty parameter
    Returns
    -------
        x : np.ndarray
            argmin solution

    """
    U, s, Vt = np.linalg.svd(y, full_matrices=False)
    s_t = soft_thresholding(s, tau)
    return U @ np.diag(s_t) @ Vt


class IALM:
    """
    Parameters
    ----------
        rho:
            learning rate used for updating mu
        tau:
            parameter used for ADMM update of mu
        max_iter:
            max number of iterations for the algorithm to run
        primal_tol:
            tolerance for the primal residual
        dual_tol:
            tolerance for the dual residual

    """
    def __init__(self, rho = 1.6, tau = 10,
                 mu_update_method = 'admm',
                 benchmark_method = 'rmse',
                 max_iter = 100, primal_tol = 1e-7, 
                 dual_tol = None, debug = False, benchmark = False,
                 show_progress = False, print_skip = None):
        '''
        Choice of default parameters:
            rho: https://arxiv.org/abs/1009.5055
            primal_tol: https://arxiv.org/abs/1009.5055
            dual_tol: https://arxiv.org/abs/1009.5055
        '''

        assert rho > 1
        assert tau > 1

        self.rho_ = rho
        self.max_iter_ = max_iter

        # mu update method
        self.mu_update_method_ = mu_update_method
        self.tau_ = tau # only used if mu_update_method = 'admm'

        # Tolerance
        if dual_tol is None: dual_tol = primal_tol * 100
        self.primal_tol_ = primal_tol
        self.dual_tol_ = dual_tol

        # Logging
        self.prog_step_skip_ = print_skip
        self.show_progress_ = show_progress
        self.is_debug_ = debug
        if self.show_progress_: self.is_debug_ = True
        if self.show_progress_ and self.prog_step_skip_ is None:
            self.prog_step_skip_ = max(1, int(self.max_iter_ / 100)) * 10
        self.logger_   = CustomLogger(__name__, is_debug = self.is_debug_)

        # flag the benchmarking, requires Ytrue for calculating errors
        self.is_benchmark_ = benchmark
        self.benchmark_method_ = benchmark_method


    def fit(self, X, lmb = None, mask = None,
            mu_0 = None,
            Xtrue = None):
        """
        Solves robust PCA using Inexact ALM as explained in Algorithm 5 of 
        Lin et. al. https://arxiv.org/abs/1009.5055
        (same as in Algorithm 12 of https://people.eecs.berkeley.edu/~elghaoui/Teaching/EE227A/reportRobustPCA.pdf).
        Returns the low-rank and sparse components
        Parameters
        ----------
            X:
                Original data matrix
            lmb: 
                penalty on E
            mu_0:
                initial value for the penalty on the squared Frobenius norm of (X - L - E)
        Returns
        -------
            self.L_:
                Low rank component of X
            self.E_:
                Sparse error component of X
        """
        assert X.ndim == 2
        X_fnorm = frobenius_norm(X)
        X_l1norm = l1_norm(X)
        X_l2norm = l2_norm(X)

        # set lambda
        if lmb is None:
            # see Sec. 4 (p.21) of Candes et. al. (https://arxiv.org/abs/0912.3599)
            lmb = 1. / np.sqrt(np.max(X.shape))

        # set mu_0
        mu = mu_0
        if mu is None: 
            # see "Choosing Parameters" in Sec. 4 of Lin et. al. (https://arxiv.org/abs/1009.5055)
            mu = 1.25 / X_l2norm 
            # Alternate definition
            # see p.29 of Candes et. al. (https://arxiv.org/abs/0912.3599)
            # mu = 0.25 * np.prod(X.shape) / X_l1norm

        self.logger_.debug(f"Fit RPCA using IALM (mu update {self.mu_update_method_}, lamba = {lmb:.4f})")

        # set initial Y
        # The function J() required for initialization of dual variables as advised in Section 3.1 of
        # https://people.eecs.berkeley.edu/~yima/matrix-rank/Files/rpca_algorithms.pdf
        # See also eq. 10 of Lin et. al. (https://arxiv.org/abs/1009.5055).
        JX = max(X_l2norm, np.max(np.abs(X)) / lmb)
        Y = X / JX
        #
        # set initial E
        E = np.zeros_like(X)
        E_last = np.empty_like(E)
        #
        # set initial L
        L = singular_value_thresholding(X - E + Y/mu, 1/mu)
        #
        # set mask
        self.mask_ = mask
        E = self._get_masked(E)

        r, h = self._get_residuals(X, L, E, E, mu, X_fnorm)
        self.mu_list_ = [mu]
        self.primal_res_ = [r]
        self.dual_res_ = [h]
        self.cpu_time_ = [1e-8] # do not use zero to avoid log10 error
        # Benchmarking
        if self.is_benchmark_:
            assert Xtrue is not None
            assert Xtrue.shape == X.shape
            self.rmse_ = [merr.get(Xtrue, L, method = self.benchmark_method_)]

        cpu_time_last = time.process_time()
        for k in range(self.max_iter_):
            # Update E first, according to Sec.4 
            # Solve argmin_S ||X - (L + E) + Y/mu||_F^2 + (lmb/mu)*||S||_1
            E_last = E.copy()
            E = soft_thresholding(X - L + Y/mu, lmb/mu)
            E = self._get_masked(E)

            # Solve argmin_L ||X - (L + E) + Y/mu||_F^2 + (lmb/mu)*||L||_*
            L = singular_value_thresholding(X - E + Y/mu, 1/mu)

            # Update dual variables Y <- Y + mu * (X - S - L)
            Y += mu * (X - E - L)
            r, h = self._get_residuals(X, L, E, E_last, mu, X_fnorm)
            self.primal_res_.append(r)
            self.dual_res_.append(h)

            # Time
            cpu_time = time.process_time()
            self.cpu_time_.append(cpu_time - cpu_time_last)
            cpu_time_last = cpu_time

            # Benchmark
            if self.is_benchmark_:
                self.rmse_.append(merr.get(Xtrue, L, method = self.benchmark_method_))

            if self.show_progress_:
                if (k % self.prog_step_skip_ == 0):
                    self.logger_.info(f"Iteration {k}. Primal residual {r:g}. Dual residual {h:g}")

            # Check stopping cirteria
            if r < self.primal_tol_ and h < self.dual_tol_:
                break

            # Update mu
            mu = self._update_mu(mu, r, h)
            self.mu_list_.append(mu)

        self.L_ = L
        self.E_ = E
        return


    def _get_masked(self, X):
        if self.mask_ is None or X is None:
            return X
        else:
            # Numpy casts True to 1 and False to 0
            # Here, we need to set True to 0 and False to 1
            return X * ~self.mask_


    @staticmethod
    def _get_residuals(X, L, E, E_last, mu, X_frobnorm):
        primal_residual = frobenius_norm(X - L - E) / X_frobnorm
        dual_residual = mu * frobenius_norm(E - E_last) / X_frobnorm
        return primal_residual, dual_residual


    def _update_mu(self, mu, r, h):
        """
        mu_update_method
        ----------------
            'admm': update from Boyd et. al. (https://stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf)
                ADMM modification of the IALM algorithm. No idea if it is good or bad.
            'ialm': update from Lin et. al. (https://arxiv.org/abs/1009.5055)
        Parameters
        ----------
            mu: 
                current value of mu
            r:
                primal residual
            h: 
                dual residual
        Returns
        -------
            mu_new:
                updated value of mu
        """
        if self.mu_update_method_ == 'admm':
            if r > self.tau_ * h:
                return mu * self.rho_
            elif h > self.tau_ * r:
                return mu / self.rho_
            else:
                return mu

        elif self.mu_update_method_ == 'ialm':
            if h < self.dual_tol_:
                return mu * self.rho_
            else:
                return mu
