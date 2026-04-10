"""
Power method to obtain the left and right singular vectors 
corresponding to the highest singular value
"""
# Author: Saikat Banerjee

import numpy as np
from sklearn.utils.extmath import randomized_svd

class TopCompSVD():

    def __init__(self, method = 'power', max_iter = 10):
        self._method = method
        self._max_iter = max_iter
        self._n_iter = 0 # number of iterations actually used
        return

    def fit(self, X, u0 = None, v0 = None, tol = 1e-8):
        if self._method == 'power':
            self._u1, self._v1h = self.fit_power(X, u0 = u0, v0 = v0, tol = tol)
        elif self._method == 'randomized':
            self._u1, self._v1h = self.fit_randomized(X)
        elif self._method == 'direct':
            self._u1, self._v1h = self.fit_direct(X)
        return

    def fit_power(self, X, u0 = None, v0 = None, tol = 1e-8):
        n, p = X.shape

        # Initialize u and guard against zero norm
        if u0 is None:
            u = np.random.normal(0.0, 1.0, n)
        else:
            u = u0.ravel().copy()

        nu = np.linalg.norm(u)
        if nu == 0:
            u = np.random.normal(0.0, 1.0, n)
            nu = np.linalg.norm(u)
        u /= nu

        # Initialize v and guard against zero norm
        if v0 is None:
            v = X.T.dot(u)
        else:
            v = v0.ravel().copy()

        nv = np.linalg.norm(v)
        if nv == 0:
            v = X.T.dot(u)
            nv = np.linalg.norm(v)
        v /= nv

        sigma_old = None
        n_iter = 0
        for _ in range(self._max_iter):
            n_iter += 1
            # even though we have max_iter, we can break early
            # if s or ||Xv - su|| or ||X.T.u - sv|| are below tolerance
            # s = u.T X v, and can be approximated cheaply.
            #
            Xu = X.dot(v)
            nu = np.linalg.norm(Xu)
            u  = Xu / nu

            XTu = X.T.dot(u)
            nv  = np.linalg.norm(XTu)
            v   = XTu / nv

            sigma = nu # or nv
            if sigma_old is not None and abs(sigma - sigma_old) <= tol * max(1.0, abs(sigma)):
                break
            sigma_old = sigma


        u1 = u.reshape(-1, 1)
        v1h = v.reshape(1, -1)
        self._n_iter = n_iter
        #sigma = u1.T @ X @ v1h.T
        #self._sigma = float(self._u1.T @ X @ self._v1h.T)

        #ru = np.linalg.norm(X @ v1h.T - sigma * u1)
        #rv = np.linalg.norm(X.T @ u1 - sigma * v1h.T)
        #self._resid = max(ru, rv)

        return u1, v1h

    def fit_randomized(self, X):
        u, s, vh = randomized_svd(X,
                        n_components = 1, n_iter = self._max_iter,
                        power_iteration_normalizer = 'none',
                        random_state = 0)
        return u, vh

    def fit_direct(self, X):
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        return u[:, [0]], vh[[0], :]

    @property
    def u1(self):
        return self._u1

    @property
    def v1_t(self):
        return self._v1h

    @property
    def n_iter(self):
        return self._n_iter
