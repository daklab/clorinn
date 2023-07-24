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
        return

    def fit(self, X):
        if self._method == 'power':
            self._u1, self._v1h = self.fit_power(X)
        elif self._method == 'randomized':
            self._u1, self._v1h = self.fit_randomized(X)
        return

    def fit_power(self, X):
        n, p = X.shape
        u = np.random.normal(0, 1, n)
        u /= np.linalg.norm(u)
        v = X.T.dot(u)
        v /= np.linalg.norm(v)
        for _ in range(self._max_iter):
            u = X.dot(v)
            u /= np.linalg.norm(u)
            v = X.T.dot(u)
            v /= np.linalg.norm(v)
        return u.reshape(-1, 1), v.reshape(1, -1)

    def fit_randomized(self, X):
        u, s, vh = sp.randomized_svd(X,
                        n_components = 1, n_iter = self._max_iter,
                        power_iteration_normalizer = 'none',
                        random_state = 0)
        return u, vh

    @property
    def u1(self):
        return self._u1

    @property
    def v1_t(self):
        return self._v1h
