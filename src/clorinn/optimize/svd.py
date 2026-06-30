"""
Top singular vectors and the nuclear norm oracle for Frank-Wolfe solvers.

This module provides two things:

  TopCompSVD          Computes the top left and right singular vectors of a
                      matrix using one of four backends: power iteration,
                      left-Gram eigendecomposition, randomised SVD, or direct
                      full SVD.

  nuclear_norm_oracle Wraps TopCompSVD to produce the Frank-Wolfe linear
                      minimisation oracle over the nuclear norm ball.

TopCompSVD is an implementation class; external code should interact with
it only through nuclear_norm_oracle.  The class is kept separate so that
the solver can manage warm-start state (u0, v0) between iterations without
nuclear_norm_oracle needing to be stateful.
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import logging
from sklearn.utils.extmath import randomized_svd

_logger = logging.getLogger(__name__)


def nuclear_norm_oracle(
        G, radius, 
        method = 'power',
        max_iter = None,
        warm_start = False, 
        u0 = None,
        v0 = None
    ):
    """
    Linear minimisation oracle over the nuclear norm ball.
 
    Solves the Frank-Wolfe linear subproblem
 
        S* = argmin_{||S||_* <= radius}  Tr(G^T S)
           = -radius * u1 * v1^T,
 
    where u1, v1 are the top left and right singular vectors of G.  This is
    the fundamental oracle used by all Frank-Wolfe variants (standard FW,
    away-step FW) for the nuclear-norm constrained problems.
 
    Parameters
    ----------
    G : np.ndarray [size (n, p); dtype: float]
        Gradient matrix at the current iterate.
 
    radius : float
        Nuclear norm constraint radius r.  Must be positive.
 
    method : string, default='left-gram'
        Algorithm used to compute the top singular pair of G.
        For options, see TopCompSVD.
 
    max_iter : integer or None, default=None
        Maximum number of iterations for the 'power' method. Ignored for
        all other methods. If None, TopCompSVD uses its own default of 100.

    warm_start: boolean, default=False
        Whether to use warm-start for the 'power' method. Ignored for
        all other methods.
 
    u0 : np.ndarray [size (n,) or (n, 1)], default=None
        Warm-start left singular vector for the 'power' method. Ignored for
        all other methods. If None, a random initialisation is used.
 
    v0 : np.ndarray [size (p,) or (1, p)], default=None
        Warm-start right singular vector for the 'power' method. Ignored for
        all other methods. If None, v0 is derived from u0 via one matrix
        product.
 
    Returns
    -------
    S : np.ndarray [size (n, p)]
        Oracle output: the vertex of the nuclear norm ball that minimises
        the linear objective Tr(G^T S). Satisfies ||S||_* == radius.
 
    u1 : np.ndarray [size (n, 1)]
        Top left singular vector of G used to construct S.  Returned for
        use as a warm-start (u0) in the next iteration, and for seeding
        the AFW active set.
 
    v1t : np.ndarray [size (1, p)]
        Top right singular vector of G used to construct S.  Returned for
        the same reasons as u1.
 
    n_iter : integer
        Number of iterations actually performed by the underlying SVD
        method. Always 1 for 'left-gram' and 'direct'. Useful for
        diagnosing convergence of the 'power' method.
 
    Notes
    -----
    The oracle is model-agnostic: the gradient G encodes the model-specific
    loss (isotropic Frobenius, masked Frobenius, or Mahalanobis) and the
    oracle itself is identical across NNM, NNM-Sparse, and NNM-Corr.
 
    Examples
    --------
    >>> G = np.random.randn(10, 500)
    >>> S, u1, v1t, n_iter = nuclear_norm_oracle(G, radius=50.0)
    >>> np.linalg.norm(S, 'nuc')   # equals radius
    50.0
    """
    _max_iter = 100 if max_iter is None else max_iter
    svd = TopCompSVD(method = method, max_iter = _max_iter)
    if warm_start and method == 'power':
        svd.fit(G, u0 = u0, v0 = v0)
    else:
        svd.fit(G)

    S = -radius * svd.u1 @ svd.v1_t
    return S, svd.u1, svd.v1_t, svd.n_iter


class TopCompSVD():
    """
    Top singular-vector solver for a single left-right pair.

    Computes approximate or exact top left and right singular vectors
    u1 (n, 1) and v1t (1, p) of an input matrix X such that
    u1^T X v1t is the largest singular value.  Four backends are
    available; the choice trades accuracy, determinism, and cost.

    Parameters
    ----------
    method : string, default='power'
        Backend algorithm.  Options:

            'power'       Power iteration.  Iterative and stochastic unless
                          warm-started.  Scales to large n and p; cost per
                          iteration is O(np).  Supports warm start via u0/v0
                          in fit().

            'left-gram'   Eigendecomposition of X X^T.  Exact and fully
                          deterministic.  Preferred when n << p (e.g. GWAS
                          matrices with n traits and p SNPs), where the Gram
                          matrix is small (n x n) and eigh() is cheap.
                          Cost O(n^2 p + n^3).

            'randomized'  Randomized SVD (sklearn).  One-shot approximation;
                          fixed random seed for reproducibility.  Useful when
                          neither n nor p is small.

            'direct'      Full SVD via numpy.linalg.svd.  Exact but
                          O(min(n,p)^3); use only for small matrices or
                          numerical reference checks.

    max_iter : integer, default=10
        Maximum number of power iterations for method='power'.  Ignored
        for all other methods.

    Attributes
    ----------
    u1 : np.ndarray [size (n, 1)]
        Top left singular vector.  Available after fit().

    v1_t : np.ndarray [size (1, p)]
        Top right singular vector, stored as a row vector.  Available
        after fit().

    n_iter : integer
        Number of iterations actually performed.  For 'power', this is
        between 1 and max_iter.  Always 1 for 'left-gram' and 'direct'.

    Notes
    -----
    The sign convention u1^T X v1t >= 0 is enforced by 'left-gram' and
    'direct'.  The 'power' method does not enforce a sign convention;
    the caller should be sign-agnostic or normalise externally.

    The 'power' method is stochastic when u0=None: it draws a random
    initial vector from N(0, I).  Pass u0 and v0 from the previous
    call to fit() to warm-start and obtain deterministic behaviour after
    the first call.
    """
    def __init__(self, method = 'power', max_iter = 10):
        self._method = method
        self._max_iter = max_iter
        self._n_iter = 0 # number of iterations actually used
        return

    def fit(self, X, u0 = None, v0 = None, tol = 1e-8):
        if self._method == 'power':
            self._u1, self._v1h = self.fit_power(X, u0 = u0, v0 = v0, tol = tol)
        elif self._method == 'left-gram':
            self._u1, self._v1h = self.fit_left_gram(X)
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

        return u1, v1h


    def fit_left_gram(self, X, tol=1e-12):
        """
        Exact top singular vectors via eigendecomposition of X X^T.

        Best when X is tall-skinny in the *row* dimension, i.e. n << p.
        For GWAS matrices with shape (n_traits, n_snps), this is ideal when
        n_traits is small and n_snps is large.
        """
        n, p = X.shape

        # Small symmetric Gram matrix
        G = X @ X.T
        G = 0.5 * (G + G.T)  # enforce symmetry numerically

        # Largest eigenpair of X X^T
        evals, U = np.linalg.eigh(G)
        idx = np.argmax(evals)

        lam1 = float(evals[idx])
        lam1 = max(lam1, 0.0)  # guard against tiny negative roundoff
        sigma1 = np.sqrt(lam1)

        u = U[:, idx]

        # Recover v from v = X^T u / sigma
        if sigma1 <= tol:
            # X is numerically zero: return zero direction so S = 0
            v = np.zeros(p, dtype=X.dtype)
        else:
            v = X.T @ u
            nv = np.linalg.norm(v)

            if nv <= tol:
                v = np.zeros(p, dtype=X.dtype)
            else:
                v /= nv

                # Sign convention: make u^T X v nonnegative
                if float(u.T @ X @ v) < 0:
                    u = -u
                    v = -v

        self._n_iter = 1
        u1 = u.reshape(-1, 1)
        v1h = v.reshape(1, -1)
        return u1, v1h


    def fit_randomized(self, X):
        u, s, vh = randomized_svd(X,
                        n_components = 1, n_iter = self._max_iter,
                        power_iteration_normalizer = 'none',
                        random_state = 0)
        self._n_iter = self._max_iter
        return u, vh


    def fit_direct(self, X):
        u, s, vh = np.linalg.svd(X, full_matrices=False)
        u1  = u[:, [0]]
        v1h = vh[[0], :]
        self._n_iter = 1
        return u1, v1h


    @property
    def u1(self):
        return self._u1

    @property
    def v1_t(self):
        return self._v1h

    @property
    def n_iter(self):
        return self._n_iter
