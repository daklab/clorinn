#!/usr/bin/env python

"""
Matrix factorization of the low-rank estimate from convex optimization.

Given the low-rank matrix X estimated by one of the Clorinn algorithms
(NNM, NNM-Sparse, or RPCA), this module extracts factors and loadings
via truncated SVD or related factorization methods.

The factor model follows the convention in Banerjee et al.:

    X = L F^T

where F := V is an orthonormal matrix of hidden factors (P x K)
and L := U diag(S) is the matrix of trait loadings (N x K).
F being orthonormal ensures that the squared cosine scores and
contribution scores have valid geometric interpretations.
"""
# Author: Saikat Banerjee

import numpy as np


class MatrixFactorization():

    """
    Parameters
    ----------
    method : string, default='svd'
        Factorization method. Options include:
            'svd' : Truncated singular value decomposition.

    k : integer or None, default=None
        Number of factors to retain. If None, it is determined
        automatically from the singular value spectrum using
        `sv_threshold`.

    sv_threshold : float, default=1e-6
        Singular values below `sv_threshold * s_1` are discarded
        when `k` is None, where `s_1` is the largest singular value.

    -----
    The orthonormality of F is required for valid interpretation
    of the squared cosine scores and contribution scores.
    The loadings L carry all of the scale from the singular values.
    The loadings can be interpreted as the projection of X on the
    factors, i.e., L = X F.

    Usage
    -----
        mf = MatrixFactorization(k = 10)
        mf.fit(X)
        L = mf.L
        F = mf.F
        cos2_traits = mf.cos2_traits
        contr_traits = mf.contribution_traits
        cos2_variants = mf.cos2_variants
        contr_variants = mf.contribution_variants
    """

    def __init__(self, method = 'svd', k = None, sv_threshold = 1e-6):
        self.method_ = method
        self.k_ = k
        self.sv_threshold_ = sv_threshold
        return


    @property
    def U(self):
        """Left singular vectors, shape (N, k)."""
        return self.U_


    @property
    def s(self):
        """Singular values, shape (k,)."""
        return self.s_


    @property
    def V(self):
        """Right singular vectors, shape (P, k)."""
        return self.V_


    @property
    def L(self):
        """Trait loadings L = U diag(S), shape (N, k)."""
        return self.L_


    @property
    def F(self):
        """Orthonormal hidden factors F = V, shape (P, k)."""
        return self.F_


    @property
    def cos2_traits(self):
        return self.cos2_traits_


    @property
    def cos2_variants(self):
        return self.cos2_variants_


    @property
    def contribution_traits(self):
        return self.contr_traits_


    @property
    def contribution_variants(self):
        return self.contr_variants_


    def fit(self, X):
        if self.method_ == 'svd':
            self._fit_svd(X)
        else:
            raise ValueError(f"Unknown factorization method: '{self.method_}'")

        self._compute_scores()
        return


    def _fit_svd(self, X):
        U, s, Vh = np.linalg.svd(X, full_matrices = False)

        # Determine the number of factors
        k = self.k_
        if k is None:
            k = int(np.sum(s > self.sv_threshold_ * s[0]))
            k = max(k, 1)

        self.U_, self.s_, self.Vh_ = U, s, Vh
        U_k  = U[:, :k]
        s_k  = s[:k]
        Vh_k = Vh[:k, :].T
        self.L_  = U_k * s_k     # (N, k)
        self.F_  = Vh_k          # (P, k)
        return


    def compute_cos(self, X):
        """
        Divide by sum of factors for a given trait/variant -- row totals.
        Gives importance of a factor k for trait n / variant p:
        which factor contributes most to a trait?

            cos2[n, k] = L[n,k]^2 / sum_k L[n,k]^2
            cos2[p, k] = F[p,k]^2 / sum_k F[p,k]^2

        """
        X2 = X ** 2
        return X2 / np.sum(X2, axis = 1, keepdims = True)


    def compute_contribution(self, X):
        """
        Divide by sum of traits/variants for a given factor -- column totals.
        How much a trait n or variant p contributes to a factor k?

            contr[n, k] = L[n,k]^2 / sum_n L[n,k]^2
            contr[p, k] = F[p,k]^2 / sum_p F[p,k]^2

        """
        X2 = X ** 2
        return X2 / np.sum(X2, axis = 0, keepdims = True)


    def _compute_scores(self):
        self.cos2_traits_    = self.compute_cos(self.L_)
        self.cos2_variants_  = self.compute_cos(self.F_)
        self.contr_traits_   = self.compute_contribution(self.L_)
        self.contr_variants_ = self.compute_contribution(self.F_)
        return
