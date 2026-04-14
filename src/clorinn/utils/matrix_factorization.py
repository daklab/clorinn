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

References
----------
    Banerjee, S. et al. Convex approaches to isolate the shared and
    distinct genetic components of complex traits. (2025).
    Section 2.5: Characterization of hidden factors.
"""
# Author: Saikat Banerjee

import numpy as np


class MatrixFactorization():

    """
    Factor extraction from low-rank matrices estimated by Clorinn.

    Given X = U D V^T (thin SVD), the class computes:
        - Factors:  F = V       (P x K, orthonormal)
        - Loadings: L = U D     (N x K)
    such that X ≈ L F^T, and provides squared cosine scores
    and contribution scores for downstream interpretation.

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

    Notes
    -----
    The orthonormality of F is required for valid interpretation
    of the squared cosine scores (Eq. 18) and contribution scores
    (Eq. 19) as defined in the Clorinn manuscript.

    The loadings L carry all of the scale from the singular values.
    The loadings can be interpreted as the projection of X on the
    factors, i.e., L = X F.

    Usage
    -----
        mf = MatrixFactorization(k = 10)
        mf.fit(X)
        L = mf.L
        F = mf.F
        cos2 = mf.cos2_score
        contr = mf.contribution_score
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
    def Vh(self):
        """Right singular vectors (transposed), shape (k, P)."""
        return self.Vh_


    @property
    def L(self):
        """Trait loadings L = U diag(S), shape (N, k)."""
        return self.L_


    @property
    def F(self):
        """Orthonormal hidden factors F = V, shape (P, k)."""
        return self.F_


    @property
    def cos2_score(self):
        """
        Squared cosine scores, shape (N, k).

        cos2[j, i] measures the importance of factor i for trait j,
        defined as the fraction of trait j's total squared loading
        attributable to factor i:

            cos2[j, i] = L[j,i]^2 / sum_i L[j,i]^2

        Rows sum to 1 (up to numerical precision).
        """
        return self.cos2_score_


    @property
    def contribution_score(self):
        """
        Contribution scores, shape (N, k).

        contr[j, i] measures the importance of trait j for factor i,
        defined as trait j's share of the total loading on factor i:

            contr[j, i] = L[j,i]^2 / sum_j L[j,i]^2

        Columns sum to 1 (up to numerical precision).
        """
        return self.contribution_score_


    def fit(self, X):
        """
        Extract factors and loadings from the low-rank matrix X.

        Parameters
        ----------
        X : np.ndarray [size (N, P); dtype: float]
            Low-rank matrix estimated by one of the Clorinn algorithms
            (e.g., FrankWolfe, IALM).
        """
        if self.method_ == 'svd':
            self._fit_svd(X)
        else:
            raise ValueError(f"Unknown factorization method: '{self.method_}'")

        self._compute_scores()
        return


    def _fit_svd(self, X):
        """
        Truncated SVD factorization.

        Computes the thin SVD of X and retains the top k components.
        """
        U, s, Vh = np.linalg.svd(X, full_matrices = False)

        # Determine the number of factors
        k = self.k_
        if k is None:
            k = int(np.sum(s > self.sv_threshold_ * s[0]))
            k = max(k, 1)

        self.U_, self.s_, self.Vh_ = U, s, Vh
        U_k  = U[:, :k]
        s_k  = s[:k]
        Vh_k = Vh[:k, :]
        self.L_  = U_k * s_k     # (N, k)
        self.F_  = Vh_k.T            # (P, k)
        return


    def _compute_scores(self):
        """
        Compute squared cosine scores and contribution scores
        from the loadings matrix L.
        """
        L2 = np.square(self.L_)

        # Squared cosine scores: importance of factor i for trait j
        row_totals = np.sum(L2, axis = 1, keepdims = True)
        row_totals = np.maximum(row_totals, 1e-30)
        self.cos2_score_ = L2 / row_totals

        # Contribution scores: importance of trait j for factor i
        col_totals = np.sum(L2, axis = 0, keepdims = True)
        col_totals = np.maximum(col_totals, 1e-30)
        self.contribution_score_ = L2 / col_totals
        return
