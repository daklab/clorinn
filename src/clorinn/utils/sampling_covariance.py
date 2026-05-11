"""
sampling_covariance.py
----------------------
Constructs and validates the sampling covariance matrix A used by the
NNM-Corr objective.

A is the N x N matrix of cross-trait sampling covariances on the Z-score
scale, assembled from LD Score Regression intercepts.
The diagonal entries are univariate LDSC intercepts; 
the off-diagonal entries are bivariate LDSC intercepts.

Because A is assembled from many noisy pairwise estimates, it is not
guaranteed to be positive definite (PD). The solvers require A ≻ 0 for
the Cholesky decomposition and the Mahalanobis gradient to be well-defined.
SamplingCovariance validates the input, optionally repairs it to the
nearest PD matrix via Higham's algorithm, and exposes the result via
a clean interface consumed by CovarianceOperator.

Constructors
------------
SamplingCovariance.from_matrix(A, ...)
    Construct from a pre-assembled N x N matrix.

SamplingCovariance.from_ldsc(...)
    Construct from LDSC intercept estimates.  Not yet implemented.

References
----------
.. [1] Higham, N. J. (2002).  Computing the nearest correlation matrix – 
       a problem from finance.  *IMA Journal of Numerical Analysis*, 
       **22**(3), 329–343. https://doi.org/10.1093/imanum/22.3.329
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import warnings
import logging
import numpy as np

from .logs import get_logger


class SamplingCovariance:
    """
    Validated positive-definite sampling covariance matrix.

    Do not instantiate directly.  Use the class-method constructors:

        SamplingCovariance.from_matrix(A)

    Parameters are documented on the constructors.

    Attributes
    ----------
    A_ : np.ndarray [shape (n, n)]
        Validated (and possibly repaired) positive-definite covariance
        matrix.

    n_ : int
        Number of traits.

    is_repaired_ : bool
        True if the input required PD repair.

    repair_info_ : dict or None
        Diagnostic information from the repair step, or None if repair
        was not applied.  Keys:
            n_iter          int     Number of Higham iterations.
            converged       bool    Whether Higham converged.
            min_eig_input   float   Smallest eigenvalue before repair.
            min_eig_output  float   Smallest eigenvalue after repair.
            reg_applied     float   Diagonal regularisation added.
    """

    def __init__(self, A, is_repaired, repair_info, logger):
        self.A_          = A
        self.n_          = A.shape[0]
        self.is_repaired_ = is_repaired
        self.repair_info_ = repair_info
        self.logger_      = logger


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def A(self):
        """Validated positive-definite covariance matrix."""
        return self.A_

    @property
    def n(self):
        """Number of traits."""
        return self.n_

    @property
    def is_repaired(self):
        """True if the input required PD repair."""
        return self.is_repaired_

    @property
    def repair_info(self):
        """Diagnostic dict from the repair step, or None."""
        return self.repair_info_


    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(cls, A,
        repair = True, reg = None, verbose = None,
        eig_tol = 0.0, conv_tol = 1e-7, max_iter = 100):

        """
        Construct a SamplingCovariance from a pre-assembled matrix.

        Parameters
        ----------
        A : np.ndarray [shape (n, n)]
            Raw sampling covariance matrix, typically assembled from
            LDSC intercepts.  Need not be positive definite on input.

        repair : bool, default=True
            If True, apply Higham's nearest-PD algorithm when A is not
            already strictly positive definite.  If False, raise
            ValueError for a non-PD input.

        reg : float or None, default=None
            Minimum diagonal regularisation added after Higham repair to
            guarantee strict PD for Cholesky.  If None, a machine-epsilon
            based default is used:  eps * n * ||A||_2.

        eig_tol : float, default=0.0
            Eigenvalues below this threshold are clipped to zero during
            the PSD projection step of Higham's algorithm.

        conv_tol : float, default=1e-7
            Convergence tolerance for Higham's alternating projections.

        max_iter : int, default=100
            Maximum number of Higham iterations.

        verbose : int or None, default=None
            Controls the verbosity of solver output.  Three levels (or None) are
            recognised:
                None  Inherit loglevel, do not change anything.
                0     Silent.  Only warnings and errors are reported.
                      Equivalent to logging.WARN.
                1     Progress.  Equivalent to logging.INFO.
                      Logs iteration count, step size, and duality gap at convergence.
                2     Debug.  Equivalent to logging.DEBUG.
                      Logs all debugging outputs.
            Higher values are treated as 2.

        Returns
        -------
        SamplingCovariance

        Raises
        ------
        ValueError
            If A is not square, contains NaN/Inf, or is non-PD and
            repair=False.
        """
        logger = get_logger(__name__, verbose=verbose, scope="solver")

        A = np.asarray(A, dtype=float)
        cls._validate_shape(A)
        cls._validate_finite(A)

        n = A.shape[0]
        logger.debug(f"SamplingCovariance.from_matrix: shape=({n}, {n})")

        is_pd, min_eig = cls._check_pd(A)

        if is_pd:
            logger.debug(
                f"Input is already PD (min eigenvalue = {min_eig:.4g}). "
                "No repair needed."
            )
            return cls(A, is_repaired=False, repair_info=None, logger = logger)

        logger.debug(f"Input is not PD (min eigenvalue = {min_eig:.4g}).")

        if not repair:
            raise ValueError(
                f"Sampling covariance matrix is not positive definite "
                f"(min eigenvalue = {min_eig:.4g}). "
                "Set repair=True to apply Higham's nearest-PD correction, "
                "or inspect the LDSC outputs for QC issues."
            )

        logger.info(
            f"Repairing sampling covariance matrix "
            f"(min eigenvalue = {min_eig:.4g}) via Higham's algorithm."
        )

        A_repaired, info = cls._nearest_pd_higham(
            A,
            reg      = reg,
            eig_tol  = eig_tol,
            conv_tol = conv_tol,
            max_iter = max_iter,
        )

        if not info['converged']:
            warnings.warn(
                f"nearest_pd did not converge in {max_iter} iterations "
                f"(final relative change reported in repair_info). "
                "Consider increasing max_iter or relaxing conv_tol.",
                RuntimeWarning,
                stacklevel=2,
            )

        logger.info(
            f"Repair complete: min eigenvalue "
            f"{info['min_eig_input']:.4g} -> {info['min_eig_output']:.4g}, "
            f"reg_applied = {info['reg_applied']:.4g}, "
            f"n_iter = {info['n_iter']}, "
            f"converged = {info['converged']}."
        )

        return cls(A_repaired, is_repaired=True, repair_info=info, logger = logger)


    @classmethod
    def from_ldsc(cls, *args, **kwargs):
        """
        Construct from LDSC intercept estimates.

        Not yet implemented.  Will accept univariate LDSC intercepts on
        the diagonal and bivariate LDSC intercepts on the off-diagonal,
        assemble A, and pass through from_matrix.
        """
        raise NotImplementedError(
            "SamplingCovariance.from_ldsc is not yet implemented."
        )


    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_shape(A):
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError(
                f"A must be a square 2-D array; got shape {A.shape}."
            )

    @staticmethod
    def _validate_finite(A):
        if not np.isfinite(A).all():
            raise ValueError(
                "Sampling covariance matrix contains NaN or Inf values. "
                "Inspect pairwise LDSC outputs before attempting PD repair."
            )

    @staticmethod
    def _check_pd(A):
        """
        Return (is_pd, min_eigenvalue).

        Uses eigvalsh for the min eigenvalue (symmetric path), then
        attempts Cholesky as the definitive PD check.
        """
        min_eig = float(np.linalg.eigvalsh(A).min())
        try:
            np.linalg.cholesky(A)
            return True, min_eig
        except np.linalg.LinAlgError:
            return False, min_eig

    @staticmethod
    def _nearest_pd_higham(
        A,
        reg      = None,
        eig_tol  = 0.0,
        conv_tol = 1e-7,
        max_iter = 100,
    ):
        """
        Higham's nearest positive-definite matrix algorithm with
        Dykstra correction and diagonal regularisation.

        Returns
        -------
        Ahat : np.ndarray    Nearest PD matrix.
        info : dict          Diagnostic information.
        """
        n = A.shape[0]

        B = (A + A.T) / 2.0
        min_eig_input = float(np.linalg.eigvalsh(B).min())

        # Fast-path: already strictly PD
        try:
            np.linalg.cholesky(B)
            info = dict(
                n_iter        = 0,
                converged     = True,
                min_eig_input = min_eig_input,
                min_eig_output= min_eig_input,
                reg_applied   = 0.0,
            )
            return B, info
        except np.linalg.LinAlgError:
            pass

        # Dykstra-corrected alternating projections
        dS         = np.zeros_like(B)
        Y          = B.copy()
        converged  = False
        rel_change = np.inf

        for k in range(max_iter):
            R = Y - dS
            eigvals, eigvecs     = np.linalg.eigh(R)
            eigvals_clipped      = np.maximum(eigvals, eig_tol)
            X                    = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
            dS                   = X - R
            Y_new                = (X + X.T) / 2.0
            rel_change           = (
                np.linalg.norm(Y_new - Y, 'fro')
                / max(1.0, np.linalg.norm(Y, 'fro'))
            )
            Y = Y_new
            if rel_change < conv_tol:
                converged = True
                break

        # Diagonal regularisation to ensure strict PD
        if reg is None:
            eps = np.finfo(float).eps
            reg = eps * n * float(np.linalg.norm(Y, 2))

        Ahat        = Y.copy()
        reg_applied = 0.0
        reg_try     = max(reg, np.finfo(float).tiny)

        for _ in range(100):
            try:
                np.linalg.cholesky(Ahat)
                break
            except np.linalg.LinAlgError:
                reg_applied = reg_try
                Ahat        = Y + reg_try * np.eye(n)
                reg_try    *= 10.0
        else:
            raise RuntimeError(
                "nearest_pd_higham: Cholesky decomposition failed after "
                "100 regularisation doublings. The input may be severely "
                "ill-conditioned or contain near-NaN values."
            )

        min_eig_output = float(np.linalg.eigvalsh(Ahat).min())

        info = dict(
            n_iter         = k + 1,
            converged      = converged,
            min_eig_input  = min_eig_input,
            min_eig_output = min_eig_output,
            reg_applied    = reg_applied,
        )
        return Ahat, info
