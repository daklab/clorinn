"""
Active set for the Away-step Frank-Wolfe (AFW) solver on the nuclear norm ball.

Stores the convex combination
    X = sum_k alpha_k S_k,
where each atom is a rank-1 vertex of the ball:
    S_k = -radius * u_k @ vt_k    (non-zero atom; u_k, vt_k unit-norm)
The residual slack alpha_zero = 1 - sum alpha_k represents unused
nuclear-norm budget. It is not an eligible away atom.

Sign convention.
The atoms are stored such that we can recover X. The FW linear oracle
output returns u1 and v1t such that,
    S = -r * u1 @ v1t,
where (u1, v1t) is the dominant singular pair of the gradient. We just
store the vectors as is. However, when seeding from SVD of X0, not of 
any gradient, we need to flip the sign of any one vector. SVD produces
    X0 = U diag(s) Vt = sum_k s_k u_k vt_k
To reconcile with our storage convention, where the active set must
reconstruct to X0, we need 
    alpha_k * S_k = s_k u_k vt_k 
for each atom, i.e., 
    alpha_k * (-r * u1 * v1t) = s_k u_k vt_k. 
With alpha_k = s_k/r > 0, this forces u1 = -u_k (or equivalently 
v1t = -vt_k). The sign flip is mandatory here - without it, 
reconstruction gives -X0, not +X0.

Storage layout.
Internal arrays are dense and packed:
    U_     : (n, K)   left singular vectors of non-zero atoms (unit-norm columns)
    Vt_    : (K, p)   right singular vectors (unit-norm rows; stored transposed)
    alpha_ : (K,)     non-negative weights of non-zero atoms
    alpha_zero_ : float    residual weight of the zero atom
The zero atom is stored as a scalar weight rather than as a column of zeros
in U_/Vt_, so oracle_away() does not need to skip it explicitly. Total weight
is alpha_.sum() + alpha_zero_, which equals 1 by construction.

Hot-path cost (n traits, p SNPs, K active atoms).
    oracle_away(G):    O(K * (n + p))    one batched matmul + one einsum
    update_fw(...):    O(n + p)          single column append
    update_away(k, .): O(K)              in-place weight update
    reconstruct():     O(n * p * K)      dense matmul; mainly diagnostic

"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np
import logging

_logger = logging.getLogger(__name__)


class ActiveSet:
    """
    Active set for AFW on the nuclear norm ball.

    Parameters
    ----------
    radius : float
        Nuclear norm constraint radius. Must be positive.

    Attributes
    ----------
    radius : float
        Nuclear norm constraint radius (read-only).
    n_atoms : int
        Number of currently active non-zero atoms (zero atom not counted).
    has_zero_atom : bool
        Whether a zero atom with positive weight is present.
    """

    def __init__(self, radius, *, shape = None):
        if radius <= 0:
            raise ValueError(f"radius must be positive; got {radius}.")
        self.radius_     = float(radius)
        self.shape_      = shape
        self.U_          = np.zeros((0, 0))     # placeholder; resized on first add
        self.Vt_         = np.zeros((0, 0))
        self.alpha_      = np.zeros(0)
        self.alpha_zero_ = 0.0
        return


    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_svd(cls, X0, radius, *, eps_zero = 1e-12):
        U, s, Vt = np.linalg.svd(X0, full_matrices = False)
        return cls.from_svd_factors(U, s, Vt, radius, eps_zero = eps_zero)


    @classmethod
    def from_svd_factors(cls, U, s, Vt, radius, *, eps_zero = 1e-12):
        """
        Seed from precomputed SVD factors X0 = U diag(s) Vt. 

        - Boundary case (||X0||_* == radius): one atom per significant singular
          triplet, alpha_k = s_k / radius. Weights sum to exactly 1.
        - Interior case (||X0||_* <  radius): same atoms plus a zero atom
          carrying residual weight (1 - ||X0||_* / radius).
        - X0 == 0: single zero atom with weight 1.

        Sign reconciliation: It receives the SVD of X0, not of any gradient.
        So it has to convert the sign to ensure S = -r u vt. We absorb the 
        sign flip into u_k = -U[:, k] at seed time so reconstruct() gives 
        back X0 (not -X0). 

        Parameters
        ----------
        U : np.ndarray, shape (n, K)

        s : np.ndarray, shape (K,)

        Vt : np.ndarray, shape (K, p)

        """
        s = np.asarray(s, dtype = float)
        nuc = float(s.sum())
        if nuc > radius * (1.0 + 1e-9):
            raise ValueError(
                f"Initial iterate not feasible: ||X0||_* = {nuc:g} "
                f"> radius = {radius:g}."
            )

        shape = (U.shape[0], Vt.shape[1])
        instance = cls(radius, shape = shape)

        if s.size > 0:
            s_max = float(s.max())
            keep  = s > eps_zero * max(s_max, 1.0)
            if keep.any():
                # Sign flip on u to reconcile theory's +r convention with this
                # class's -r storage convention; see class docstring.
                instance.U_     = -U[:, keep].copy()
                instance.Vt_    =  Vt[keep, :].copy()
                instance.alpha_ = (s[keep] / radius).astype(float)

        # Residual weight goes on the zero atom.
        residual = 1.0 - float(instance.alpha_.sum())
        if residual < -eps_zero:
            # Should not happen given the feasibility check above.
            raise RuntimeError(
                f"Residual weight is negative ({residual:g}); "
                "seed-time sanity check failed."
            )
        instance.alpha_zero_ = residual if residual > eps_zero else 0.0
        #instance.alpha_zero_ = max(residual, 0.0)

        _logger.debug(
            f"ActiveSet seeded: n_atoms={instance.n_atoms}, "
            f"alpha_zero={instance.alpha_zero_:.4g}, "
            f"||X0||_*={nuc:g}, radius={radius:g}."
        )
        return instance


    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def radius(self):
        return self.radius_


    @property
    def n_atoms(self):
        return int(self.alpha_.shape[0])


    @property
    def has_zero_atom(self):
        return self.alpha_zero_ > 0.0


    @property
    def total_weight(self):
        """
        Sum of all weights (including zero atom).
        Equals 1 by construction.
        """
        return float(self.alpha_.sum()) + self.alpha_zero_


    # ------------------------------------------------------------------
    # AFW operations
    # ------------------------------------------------------------------

    def update_fw(self, u, vt, gamma):
        """
        FW step update: scale all weights by (1 - gamma), append new atom
        (u, vt) with weight gamma.

        The new atom is always appended as a fresh column; deduplication is
        delegated to prune(). For typical AFW runs, near-duplicate atoms
        accumulate negligible weight and are pruned automatically; explicit
        merging on this hot path is not worth the complexity.

        Parameters
        ----------
        u : np.ndarray, shape (n,)
            Left singular vector of the new FW vertex; unit-norm.
        vt : np.ndarray, shape (p,)
            Right singular vector (transposed); unit-norm.
        gamma : float
            Step size in [0, 1].
        """
        if not (0.0 <= gamma <= 1.0):
            raise ValueError(f"gamma must be in [0, 1]; got {gamma}.")

        if gamma == 0.0:
            return

        # Scale all existing weights.
        self.alpha_      = self.alpha_ * (1.0 - gamma)
        self.alpha_zero_ = self.alpha_zero_ * (1.0 - gamma)

        # Drop atoms whose post-scale weight is zero.
        # Only relevant when gamma = 1 (everything scales to 0).
        self.prune(eps_rel = 0.0)

        # Append new atom. Reshape for column / row stacking.
        u_col  = np.asarray(u, dtype = float).reshape(-1, 1)
        vt_row = np.asarray(vt, dtype = float).reshape(1, -1)

        new_shape = (u_col.shape[0], vt_row.shape[1])

        if self.shape_ is None:
            self.shape_ = new_shape
        elif self.shape_ != new_shape:
            raise ValueError(
                f"Atom shape {new_shape} incompatible with active-set shape "
                f"{self.shape_}."
            )

        if self.n_atoms == 0:
            self.U_  = u_col
            self.Vt_ = vt_row
        else:
            self.U_  = np.concatenate([self.U_,  u_col],  axis = 1)
            self.Vt_ = np.concatenate([self.Vt_, vt_row], axis = 0)
        self.alpha_ = np.concatenate([self.alpha_, [gamma]])
        return


    def update_away(self, k, gamma, eps=1e-9):
        """
        Away step update: scale all weights by (1 + gamma), subtract gamma
        from alpha_[k]. If gamma equals gamma_max(k), atom k is dropped.

        The user gives gamma (= gamma_t). We have gmax (= gamma_max[k]). 
        Behavior at the boundary:
        
          1. gamma == gmax
          residual = 0 exactly; drop atom, weight conserved.
            
          2. gamma = gmax(1 + eps)
          residual = -eps * alpha_k, possible floating point error from 
          caller. Log warning and proceed: clamp to gmax; same as 1.
          Never observable to external code because `_drop_atom` removes
          the entry before returning.
        
          3. gamma = gmax(1 - eps)
          close to gmax, but not quite. floating point error or does
          the user really wants a gamma close to gmax? Respect caller's
          intent, prune cleans up if needed.
          residual = eps * alpha_k. keep atom, weight conserved.
        
          4. gamma > gmax(1 + eps)
          pathological; caller error, raises ValueError.

        Parameters
        ----------
        k : int
            Index of the away atom in the non-zero atoms array.
        gamma : float
            Step size in [0, gamma_max(k)].
        """
        if not (0 <= k < self.n_atoms):
            raise IndexError(
                f"k = {k} out of range for {self.n_atoms} non-zero atoms."
            )
        if gamma < 0.0:
            raise ValueError(f"gamma must be non-negative; got {gamma}.")

        gmax = self.gamma_max(k)
        if gamma > gmax * (1.0 + eps):
            raise ValueError(
                f"gamma = {gamma:g} exceeds gamma_max[{k}] = {gmax:g}, "
                f"tolerance = {eps:g}."
            )

        # Now gamma <= gmax * (1 + eps)
        if gamma > gmax:
            _logger.warning(
                f"gamma = {gamma:g} exceeds gamma_max[{k}] = {gmax:g}, "
                f"but within tolerance = {eps:g}."
            )
        # brief negative alpha_k if gamma = gmax (1 + eps)
        # don't let that happen.
        gamma = min(gamma, gmax)
        # alpha_k = 0 if gamma = gmax
        # alpha_k > 0 if gamma < gmax
        self.alpha_      = self.alpha_ * (1.0 + gamma)
        self.alpha_zero_ = self.alpha_zero_ * (1.0 + gamma)
        self.alpha_[k]  -= gamma

        # handle the negative weight
        if self.alpha_[k] <= 0.0:
            self._drop_atom(k)

        return


    def _drop_atom(self, k):
        keep = np.ones(self.n_atoms, dtype = bool)
        keep[k] = False
        self.U_     = self.U_[:, keep]
        self.Vt_    = self.Vt_[keep, :]
        self.alpha_ = self.alpha_[keep]
        return


    def gamma_max(self, k):
        """
        gamma_max = alpha_[k] / (1 - alpha_[k]).
        Returns +inf when alpha_[k] = 1 (only one atom carries all weight).
        """
        a = float(self.alpha_[k])
        if a >= 1.0:
            return np.inf
        return a / (1.0 - a)


    def oracle_away(self, G):
        """
        Find the worst active atom: the k that maximizes <G, S_k> over the
        current non-zero active atoms.

        Zero atoms are not considered. S_zero = 0 gives <G, S_zero> = 0,
        which can never exceed a positive maximum among non-zero atoms in
        the regime where AFW makes meaningful progress.

        Parameters
        ----------
        G : np.ndarray, shape (n, p)
            Current gradient.

        Returns
        -------
        k_aw : int
            Index of the worst atom in the non-zero atoms array.
        alpha_aw : float
            Weight of that atom (used to compute gamma_max).
        inner_aw : float
            <G, S_{k_aw}>; used as part of the away duality gap.

        Raises
        ------
        RuntimeError
            If the active set has no non-zero atoms (only the zero atom).
        """
        if self.n_atoms == 0:
            raise RuntimeError(
                "oracle_away called on an active set with no non-zero atoms; "
                "AFW cannot take an away step. Caller must take the FW step."
            )

        GVt    = G @ self.Vt_.T                                # (n, K)
        inner  = -self.radius_ * np.einsum('ij,ij->j', self.U_, GVt)
        k_aw     = int(np.argmax(inner))
        alpha_aw = float(self.alpha_[k_aw])
        inner_aw = float(inner[k_aw])
        return k_aw, alpha_aw, inner_aw


    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def prune(self, eps_rel = 1e-10):
        """
        Remove non-zero atoms with negligible weight.

        Atoms with alpha_[k] <= eps_rel * total_weight are dropped, and their
        weight is transferred to the zero atom to preserve the convex-combination
        invariant sum(alpha) + alpha_zero = total_weight.

        Strict inequality on the keep side: prune(eps_rel=0) drops atoms with
        weight exactly zero, which is what update_fw needs after a gamma=1
        step zeros out all prior atoms.

        Parameters
        ----------
        eps_rel : float, default=1e-10
            Relative threshold at or below which an atom is considered negligible.

        Returns
        -------
        n_dropped : int
            Number of atoms dropped.
        """
        if self.n_atoms == 0:
            return 0

        threshold = eps_rel * self.total_weight
        keep      = self.alpha_ > threshold
        n_dropped = int(self.n_atoms - keep.sum())

        if n_dropped == 0:
            return 0

        # Transfer dropped weight to the zero atom.
        self.alpha_zero_ += float(self.alpha_[~keep].sum())
        self.U_           = self.U_[:, keep]
        self.Vt_          = self.Vt_[keep, :]
        self.alpha_       = self.alpha_[keep]

        _logger.debug(
            f"ActiveSet pruned: dropped {n_dropped} atom(s), "
            f"n_atoms={self.n_atoms}, alpha_zero={self.alpha_zero_:.4g}."
        )
        return n_dropped


    def reconstruct(self):
        """
        Reconstruct the dense iterate X = sum_k alpha_k S_k.

        Cost: O(n * p * K). Mainly for diagnostics, invariant tests, and
        warm-start handoff. The AFW iteration loop should *not* call this
        on the hot path; gradient evaluation works directly with the
        dense iterate maintained by the solver.

        Returns
        -------
        X : np.ndarray, shape (n, p)
            Dense reconstruction. The zero atom contributes nothing.
        """
        if self.n_atoms == 0:
            # In practice the active set is never reconstructed in this state
            # (every problem starts with at least the zero atom and
            # add_fw atoms before reconstruct is meaningful).
            if self.shape_ is None:
                # caller built an active set manually, no shape was given.
                raise RuntimeError(
                    "reconstruct() requires a valid shape of X; None provided."
                )
            return np.zeros(self.shape_)
        return -self.radius_ * (self.U_ * self.alpha_) @ self.Vt_
