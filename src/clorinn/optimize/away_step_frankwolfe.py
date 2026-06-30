"""
Away-step Frank-Wolfe (AFW) solver.

Inherits from FrankWolfe and overrides only what is genuinely different:
  * iteration-state initialization adds an ActiveSet seeded from X0
  * the per-step X-block update method (_fw_X_block_update)
    choose between the FW direction and the away direction
  * _init_iter_history / _append_iter_history attach AFW-specific
    history fields (step_kind, n_atoms_active, step_gap, away_gap)

Everything else (oracles, dg-retry, step-size line search, stopping logic,
fit() loop body, sparse M-block update) is reused unchanged from FrankWolfe.

"""
# Author: Saikat Banerjee
# License: BSD 3 clause


import numpy as np

from .frankwolfe import FrankWolfe
from .active_set import ActiveSet
from .state import StepInfo

class AwayStepFrankWolfe(FrankWolfe):
    """
    Away-step Frank-Wolfe (AFW) for nuclear-norm constrained problems.

    AFW-specific behavior:
        * IterState is seeded with an ActiveSet from X0 (or zero).
        * Per-iteration step picks max-gap direction (FW vs away).
        * Active set is pruned every cfg_.prune_every iterations.
        * History gains step_kind, n_atoms_active, step_gap, away_gap.
    """

    # ------------------------------------------------------------------
    # IterState hook: seed active set from X0
    # ------------------------------------------------------------------

    def _init_iter_state(self, X0):
        """
        Same as FrankWolfe._init_iter_state but additionally seeds the
        active set from X0.
        """
        iter_state = super()._init_iter_state(X0)
        iter_state.active_set = ActiveSet.from_svd(
            iter_state.X, radius = self.obj_.radius_,
        )
        return iter_state

    # ------------------------------------------------------------------
    # History hooks
    # ------------------------------------------------------------------
 
    def _init_iter_history(self, iter_state):
        '''
        Build the History with AFW-specific fields included from the start.
        '''
        history = super()._init_iter_history(iter_state)
        history.step_kind      = ['init']
        history.n_atoms_active = [iter_state.active_set.n_atoms]
        history.step_gap       = [np.inf]
        history.away_gap       = [np.inf]
        return history


    def _append_iter_history(self, history, iter_state, info, cpu_time):
        '''
        Append base-class fields plus AFW-specific fields.
        AFW fields are read from StepInfo, populated by
        _fw_X_block_update.
        '''
        super()._append_iter_history(history, iter_state, info, cpu_time)
        history.step_kind.append(info.step_kind)
        history.n_atoms_active.append(iter_state.active_set.n_atoms)
        history.step_gap.append(info.step_gap)
        history.away_gap.append(info.away_gap)
        return




    # ------------------------------------------------------------------
    # Per-step X-block override
    # ------------------------------------------------------------------

    def _fw_X_block_update(self, G, iter_state):
        """
        AFW X-block update.

        history.duality_gap records dg_fw, the convergence certificate.
        history.step_gap records the gap of the direction actually taken.
        history.away_gap records dg_aw, the away step gap.
        """
        active = iter_state.active_set
 
        # FW direction (with negative-dg retry on power iteration)
        S_fw, D_fw, dg_fw = self._try_positive_dg(G, iter_state = iter_state)

        take_fw = False
        k_aw = None
        dg_aw = 0.0

        if active.n_atoms == 0:
            # Empty active set -> force FW (no away atom available).
            take_fw = True
        else:
            # Choose direction
            k_aw, alpha_aw, inner_aw = active.oracle_away(G)
            inner_X = float(np.sum(iter_state.X * G))
            dg_aw   = inner_aw - inner_X
            take_fw = (k_aw is None) or (dg_fw >= dg_aw)

        if take_fw:
            step_size, step_kind = self._take_fw_step(iter_state, S_fw, D_fw, dg_fw)
            step_gap = dg_fw
        else:
            step_size, step_kind = self._take_away_step(iter_state, k_aw, dg_aw)
            step_gap = dg_aw
            
        self._maybe_prune(iter_state)

        return StepInfo(
            dg        = dg_fw,
            step_size = step_size,
            step_kind = step_kind,
            step_gap  = step_gap,
            away_gap  = dg_aw,
        )


    # ------------------------------------------------------------------
    # Step helpers
    # ------------------------------------------------------------------
 
    def _take_fw_step(self, iter_state, S_fw, D_fw, dg_fw):
        '''
        Take a Frank-Wolfe step and update the active set.
        Returns (step_size, step_kind), where step_kind is always 'fw'.
        '''
        step_size = self._compute_step_size(
            dg_fw, D_fw, iter_state, gamma_max = 1.0,
        )
        iter_state.X = iter_state.X - step_size * D_fw
 
        # Update active set with the FW vertex's singular factors.
        u  = iter_state.svd_u.ravel()
        vt = iter_state.svd_vt.ravel()
        iter_state.active_set.update_fw(u, vt, gamma = step_size)
        return step_size, 'fw'


    def _take_away_step(self, iter_state, k_aw, dg_aw):
        """
        Take an away step and update the active set.

        Returns (step_size, step_kind). step_kind is 'away' or 'drop'
        depending on whether the atom survived the update.
        """
        active = iter_state.active_set
        gmax   = active.gamma_max(k_aw)

        # Materialize the away vertex.
        S_aw = - self.obj_.radius_ * (
            active.U_[:, k_aw : k_aw + 1] @ active.Vt_[k_aw : k_aw + 1, :]
        )
        D_aw = S_aw - iter_state.X

        # Line search clamped to gmax. Belt-and-braces: clamp again before
        # calling update_away to absorb FP drift, so the snap-warn path
        # inside update_away does not fire on routine iterations.
        step_size = self._compute_step_size(
            dg_aw, D_aw, iter_state, gamma_max = gmax,
        )
        if np.isfinite(gmax):
            step_size = min(step_size, gmax)

        n_atoms_before = active.n_atoms
        iter_state.X = iter_state.X - step_size * D_aw
        active.update_away(k_aw, gamma = step_size)
        step_kind = 'drop' if active.n_atoms < n_atoms_before else 'away'
        return step_size, step_kind


    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def _maybe_prune(self, iter_state):
        """Prune the active set every prune_every iterations."""
        every = self.cfg_.prune_every
        if every <= 0:
            return
        if iter_state.istep % every != 0:
            return
        n_dropped = iter_state.active_set.prune(
            eps_rel = self.cfg_.prune_eps_rel,
        )
        if n_dropped > 0:
            self.logger_.debug(
                f"Iteration {iter_state.istep}: pruned {n_dropped} low-weight atoms."
            )
        return
