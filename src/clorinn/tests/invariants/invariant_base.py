"""
invariant_base.py
-----------------
Base classes defining solver invariant checks.

Every invariant is a single test method. Concrete subclasses supply
the solver result via setUpClass; they inherit all applicable checks
automatically.

Hierarchy
---------
FWInvariantBase
    Core FW invariants: nuclear norm feasibility, objective non-increasing,
    duality gap nonnegative, step size bounds.

FWSparseInvariantBase(FWInvariantBase)
    Adds l1 norm feasibility for NNM-Sparse objectives, duality gap of
    sparse block close to zero.

AFWInvariantBase(FWInvariantBase)
    Inherits FW invariants and adds AFW-specific checks: history-field
    presence and length consistency, valid step_kind values, n_atoms
    transitions consistent with step_kind, away_gap nonnegative,
    step_gap matches direction taken.
 
AFWSparseInvariantBase(AFWInvariantBase, FWSparseInvariantBase)
    AFW-Sparse invariants: combines AFW-specific checks with l1
    feasibility and sparse-block-zero-gap checks.

PGDInvariantBase
    PGD invariants: nuclear norm feasibility, objective non-increasing.
    No duality gap or step size checks (PGD uses a fixed step eta=1/Lf
    and does not compute a duality gap).
 
PGDSparseInvariantBase(PGDInvariantBase)
    Adds l1 norm feasibility for NNM-Sparse objectives.

Adding a new invariant
----------------------
Add a test method to the appropriate base class. All concrete subclasses
that inherit from it pick up the new check automatically. Use skipTest()
with a hasattr guard for invariants that apply only to a subset of
configurations (e.g. weights, missing data).

Running
-------
    python -m unittest clorinn.tests.invariants.test_fw_nnm -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest
import numpy as np


# Numerical tolerance used across all invariant checks.
# Chosen to be tight enough to catch genuine violations but loose enough
# to absorb floating-point accumulation over thousands of FW iterations.
_FEAS_TOL  = 1e-9   # feasibility: ||X||_* <= radius + _FEAS_TOL
_MONO_TOL  = 1e-9   # monotonicity: f(t+1) - f(t) <= _MONO_TOL
_DG_TOL    = 1e-9   # duality gap: g_t >= -_DG_TOL
_STEP_TOL  = 1e-9   # step size: 0 <= gamma_t <= 1 + _STEP_TOL


# ---------------------------------------------------------------------------
# Frank-Wolfe base
# ---------------------------------------------------------------------------

class FWInvariantBase(unittest.TestCase):
    """
    Invariant checks shared by all Frank-Wolfe solver / objective pairs.

    Subclasses must set in setUpClass:
        cls.result  : FitResult
        cls.radius  : float   (nuclear norm constraint radius)
    """

    result = None
    radius = None

    @classmethod
    def setUpClass(cls):
        if cls.result is None:
            raise unittest.SkipTest("base class")

    # ------------------------------------------------------------------
    # Nuclear norm feasibility
    # ------------------------------------------------------------------

    def test_nuclear_norm_feasibility(self):
        """||X||_* <= radius at the converged iterate."""
        nuc = np.linalg.norm(self.result.X, 'nuc')
        self.assertLessEqual(
            nuc, self.radius + _FEAS_TOL,
            msg=f"||X||_* = {nuc:.6g} > radius = {self.radius:.6g}",
        )

    # ------------------------------------------------------------------
    # Objective non-increasing
    # ------------------------------------------------------------------

    def test_objective_nonincreasing(self):
        """f(X_t) is non-increasing across all iterations."""
        loss = np.array(self.result.history.loss)
        diffs = np.diff(loss)
        violations = np.where(diffs > _MONO_TOL)[0]
        msg = (
            f"Objective increased at {len(violations)} step(s). "
            f"Max increase: {diffs[violations].max():.3g} at steps {violations}."
        ) if len(violations) > 0 else ""
        self.assertEqual(len(violations), 0, msg=msg)

    # ------------------------------------------------------------------
    # Duality gap nonnegative
    # ------------------------------------------------------------------

    def test_duality_gap_nonnegative(self):
        """Duality gap g_t >= 0 at every iteration (skip step-0 inf)."""
        dg = np.array(self.result.history.duality_gap[1:])
        violations = np.where(dg < -_DG_TOL)[0]
        msg = (
            f"Negative duality gap at {len(violations)} step(s). "
            f"Min value: {dg[violations].min():.3g} at steps {violations + 1}."
        ) if len(violations) > 0 else ""
        self.assertEqual(len(violations), 0, msg=msg)

    # ------------------------------------------------------------------
    # Step size bounds
    # ------------------------------------------------------------------

    def test_step_size_bounds(self):
        """Step size gamma_t in [0, 1] at every iteration (skip step-0)."""
        ss = np.array(self.result.history.step_size[1:])
        below = np.where(ss < 0.0)[0]
        above = np.where(ss > 1.0 + _STEP_TOL)[0]
        self.assertEqual(
            len(below), 0,
            msg=f"Step size < 0 at steps: {below + 1}.",
        )
        msg_above = (
            f"Step size > 1 at {len(above)} step(s). "
            f"Max value: {ss[above].max():.3g} at steps {above + 1}."
        ) if len(above) > 0 else ""
        self.assertEqual(len(above), 0, msg=msg_above)


# ---------------------------------------------------------------------------
# Frank-Wolfe sparse extension
# ---------------------------------------------------------------------------

class FWSparseInvariantBase(FWInvariantBase):
    """
    Extends FWInvariantBase with the l1 feasibility check for NNM-Sparse.

    Subclasses must additionally set in setUpClass:
        cls.l1_threshold : float   (scaled l1 constraint radius)
    """

    l1_threshold = None

    @classmethod
    def setUpClass(cls):
        if cls.result is None:
            raise unittest.SkipTest("base class")

    # ------------------------------------------------------------------
    # l1 norm feasibility
    # ------------------------------------------------------------------
    def test_l1_norm_feasibility(self):
        """||M||_1 <= l1_threshold at the converged iterate."""
        #l1 = np.linalg.norm(self.result.M, 1)
        l1 = np.sum(np.abs(self.result.M))
        self.assertLessEqual(
            l1, self.l1_threshold + _FEAS_TOL,
            msg=f"||M||_1 = {l1:.6g} > l1_threshold = {self.l1_threshold:.6g}",
        )

    # ------------------------------------------------------------------
    # duality gap of M = 0
    # ------------------------------------------------------------------

    def test_sparse_block_at_projection(self):
        """gM should be ~0 at every iteration."""
        if self.result.history.duality_gap_sparse is None:
            self.skipTest("not a sparse run")
        gM = np.array(self.result.history.duality_gap_sparse)
        # Skip iter 0 (gM is meaningful only after first projection)
        max_gM = np.abs(gM[1:]).max() if len(gM) > 1 else 0.0
        self.assertLess(
            max_gM, _DG_TOL,
            msg=f"gM exceeded tolerance ({max_gM:.3g}); "
                "projection invariant likely violated."
        )

    # ------------------------------------------------------------------
    # l1 ball is active for this test problem
    # ------------------------------------------------------------------

    def test_l1_ball_is_active_in_test_problem(self):
        """Pre-condition for test_sparse_block_at_projection: the test
        fixture must exercise an active l1 ball, otherwise gM = 0 holds
        trivially because G ≈ 0."""
        m = self.result
        self.assertGreaterEqual(
            m.metrics['l1_norm'],
            m.metrics['l1_threshold'] - 1e-6,
            msg=f"l1 ball is inactive (||M||_1 = {m.metrics['l1_norm']:.4g} "
                f"< l1_threshold = {m.metrics['l1_threshold']:.4g}). "
                "gM = 0 invariant holds vacuously. Tighten l1_threshold."
        )

# ---------------------------------------------------------------------------
# Away-step Frank-Wolfe
# ---------------------------------------------------------------------------
 
class AFWInvariantBase(FWInvariantBase):
    """
    AFW invariants. Inherits all FW invariants (nuclear-norm feasibility,
    objective non-increasing, dg >= 0, step-size bounds) and adds checks
    on the active set and AFW-specific history fields.
 
    Subclasses must set in setUpClass:
        cls.result  : FitResult
        cls.radius  : float
    """
 
    @classmethod
    def setUpClass(cls):
        if cls.result is None:
            raise unittest.SkipTest("base class")


    # ------------------------------------------------------------------
    # Away step size can be greater than 1.0
    # ------------------------------------------------------------------
    def test_step_size_bounds(self):
        """
        FW step size in [0,1]. 
        Away/drop steps must satisfy gamma >= 0, hence excluded.
        """
        h = self.result.history
        ss = np.array(h.step_size[1:])
        step_kinds = h.step_kind[1:]

        below = np.where(ss < 0.0)[0]
        self.assertEqual(
            len(below), 0,
            msg=f"Step size < 0 at steps: {below + 1}.",
        )

        nonfinite = np.where(~np.isfinite(ss))[0]
        self.assertEqual(
            len(nonfinite), 0,
            msg=f"Non-finite step size at steps: {nonfinite + 1}.",
        )

        above = ss > 1.0 + _STEP_TOL
        kind_fw = step_kinds == "fw"
        above_fw = np.where(above & kind_fw)[0]
        msg_above = (
            f"Step size > 1 at {len(above_fw)} step(s). "
            f"Max value: {ss[above_fw].max():.3g} at steps {above_fw + 1}."
        ) if len(above_fw) > 0 else ""
        self.assertEqual(len(above_fw), 0, msg=msg_above)
 
    # ------------------------------------------------------------------
    # AFW history field presence + length consistency
    # ------------------------------------------------------------------
 
    def test_afw_history_fields_populated(self):
        """AFW-specific history fields are non-None lists for an AFW run."""
        h = self.result.history
        for name in ('step_kind', 'n_atoms_active', 'step_gap', 'away_gap'):
            self.assertIsNotNone(
                getattr(h, name),
                msg=f"history.{name} is None on an AFW result",
            )
 
    def test_afw_history_lengths_consistent(self):
        """All AFW history lists match the length of duality_gap."""
        h = self.result.history
        n = len(h.duality_gap)
        for name in ('step_kind', 'n_atoms_active', 'step_gap', 'away_gap'):
            self.assertEqual(
                len(getattr(h, name)), n,
                msg=f"history.{name} has length {len(getattr(h, name))}, "
                    f"expected {n}",
            )
 
    # ------------------------------------------------------------------
    # step_kind values
    # ------------------------------------------------------------------
 
    def test_step_kind_values_valid(self):
        """Each step_kind entry is one of the allowed labels."""
        valid = {'init', 'fw', 'away', 'drop'}
        kinds = self.result.history.step_kind
        bad   = [(i, k) for i, k in enumerate(kinds) if k not in valid]
        self.assertEqual(
            bad, [],
            msg=f"step_kind has invalid entries: {bad[:5]}",
        )
 
    def test_step_kind_init_only_at_iter_zero(self):
        """The 'init' label appears only at index 0."""
        kinds = self.result.history.step_kind
        if not kinds:
            return
        self.assertEqual(kinds[0], 'init', msg="step_kind[0] != 'init'")
        bad = [i for i, k in enumerate(kinds[1:], start=1) if k == 'init']
        self.assertEqual(bad, [], msg=f"'init' appears past index 0: {bad}")
 
    # ------------------------------------------------------------------
    # n_atoms_active transitions
    # ------------------------------------------------------------------
 
    def test_n_atoms_active_nonnegative(self):
        """Active-set size is >= 0 at every iteration."""
        n_atoms = np.array(self.result.history.n_atoms_active)
        self.assertTrue(
            np.all(n_atoms >= 0),
            msg=f"n_atoms_active has negative values: "
                f"{n_atoms[n_atoms < 0]}",
        )
 
    def test_n_atoms_change_consistent_with_step_kind(self):
        """
        Per-iteration n_atoms transitions are consistent with step_kind:
            'fw'    : n_atoms increases by 1 (normal case),
                      OR resets to 1 (gamma=1 case: all prior atoms had
                      weight scaled to 0 and were pruned).
            'away'  : n_atoms unchanged.
            'drop'  : n_atoms decreases by 1.
        Skip iter 0 (init).
        """
        kinds      = self.result.history.step_kind
        n_atoms    = self.result.history.n_atoms_active
        step_sizes = self.result.history.step_size
        bad        = []
        for t in range(1, len(kinds)):
            delta = n_atoms[t] - n_atoms[t - 1]
            k     = kinds[t]
            if k == 'fw':
                # Three valid cases:
                #   normal:       n_atoms += 1   (gamma in (0, 1))
                #   wipeout:      n_atoms = 1    (gamma == 1, prior atoms zeroed and pruned)
                #   degenerate:   n_atoms unchanged  (gamma == 0, update_fw is a no-op)
                gamma_one = abs(step_sizes[t] - 1.0) < 1e-12
                gamma_zero = step_sizes[t] == 0.0
                ok = (
                    (delta == 1) # normal
                    or (gamma_one  and n_atoms[t] == 1) # wipeout
                    or (gamma_zero and delta == 0) # degenerate, no-op
                )
            elif k == 'away':
                ok = (delta == 0)
            elif k == 'drop':
                ok = (delta == -1)
            else:
                ok = False
            if not ok:
                bad.append((t, k, n_atoms[t - 1], n_atoms[t], step_sizes[t]))
        self.assertEqual(
            bad, [],
            msg=f"step_kind / n_atoms inconsistency at {len(bad)} step(s); "
                f"first 5: {bad[:5]}",
        )
 
    # ------------------------------------------------------------------
    # away_gap nonnegative
    # ------------------------------------------------------------------
 
    def test_away_gap_nonnegative(self):
        """Away gap g_aw_t >= 0 at every iteration (skip step-0 inf)."""
        gw = np.array(self.result.history.away_gap[1:])
        violations = np.where(gw < -_DG_TOL)[0]
        msg = (
            f"Negative away_gap at {len(violations)} step(s). "
            f"Min: {gw[violations].min():.3g} at steps {violations + 1}."
        ) if len(violations) > 0 else ""
        self.assertEqual(len(violations), 0, msg=msg)
 
    # ------------------------------------------------------------------
    # step_gap consistent with chosen direction
    # ------------------------------------------------------------------
 
    def test_step_gap_matches_direction_taken(self):
        """
        step_gap[t] equals duality_gap[t] (= dg_fw) when an FW step was
        taken, and equals away_gap[t] (= dg_aw) when an away or drop step
        was taken. Holds within numerical tolerance.
        """
        h     = self.result.history
        kinds = h.step_kind
        sg    = np.array(h.step_gap)
        dg_fw = np.array(h.duality_gap)
        dg_aw = np.array(h.away_gap)
        bad   = []
        for t in range(1, len(kinds)):
            k = kinds[t]
            expected = dg_fw[t] if k == 'fw' else dg_aw[t]
            if not np.isclose(sg[t], expected, atol=_DG_TOL, rtol=1e-9):
                bad.append((t, k, sg[t], expected))
        self.assertEqual(
            bad, [],
            msg=f"step_gap inconsistency at {len(bad)} step(s); "
                f"first 5: {bad[:5]}",
        )


class AFWSparseInvariantBase(AFWInvariantBase, FWSparseInvariantBase):
    """
    AFW invariants for the NNM-Sparse model: combines AFW-specific
    checks with the l1 feasibility / sparse-block-zero-gap checks
    from FW-sparse.

    Subclasses must additionally set in setUpClass:
        cls.l1_threshold : float
    """

    @classmethod
    def setUpClass(cls):
        if cls.result is None:
            raise unittest.SkipTest("base class")


# ---------------------------------------------------------------------------
# PGD base
# ---------------------------------------------------------------------------

class PGDInvariantBase(unittest.TestCase):
    """
    Invariant checks for the PGD solver.

    PGD does not compute a duality gap and uses a fixed step size
    eta = 1/Lf, so those checks are absent here.

    Subclasses must set in setUpClass:
        cls.result  : FitResult
        cls.radius  : float   (nuclear norm constraint radius)
    """

    result = None
    radius = None

    @classmethod
    def setUpClass(cls):
        if cls.result is None:
            raise unittest.SkipTest("base class")

    # ------------------------------------------------------------------
    # Nuclear norm feasibility
    # ------------------------------------------------------------------

    def test_nuclear_norm_feasibility(self):
        """||X||_* <= radius at every PGD iterate."""
        nuc = np.linalg.norm(self.result.X, 'nuc')
        self.assertLessEqual(
            nuc, self.radius + _FEAS_TOL,
            msg=f"||X||_* = {nuc:.6g} > radius = {self.radius:.6g}",
        )

    # ------------------------------------------------------------------
    # Objective non-increasing
    # ------------------------------------------------------------------

    def test_objective_nonincreasing(self):
        """f(X_t) is non-increasing across all PGD iterations."""
        loss = np.array(self.result.history.loss)
        diffs = np.diff(loss)
        violations = np.where(diffs > _MONO_TOL)[0]
        msg = (
            f"Objective increased at {len(violations)} step(s). "
            f"Max increase: {diffs[violations].max():.3g} at steps {violations}."
        ) if len(violations) > 0 else ""
        self.assertEqual(len(violations), 0, msg=msg)


# ---------------------------------------------------------------------------
# PGD sparse extension
# ---------------------------------------------------------------------------

class PGDSparseInvariantBase(PGDInvariantBase):
    """
    Extends PGDInvariantBase with the l1 feasibility check for NNM-Sparse.

    Subclasses must additionally set in setUpClass:
        cls.l1_threshold : float   (scaled l1 constraint radius)
    """

    l1_threshold = None

    @classmethod
    def setUpClass(cls):
        if cls.result is None:
            raise unittest.SkipTest("base class")

    def test_l1_norm_feasibility(self):
        """||M||_1 <= l1_threshold at the converged iterate."""
        l1 = np.sum(np.abs(self.result.M))
        self.assertLessEqual(
            l1, self.l1_threshold + _FEAS_TOL,
            msg=f"||M||_1 = {l1:.6g} > l1_threshold = {self.l1_threshold:.6g}",
        )
