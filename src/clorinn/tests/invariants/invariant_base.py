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

PGDInvariantBase
    PGD invariants: nuclear norm feasibility, objective non-increasing.
    No duality gap or step size checks (PGD uses a fixed step eta=1/Lf
    and does not compute a duality gap).

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
