"""
test_active_set.py
------------------
Unit tests for the ActiveSet helper class used by the AFW solver.

Test classes
------------
TestActiveSetConstruct       Three §6.2 seed cases plus error paths.
TestActiveSetReconstruct     Round-trip from seed.
TestActiveSetUpdateFW        FW update semantics.
TestActiveSetUpdateAway      Away update semantics.
TestActiveSetOracleAway      Worst-atom selection.
TestActiveSetGammaMax        Step-size upper bound.
TestActiveSetPrune           Atom removal by relative threshold.
TestActiveSetInvariants      Cross-cutting invariants.
TestActiveSetLongRunSequence Random-sequence stability.

Running
-------
    python -m unittest clorinn.tests.unit.test_active_set -v
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import unittest
import numpy as np

from clorinn.optimize.active_set import ActiveSet


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(rng, n):
    """Random unit-norm vector of length n."""
    v = rng.standard_normal(n)
    v /= np.linalg.norm(v)
    return v


def _seed_X0(rng, n, p, radius, *, scale = 0.5):
    """
    Construct an iterate with ||X0||_* = scale * radius.
    scale = 1.0 -> boundary, scale < 1 -> interior, scale = 0 -> zero.
    """
    if scale == 0.0:
        return np.zeros((n, p))
    X = rng.standard_normal((n, p))
    X *= (scale * radius) / np.linalg.norm(X, 'nuc')
    return X


# Tolerances. These are the only tunable knobs; if anything fails just below
# threshold, lift the relevant constant rather than relaxing per-test.
_RECON_TOL    = 1e-12       # reconstruct() round-trip
_WEIGHT_TOL   = 1e-12       # total_weight invariant
_FEAS_TOL     = 1e-9        # ||X||_* <= radius after operations
_LONG_TOL     = 1e-10       # invariants over long random sequences


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------

class TestActiveSetConstruct(unittest.TestCase):
    """Construction from SVD factors and from raw X0; error paths."""

    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n, self.p, self.r = 5, 20, 10.0


    def test_boundary_seed_no_zero_atom(self):
        """Boundary case seeds with no zero atom."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 1.0)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        self.assertFalse(A.has_zero_atom,
            msg = f"alpha_zero = {A.alpha_zero_:.3e} (should be 0 at boundary)")
        self.assertGreater(A.n_atoms, 0)


    def test_interior_seed_has_zero_atom(self):
        """Interior case seeds with zero atom residual."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 0.5)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        self.assertTrue(A.has_zero_atom)
        self.assertAlmostEqual(A.alpha_zero_, 0.5, places = 9)


    def test_zero_seed_single_zero_atom(self):
        """Zero matrix gives single zero atom."""
        X0 = np.zeros((self.n, self.p))
        A  = ActiveSet.from_svd(X0, radius = self.r)
        self.assertEqual(A.n_atoms, 0)
        self.assertEqual(A.alpha_zero_, 1.0)


    def test_infeasible_seed_raises(self):
        """Infeasible X0 raises ValueError."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 5.0)  # ||X0||_* = 5r
        with self.assertRaises(ValueError):
            ActiveSet.from_svd(X0, radius = self.r)


    def test_empty_svd_factors_construct_cleanly(self):
        """Empty SVD factors construct cleanly."""
        U  = np.empty((self.n, 0))
        s  = np.empty(0)
        Vt = np.empty((0, self.p))
        A  = ActiveSet.from_svd_factors(U, s, Vt, radius = self.r)
        self.assertEqual(A.n_atoms, 0)
        self.assertEqual(A.alpha_zero_, 1.0)
        self.assertEqual(A.shape_, (self.n, self.p))


    def test_nonpositive_radius_raises(self):
        """Non-positive radius raises ValueError."""
        for bad in [0.0, -1.0, -1e-12]:
            with self.assertRaises(ValueError, msg = f"radius={bad}"):
                ActiveSet(radius = bad)


# ---------------------------------------------------------------------------
# Reconstruction
# ---------------------------------------------------------------------------

class TestActiveSetReconstruct(unittest.TestCase):
    """Reconstruct round-trip from seed across all three §6.2 cases."""

    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n, self.p, self.r = 5, 20, 10.0


    def test_boundary_reconstructs_X0(self):
        """Boundary seed reconstructs X0 exactly."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 1.0)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        self.assertLess(np.linalg.norm(A.reconstruct() - X0), _RECON_TOL)


    def test_interior_reconstructs_X0(self):
        """Interior seed reconstructs X0 exactly."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 0.4)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        self.assertLess(np.linalg.norm(A.reconstruct() - X0), _RECON_TOL)


    def test_zero_reconstructs_zero(self):
        """Zero seed reconstructs zero matrix."""
        X0 = np.zeros((self.n, self.p))
        A  = ActiveSet.from_svd(X0, radius = self.r)
        np.testing.assert_array_equal(A.reconstruct(), X0)


    def test_no_shape_reconstruct_raises(self):
        """Empty set with no shape raises RuntimeError on reconstruct."""
        A = ActiveSet(radius = self.r)
        self.assertEqual(A.n_atoms, 0)
        self.assertIsNone(A.shape_)
        with self.assertRaises(RuntimeError):
            A.reconstruct()


# ---------------------------------------------------------------------------
# update_fw
# ---------------------------------------------------------------------------

class TestActiveSetUpdateFW(unittest.TestCase):
    """Semantics of the FW step update."""

    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.n, self.p, self.r = 5, 20, 10.0
        self.u  = _unit(self.rng, self.n)
        self.vt = _unit(self.rng, self.p)


    def test_appends_atom_with_weight_gamma(self):
        """Appends new atom with weight gamma."""
        A = ActiveSet.from_svd(np.zeros((self.n, self.p)), radius = self.r)
        A.update_fw(self.u, self.vt, gamma = 0.3)
        self.assertEqual(A.n_atoms, 1)
        self.assertAlmostEqual(float(A.alpha_[0]), 0.3, places = 12)


    def test_scales_prior_weights_by_one_minus_gamma(self):
        """Scales prior weights by (1 - gamma)."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 0.5)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        prior_alpha = A.alpha_.copy()
        prior_zero  = A.alpha_zero_
        gamma = 0.4
        A.update_fw(self.u, self.vt, gamma = gamma)
        np.testing.assert_allclose(
            A.alpha_[:-1], prior_alpha * (1 - gamma), atol = 1e-12)
        self.assertAlmostEqual(A.alpha_zero_, prior_zero * (1 - gamma), places = 12)


    def test_gamma_zero_is_noop(self):
        """Gamma equals zero is a no-op."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 0.5)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        before = (A.n_atoms, A.alpha_.copy(), A.alpha_zero_)
        A.update_fw(self.u, self.vt, gamma = 0.0)
        self.assertEqual(A.n_atoms, before[0])
        np.testing.assert_array_equal(A.alpha_, before[1])
        self.assertEqual(A.alpha_zero_, before[2])


    def test_gamma_one_wipes_prior_atoms(self):
        """Gamma equals one wipes prior atoms."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 0.6)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        A.update_fw(self.u, self.vt, gamma = 1.0)
        self.assertEqual(A.n_atoms, 1)
        self.assertEqual(A.alpha_zero_, 0.0)
        self.assertAlmostEqual(float(A.alpha_[0]), 1.0, places = 12)


    def test_shape_inferred_from_first_atom(self):
        """Shape inferred from first atom when constructor shape was None."""
        A = ActiveSet(radius = self.r)
        self.assertIsNone(A.shape_)
        A.update_fw(self.u, self.vt, gamma = 0.5)
        self.assertEqual(A.shape_, (self.n, self.p))


    def test_shape_mismatch_raises(self):
        """Shape mismatch on subsequent atom raises ValueError."""
        A = ActiveSet(radius = self.r)
        A.update_fw(self.u, self.vt, gamma = 0.5)
        bad_u  = _unit(self.rng, self.n + 1)   # wrong row count
        bad_vt = _unit(self.rng, self.p)
        with self.assertRaises(ValueError):
            A.update_fw(bad_u, bad_vt, gamma = 0.3)


    def test_out_of_range_gamma_raises(self):
        """Out-of-range gamma raises ValueError."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        for bad in [-0.1, 1.5, np.nan]:
            with self.assertRaises(ValueError, msg = f"gamma={bad}"):
                A.update_fw(self.u, self.vt, gamma = bad)


    def test_stored_u_vt_remain_unit_norm(self):
        """Stored u and vt remain unit-norm after update_fw."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(self.u, self.vt, gamma = 0.5)
        self.assertAlmostEqual(np.linalg.norm(A.U_[:, 0]),  1.0, places = 12)
        self.assertAlmostEqual(np.linalg.norm(A.Vt_[0, :]), 1.0, places = 12)


# ---------------------------------------------------------------------------
# update_away
# ---------------------------------------------------------------------------

class TestActiveSetUpdateAway(unittest.TestCase):
    """Semantics of the away step update."""

    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.n, self.p, self.r = 5, 20, 10.0


    def _two_atom_set(self, alpha_first):
        """
        Build an active set with two non-zero atoms, weights
        (alpha_first, 1 - alpha_first).
        """
        u1 = _unit(self.rng, self.n);  v1 = _unit(self.rng, self.p)
        u2 = _unit(self.rng, self.n);  v2 = _unit(self.rng, self.p)
        A  = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(u1, v1, gamma = 1.0)
        A.update_fw(u2, v2, gamma = 1.0 - alpha_first)
        return A


    def test_subtracts_gamma_from_selected_atom(self):
        """Subtracts gamma from selected atom weight."""
        A = self._two_atom_set(alpha_first = 0.3)
        # alpha = [0.3, 0.7]; take gamma = 0.1 < gmax(0)
        A.update_away(0, gamma = 0.1)
        # alpha[0] = 0.3 * 1.1 - 0.1 = 0.23
        # alpha[1] = 0.7 * 1.1       = 0.77
        np.testing.assert_allclose(A.alpha_, [0.23, 0.77], atol = 1e-12)


    def test_gamma_equal_gmax_drops_atom(self):
        """Gamma equal to gmax drops the atom."""
        A = self._two_atom_set(alpha_first = 0.3)
        gmax = A.gamma_max(0)
        A.update_away(0, gamma = gmax)
        self.assertEqual(A.n_atoms, 1)
        self.assertAlmostEqual(A.total_weight, 1.0, places = 12)


    def test_gamma_slightly_above_gmax_logs_warning(self):
        """Gamma slightly above gmax logs warning and clamps."""
        A = self._two_atom_set(alpha_first = 0.3)
        gmax = A.gamma_max(0)
        with self.assertLogs('clorinn.optimize.active_set', level = 'WARNING'):
            A.update_away(0, gamma = gmax * (1 + 1e-11))
        # Atom dropped (clamped to gmax behaves as drop)
        self.assertEqual(A.n_atoms, 1)
        self.assertAlmostEqual(A.total_weight, 1.0, places = 12)


    def test_gamma_well_above_gmax_raises(self):
        """Gamma well above gmax raises ValueError."""
        A = self._two_atom_set(alpha_first = 0.3)
        gmax = A.gamma_max(0)
        with self.assertRaises(ValueError):
            A.update_away(0, gamma = gmax * 2.0)


    def test_out_of_range_k_raises(self):
        """Out-of-range k raises IndexError."""
        A = self._two_atom_set(alpha_first = 0.3)
        for bad_k in [-1, A.n_atoms]:
            with self.assertRaises(IndexError, msg = f"k={bad_k}"):
                A.update_away(bad_k, gamma = 0.1)


    def test_negative_gamma_raises(self):
        """Negative gamma raises ValueError."""
        A = self._two_atom_set(alpha_first = 0.3)
        with self.assertRaises(ValueError):
            A.update_away(0, gamma = -0.01)


    def test_single_atom_gmax_is_infinity(self):
        """Single-atom case returns gmax = +inf."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(_unit(self.rng, self.n), _unit(self.rng, self.p), gamma = 1.0)
        self.assertEqual(A.gamma_max(0), np.inf)


# ---------------------------------------------------------------------------
# oracle_away
# ---------------------------------------------------------------------------

class TestActiveSetOracleAway(unittest.TestCase):
    """Worst-atom selection."""

    def setUp(self):
        self.rng = np.random.default_rng(4)
        self.n, self.p, self.r = 5, 20, 10.0


    def test_returns_atom_with_max_inner(self):
        """Returns atom with maximum inner product."""
        # Build two distinct atoms and a gradient that favours atom 0.
        u1 = np.zeros(self.n); u1[0] = 1.0
        v1 = np.zeros(self.p); v1[0] = 1.0
        u2 = np.zeros(self.n); u2[1] = 1.0
        v2 = np.zeros(self.p); v2[1] = 1.0

        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(u1, v1, gamma = 1.0)
        A.update_fw(u2, v2, gamma = 0.6)

        # update_fw stores (u, vt) as-is. S_k = -r * u_k @ vt_k^T.
        # For basis-vector atoms (u = e_i, vt = e_j),
        #   <G, S_k> = -r * u_k^T G vt_k = -r * G[i, j].
        # Pick G[0, 0] = -1: gives <G, S_0> = +r, the largest among atoms.
        G = np.zeros((self.n, self.p))
        G[0, 0] = -1.0
        k_aw, alpha_aw, inner_aw = A.oracle_away(G)
        self.assertEqual(k_aw, 0)


    def test_returns_three_tuple_of_correct_types(self):
        """Returns three-tuple of (int, float, float)."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(_unit(self.rng, self.n), _unit(self.rng, self.p), gamma = 1.0)
        out = A.oracle_away(self.rng.standard_normal((self.n, self.p)))
        self.assertEqual(len(out), 3)
        self.assertIsInstance(out[0], int)
        self.assertIsInstance(out[1], float)
        self.assertIsInstance(out[2], float)


    def test_returns_negative_inner_when_all_negative(self):
        """
        When all non-zero atoms have <G, S_k> < 0 and there is no zero
        atom, oracle returns the least-bad non-zero atom (its inner is
        still negative). When a zero atom is present, the zero atom
        wins instead — see test_zero_atom_wins_when_all_inner_negative.
        """
        u = _unit(self.rng, self.n);  vt = _unit(self.rng, self.p)
        # Boundary seed (no zero atom) followed by an all-FW-step build
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(u, vt, gamma = 1.0)
        self.assertFalse(A.has_zero_atom)
        # update_fw stores (u, vt) as-is.
        # So S = -r * u @ vt^T and <G, S> = -r * u^T G vt.
        # Pick G = +outer(u, vt), giving u^T G vt = 1 and <G, S> = -r < 0.
        G = np.outer(u, vt)
        k_aw, _, inner_aw = A.oracle_away(G)
        self.assertEqual(k_aw, 0)
        self.assertLess(inner_aw, 0.0)


    def test_empty_active_set_raises(self):
        """Empty active set raises RuntimeError."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        self.assertEqual(A.n_atoms, 0)
        self.assertFalse(A.has_zero_atom)
        with self.assertRaises(RuntimeError):
            A.oracle_away(self.rng.standard_normal((self.n, self.p)))


    def test_zero_atom_wins_when_all_inner_negative(self):
        """
        When the active set has a zero atom AND every non-zero atom has
        <G, S_k> < 0, the zero atom is the away candidate (its inner is
        0, which beats any negative value).
        """
        u = _unit(self.rng, self.n);  vt = _unit(self.rng, self.p)
        # Interior seed: ||X0||_* < radius gives a zero atom.
        X0 = 0.3 * self.r * np.outer(u, vt)    # ||X0||_* = 0.3 * r
        A = ActiveSet.from_svd(X0, radius = self.r)
        self.assertTrue(A.has_zero_atom)
        self.assertEqual(A.n_atoms, 1)

        # Pick G so that the single non-zero atom has negative inner.
        # from_svd applies a sign flip: stored u_atom = -u, vt_atom = vt.
        # S = -r * (-u) * vt^T = r * u * vt^T.
        # <G, S> = r * u^T G vt. For G = +outer(u, vt), this is +r > 0.
        # We want negative, so use G = -outer(u, vt), giving <G, S> = -r.
        G = -np.outer(u, vt)
        k_aw, alpha_aw, inner_aw = A.oracle_away(G)
        self.assertIsNone(k_aw)
        self.assertEqual(inner_aw, 0.0)
        self.assertAlmostEqual(alpha_aw, A.alpha_zero_, places = 12)


    def test_zero_atom_loses_when_a_nonzero_has_positive_inner(self):
        """
        When the zero atom is present but at least one non-zero atom has
        <G, S_k> > 0, the non-zero atom wins (positive > 0 = zero atom's
        inner).
        """
        u = _unit(self.rng, self.n);  vt = _unit(self.rng, self.p)
        X0 = 0.3 * self.r * np.outer(u, vt)
        A = ActiveSet.from_svd(X0, radius = self.r)
        self.assertTrue(A.has_zero_atom)

        # Make the stored atom's inner positive: G = -outer is negative
        # for our atom convention, so flip sign back.
        # Stored: S_atom = r * u * vt^T (after sign flip in from_svd).
        # G = +outer(u, vt) gives <G, S> = +r > 0.
        G = np.outer(u, vt)
        k_aw, alpha_aw, inner_aw = A.oracle_away(G)
        self.assertEqual(k_aw, 0)
        self.assertGreater(inner_aw, 0.0)


    def test_zero_seeded_from_svd_returns_zero_atom_not_raise(self):
        """
        from_svd(zero matrix) seeds an active set with a single zero
        atom of weight 1. oracle_away on that returns is_zero=True.
        """
        A = ActiveSet.from_svd(np.zeros((self.n, self.p)), radius = self.r)
        self.assertTrue(A.has_zero_atom)
        self.assertEqual(A.n_atoms, 0)
        k_aw, alpha_aw, inner_aw = A.oracle_away(
            self.rng.standard_normal((self.n, self.p))
        )
        self.assertIsNone(k_aw)
        self.assertEqual(inner_aw, 0.0)
        self.assertAlmostEqual(alpha_aw, 1.0, places = 12)


# ---------------------------------------------------------------------------
# gamma_max
# ---------------------------------------------------------------------------

class TestActiveSetGammaMax(unittest.TestCase):
    """Step-size upper bound."""

    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.n, self.p, self.r = 5, 20, 10.0


    def test_standard_formula(self):
        """gamma_max = alpha[k] / (1 - alpha[k]) for alpha < 1."""
        u1 = _unit(self.rng, self.n); v1 = _unit(self.rng, self.p)
        u2 = _unit(self.rng, self.n); v2 = _unit(self.rng, self.p)
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(u1, v1, gamma = 1.0)   # alpha = [1.0]
        A.update_fw(u2, v2, gamma = 0.7)   # alpha = [0.3, 0.7]
        self.assertAlmostEqual(A.gamma_max(0), 0.3 / 0.7, places = 12)
        self.assertAlmostEqual(A.gamma_max(1), 0.7 / 0.3, places = 12)


    def test_alpha_one_returns_infinity(self):
        """gamma_max returns +inf when alpha == 1."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        A.update_fw(_unit(self.rng, self.n), _unit(self.rng, self.p), gamma = 1.0)
        self.assertEqual(A.gamma_max(0), np.inf)


# ---------------------------------------------------------------------------
# prune
# ---------------------------------------------------------------------------

class TestActiveSetPrune(unittest.TestCase):
    """Atom removal by relative threshold."""

    def setUp(self):
        self.rng = np.random.default_rng(6)
        self.n, self.p, self.r = 5, 20, 10.0


    def _set_with_known_weights(self, alphas):
        """Build an active set with prescribed alpha weights (sum <= 1)."""
        A = ActiveSet(radius = self.r, shape = (self.n, self.p))
        # Add atoms one by one. After each, weights are scaled by (1-gamma).
        # Easiest: pick a sequence of gammas so the final alphas match.
        # Reverse-engineer: with normalized alphas, just add the first atom
        # with gamma=1, then for atom k>0, gamma = alpha[k] / (sum of alpha[k..]).
        # Simpler: build by direct manipulation since the class supports it.
        u_list = [_unit(self.rng, self.n) for _ in alphas]
        v_list = [_unit(self.rng, self.p) for _ in alphas]
        # Stamp arrays directly.
        A.U_     = -np.column_stack(u_list)            # sign flip, like from_svd
        A.Vt_    =  np.vstack(v_list)
        A.alpha_ = np.asarray(alphas, dtype = float)
        residual = 1.0 - A.alpha_.sum()
        A.alpha_zero_ = max(residual, 0.0)
        return A


    def test_drops_atoms_below_relative_threshold(self):
        """prune drops atoms below eps_rel * total_weight."""
        A = self._set_with_known_weights([0.001, 0.499, 0.5])
        n_dropped = A.prune(eps_rel = 0.01)
        self.assertEqual(n_dropped, 1)
        self.assertEqual(A.n_atoms, 2)


    def test_atom_just_above_threshold_kept(self):
        """Atom with weight just above threshold is kept."""
        # threshold = 0.1 * 1.0 = 0.1; alpha[0] = 0.1 + 1e-15 > threshold -> kept.
        A = self._set_with_known_weights([0.1 + 1e-15, 0.9 - 1e-15])
        n_dropped = A.prune(eps_rel = 0.1)
        self.assertEqual(n_dropped, 0)


    def test_atom_exactly_at_threshold_dropped(self):
        """Atom with weight exactly at threshold is dropped (strict inequality)."""
        # threshold = 0.1 * 1.0 = 0.1; alpha[0] = 0.1 == threshold -> dropped.
        A = self._set_with_known_weights([0.1, 0.9])
        n_dropped = A.prune(eps_rel = 0.1)
        self.assertEqual(n_dropped, 1)
        np.testing.assert_allclose(A.alpha_, [0.9], atol = 1e-12)


    def test_eps_rel_zero_drops_only_exact_zeros(self):
        """prune(eps_rel=0) drops only atoms with exactly zero weight."""
        A = self._set_with_known_weights([0.0, 0.5, 0.5])
        n_dropped = A.prune(eps_rel = 0.0)
        self.assertEqual(n_dropped, 1)
        self.assertEqual(A.n_atoms, 2)
        np.testing.assert_allclose(A.alpha_, [0.5, 0.5], atol = 1e-12)


    def test_dropped_weight_transferred_to_zero_atom(self):
        """Dropped weight is transferred to the zero atom."""
        A = self._set_with_known_weights([0.001, 0.5, 0.499])
        zero_before = A.alpha_zero_
        A.prune(eps_rel = 0.01)
        # Atom 0 (alpha=0.001) is dropped; its weight goes to alpha_zero_.
        self.assertAlmostEqual(
            A.alpha_zero_, zero_before + 0.001, places = 12)
        self.assertAlmostEqual(A.total_weight, 1.0, places = 12)


    def test_empty_set_returns_zero_dropped(self):
        """prune on an empty active set returns 0."""
        A = ActiveSet.from_svd(np.zeros((self.n, self.p)), radius = self.r)
        self.assertEqual(A.prune(eps_rel = 0.5), 0)


# ---------------------------------------------------------------------------
# Cross-cutting invariants
# ---------------------------------------------------------------------------

class TestActiveSetInvariants(unittest.TestCase):
    """Invariants that must hold across all operations."""

    def setUp(self):
        self.rng = np.random.default_rng(7)
        self.n, self.p, self.r = 5, 20, 10.0


    def _build_random_active_set(self):
        """Seed from interior X0, then take a few mixed updates."""
        X0 = _seed_X0(self.rng, self.n, self.p, self.r, scale = 0.5)
        A  = ActiveSet.from_svd(X0, radius = self.r)
        for _ in range(5):
            u  = _unit(self.rng, self.n)
            vt = _unit(self.rng, self.p)
            A.update_fw(u, vt, gamma = self.rng.uniform(0.05, 0.5))
        return A


    def test_total_weight_one(self):
        """Total weight stays at one across operations."""
        A = self._build_random_active_set()
        self.assertAlmostEqual(A.total_weight, 1.0, delta = _WEIGHT_TOL)


    def test_alpha_nonnegative(self):
        """All alpha values stay non-negative."""
        A = self._build_random_active_set()
        self.assertTrue(np.all(A.alpha_ >= 0.0))
        self.assertGreaterEqual(A.alpha_zero_, 0.0)


    def test_u_vt_unit_norm(self):
        """All stored u and vt remain unit-norm."""
        A = self._build_random_active_set()
        u_norms  = np.linalg.norm(A.U_,  axis = 0)
        vt_norms = np.linalg.norm(A.Vt_, axis = 1)
        np.testing.assert_allclose(u_norms,  1.0, atol = 1e-12)
        np.testing.assert_allclose(vt_norms, 1.0, atol = 1e-12)


    def test_reconstruct_inside_nuclear_ball(self):
        """Reconstructed X stays inside the nuclear-norm ball."""
        A = self._build_random_active_set()
        X = A.reconstruct()
        self.assertLessEqual(
            np.linalg.norm(X, 'nuc'), self.r * (1 + _FEAS_TOL))


# ---------------------------------------------------------------------------
# Long-run stability
# ---------------------------------------------------------------------------

class TestActiveSetLongRunSequence(unittest.TestCase):
    """Random-sequence stability over many operations."""

    def setUp(self):
        self.rng = np.random.default_rng(8)
        self.n, self.p, self.r = 5, 20, 10.0


    def test_random_ops_preserve_invariants(self):
        """100 random operations preserve all invariants."""
        A = ActiveSet.from_svd(np.zeros((self.n, self.p)), radius = self.r)
        for t in range(100):
            op = self.rng.choice(['fw', 'away'])
            if op == 'fw' or A.n_atoms == 0:
                u  = _unit(self.rng, self.n)
                vt = _unit(self.rng, self.p)
                A.update_fw(u, vt, gamma = self.rng.uniform(0.0, 1.0))
            else:
                k = int(self.rng.integers(A.n_atoms))
                gmax = A.gamma_max(k)
                # gmax = inf when the selected atom has alpha = 1 (single-atom
                # active set). rng.uniform raises OverflowError on inf range,
                # so cap at an arbitrary finite ceiling for sampling.
                gamma_hi = 1.0 if np.isinf(gmax) else gmax
                A.update_away(k, gamma = self.rng.uniform(0.0, gamma_hi))
            self.assertAlmostEqual(
                A.total_weight, 1.0, delta = _LONG_TOL,
                msg = f"step {t}, op = {op}")
            self.assertTrue(np.all(A.alpha_ >= 0.0), msg = f"step {t}")


    def test_reconstruct_feasible_after_long_run(self):
        """Reconstructed X stays feasible after long run."""
        A = ActiveSet.from_svd(np.zeros((self.n, self.p)), radius = self.r)
        for _ in range(200):
            op = self.rng.choice(['fw', 'away'])
            if op == 'fw' or A.n_atoms == 0:
                u  = _unit(self.rng, self.n)
                vt = _unit(self.rng, self.p)
                A.update_fw(u, vt, gamma = self.rng.uniform(0.0, 1.0))
            else:
                k    = int(self.rng.integers(A.n_atoms))
                gmax = A.gamma_max(k)
                gamma_hi = 1.0 if np.isinf(gmax) else gmax
                A.update_away(k, gamma = self.rng.uniform(0.0, gamma_hi))
        X = A.reconstruct()
        self.assertLessEqual(
            np.linalg.norm(X, 'nuc'), self.r * (1 + _FEAS_TOL))


# ---------------------------------------------------------------------------

if __name__ == '__main__':
    unittest.main(verbosity = 2)
