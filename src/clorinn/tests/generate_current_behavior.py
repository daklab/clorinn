"""
generate_current_behavior.py
======================================
Run this script once to snapshot the current solver behavior. Each fixture
is a .npz file containing the numerical inputs and every output the solver
produced. test_current_behavior.py replays the same call (using
the shared solver config defined in regression_config.py) and asserts
bit-for-bit identical results.

Usage
-----
    pip install -e . --no-build-isolation
    python -m clorinn.tests.generate_current_behavior

Fixture files are written to src/clorinn/tests/fixtures/.

Re-run only when you intentionally change solver behavior. Commit the
updated .npz files alongside the code change so the diff documents what
shifted in the iteration traces.

Fixture schema
--------------
Every .npz stores only numerical inputs and outputs. The solver config is
intentionally NOT stored here — it lives in regression_config.py, shared
with the test file, so there is a single source of truth.

  Inputs (FrankWolfe)
  -------------------
  Y             float64 (n, p)    input matrix
  r             float64 scalar    nuclear-norm constraint radius
  mask          bool    (n, p)    [omitted when not used]
  L_inv         float64 (n, n)    [omitted when not used]
  Sigma_inv     float64 (n, n)    [omitted when not used]
  l1_multiplier float64 scalar    multiplier passed as r=(r, mult) [omitted when not used]

  Outputs (FrankWolfe)
  --------------------
  X             float64 (n, p)
  M             float64 (n, p)    [omitted when not sparse]
  fx            float64 (n_iter+1,)
  dg            float64 (n_iter+1,)
  steps         float64 (n_iter+1,)
  n_iter        int64   scalar

  Inputs / Outputs (PGDWarmStart)
  --------------------------------
  Y             float64 (n, p)
  r             float64 scalar
  mask          bool    (n, p)    [omitted when not used]
  X             float64 (n, p)
  fx            float64 (n_iter,)
  n_iter        int64   scalar
  converged_in_interior  bool scalar
"""

import os
import numpy as np

from clorinn.optimize import FrankWolfe, PGDWarmStart
from clorinn.tests import toy_data
from clorinn.tests.regression_config import FW_CONFIG, PGD_CONFIG, R_NUC, L1_MULT

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
os.makedirs(FIXTURES_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Shared problem  (fixed seeds)
# ---------------------------------------------------------------------------

def _build_problem():
    """
    n=15 traits, p=300 SNPs, k=4 latent factors, Q=2 trait blocks.
    Small enough for fast test runs; non-trivial enough that the solver
    takes more than one step on most paths.
    """
    np.random.seed(0)
    n, p, k, Q = 15, 300, 4, 2
    Z, _, _, *_ = toy_data.effect_size(
        n, p, k, Q,
        h2=0.4, g2=0.2, aq=0.6, a0=0.2, nsample=8000,
        cov_design='blockdiag', seed=0,
    )
    Y = toy_data.do_standardize(Z, scale=False)   # column-center, no scaling

    mask = toy_data.generate_mask(n, p, ratio=0.10, seed=5)

    # PD sampling covariance  A = BB'/n + 0.5*I
    rng = np.random.default_rng(7)
    B = rng.standard_normal((n, n))
    A = B @ B.T / n + np.eye(n) * 0.5
    L_chol = np.linalg.cholesky(A)
    L_inv = np.linalg.solve(L_chol, np.eye(n))
    Sigma_inv = L_inv.T @ L_inv                    # == A^{-1}

    return dict(Y=Y, mask=mask, L_inv=L_inv, Sigma_inv=Sigma_inv)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------

def _save(path, **arrays):
    np.savez_compressed(path, **arrays)
    print(f"  saved {os.path.relpath(path)}")


# ---------------------------------------------------------------------------
# Individual fixture generators
# ---------------------------------------------------------------------------

def gen_nnm(prob):
    """TC1: NNM, fully observed."""
    m = FrankWolfe(model='nnm', **FW_CONFIG)
    m.fit(prob['Y'], r=R_NUC)
    _save(
        os.path.join(FIXTURES_DIR, 'nnm.npz'),
        Y=prob['Y'], r=np.float64(R_NUC),
        X=m.result.X, fx=np.array(m.result.history.loss), dg=np.array(m.result.history.duality_gap),
        steps=np.array(m.result.history.step_size), n_iter=np.int64(m.result.n_iter),
    )


def gen_nnm_mask(prob):
    """TC2: NNM, 10 % missing data."""
    m = FrankWolfe(model='nnm', **FW_CONFIG)
    m.fit(prob['Y'], r=R_NUC, mask=prob['mask'])
    _save(
        os.path.join(FIXTURES_DIR, 'nnm_mask.npz'),
        Y=prob['Y'], r=np.float64(R_NUC), mask=prob['mask'],
        X=m.result.X, fx=np.array(m.result.history.loss), dg=np.array(m.result.history.duality_gap),
        steps=np.array(m.result.history.step_size), n_iter=np.int64(m.result.n_iter),
    )


def gen_nnm_sparse(prob):
    """TC3: NNM-Sparse, fully observed."""
    m = FrankWolfe(model='nnm-sparse', **FW_CONFIG)
    m.fit(prob['Y'], r=(R_NUC, L1_MULT))
    _save(
        os.path.join(FIXTURES_DIR, 'nnm_sparse.npz'),
        Y=prob['Y'], r=np.float64(R_NUC), l1_multiplier=np.float64(L1_MULT),
        X=m.result.X, M=m.result.M, fx=np.array(m.result.history.loss), dg=np.array(m.result.history.duality_gap),
        steps=np.array(m.result.history.step_size), n_iter=np.int64(m.result.n_iter),
    )


def gen_nnm_sparse_mask(prob):
    """TC4: NNM-Sparse, 10 % missing data."""
    m = FrankWolfe(model='nnm-sparse', **FW_CONFIG)
    m.fit(prob['Y'], r=(R_NUC, L1_MULT), mask=prob['mask'])
    _save(
        os.path.join(FIXTURES_DIR, 'nnm_sparse_mask.npz'),
        Y=prob['Y'], r=np.float64(R_NUC), l1_multiplier=np.float64(L1_MULT),
        mask=prob['mask'],
        X=m.result.X, M=m.result.M, fx=np.array(m.result.history.loss), dg=np.array(m.result.history.duality_gap),
        steps=np.array(m.result.history.step_size), n_iter=np.int64(m.result.n_iter),
    )


def gen_nnm_corr(prob):
    """TC5: NNM-Corr (Mahalanobis loss), fully observed."""
    m = FrankWolfe(model='nnm-corr', **FW_CONFIG)
    m.fit(prob['Y'], r=R_NUC, L_inv=prob['L_inv'], Sigma_inv=prob['Sigma_inv'])
    _save(
        os.path.join(FIXTURES_DIR, 'nnm_corr.npz'),
        Y=prob['Y'], r=np.float64(R_NUC),
        L_inv=prob['L_inv'], Sigma_inv=prob['Sigma_inv'],
        X=m.result.X, fx=np.array(m.result.history.loss), dg=np.array(m.result.history.duality_gap),
        steps=np.array(m.result.history.step_size), n_iter=np.int64(m.result.n_iter),
    )


def gen_nnm_corr_mask(prob):
    """TC6: NNM-Corr, 10 % missing data."""
    m = FrankWolfe(model='nnm-corr', **FW_CONFIG)
    m.fit(prob['Y'], r=R_NUC, mask=prob['mask'],
          L_inv=prob['L_inv'], Sigma_inv=prob['Sigma_inv'])
    _save(
        os.path.join(FIXTURES_DIR, 'nnm_corr_mask.npz'),
        Y=prob['Y'], r=np.float64(R_NUC), mask=prob['mask'],
        L_inv=prob['L_inv'], Sigma_inv=prob['Sigma_inv'],
        X=m.result.X, fx=np.array(m.result.history.loss), dg=np.array(m.result.history.duality_gap),
        steps=np.array(m.result.history.step_size), n_iter=np.int64(m.result.n_iter),
    )


def gen_pgd(prob):
    """TC7: PGD warm start, fully observed."""
    pgd = PGDWarmStart(**PGD_CONFIG)
    pgd.fit(prob['Y'], r=R_NUC)
    _save(
        os.path.join(FIXTURES_DIR, 'pgd.npz'),
        Y=prob['Y'], r=np.float64(R_NUC),
        X=pgd.result.X, fx=np.array(pgd.result.history.loss),
        n_iter=np.int64(pgd.result.n_iter),
        converged_in_interior=np.bool_(pgd.result.converged),
    )


def gen_pgd_mask(prob):
    """TC8: PGD warm start, 10 % missing data."""
    pgd = PGDWarmStart(**PGD_CONFIG)
    pgd.fit(prob['Y'], r=R_NUC, mask=prob['mask'])
    _save(
        os.path.join(FIXTURES_DIR, 'pgd_mask.npz'),
        Y=prob['Y'], r=np.float64(R_NUC), mask=prob['mask'],
        X=pgd.result.X, fx=np.array(pgd.result.history.loss),
        n_iter=np.int64(pgd.result.n_iter),
        converged_in_interior=np.bool_(pgd.result.converged),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

GENERATORS = [
    gen_nnm, gen_nnm_mask,
    gen_nnm_sparse, gen_nnm_sparse_mask,
    gen_nnm_corr, gen_nnm_corr_mask,
    gen_pgd, gen_pgd_mask,
]


def main():
    print(f"Writing fixtures to: {FIXTURES_DIR}\n")
    prob = _build_problem()
    for gen in GENERATORS:
        print(f"[{gen.__name__}]  {gen.__doc__.strip()}")
        gen(prob)
        print()
    print(f"Done. {len(GENERATORS)} fixtures written.")


if __name__ == '__main__':
    main()
