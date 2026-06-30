"""
invariant_utils.py
------------------
Shared problem setup and solver configuration for invariant tests.

_build_problem() returns a dict with Y, mask, noise_cov.
The problem is intentionally small (n=15, p=300) for fast test runs
while being non-trivial enough that solvers take multiple steps.

The solver configs are kept deliberately loose (many iterations, tight
tolerances) so that invariants are stress-tested over a long run rather
than a handful of steps.
"""
# Author: Saikat Banerjee
# License: BSD 3 clause

import numpy as np

from clorinn.tests import toy_data
from clorinn.utils import SamplingCovariance


# ---------------------------------------------------------------------------
# Solver configuration
# ---------------------------------------------------------------------------

# FrankWolfe kwargs shared by all invariant FW runs (model set per-test).
# svd_method='direct' for full determinism across platforms.
FW_CONFIG = dict(
    max_iter        = 2000,
    svd_method      = 'direct',
    stop_criteria   = ['duality_gap', 'step_size', 'relative_loss', 'relative_dg'],
    tol             = 1e-4,
    step_tol        = 1e-4,
    rel_tol         = 1e-8,
    verbose         = 0,
)

# ProjectedGradientDescent kwargs shared by all invariant PGD runs.
PGD_CONFIG = dict(
    max_iter        = 50,
    rel_tol         = 1e-8,
    stop_criteria   = ('relative_loss',),
    verbose         = 0,
)

# Constraints
R_NUC   = 60.0   # nuclear-norm radius  (active: R_NUC < sv_1(Y) ≈ 146)
L1_MULT = 0.5    # l1 multiplier for NNM-Sparse


# ---------------------------------------------------------------------------
# Problem builder
# ---------------------------------------------------------------------------

def _build_problem():
    """
    Build the shared test problem.

    Returns
    -------
    dict with keys:
        Y         : np.ndarray (n, p)  column-centred Z-score matrix
        mask      : np.ndarray (n, p)  bool, 10 % missing entries
        noise_cov : np.ndarray (n, p)  sampling covariance matrix
    """
    np.random.seed(0)
    n, p, k, Q = 15, 300, 4, 2
    Z, _, _, *_ = toy_data.effect_size(
        n, p, k, Q,
        h2=0.4, g2=0.2, aq=0.6, a0=0.2, nsample=8000,
        cov_design='blockdiag', seed=0,
    )
    Y = toy_data.do_standardize(Z, scale=False)

    mask = toy_data.generate_mask(n, p, ratio=0.10, seed=5)

    # PD sampling covariance  A = BB^T/n + 0.5*I
    rng = np.random.default_rng(7)
    B = rng.standard_normal((n, n))
    A = B @ B.T / n + np.eye(n) * 0.5
    noise_cov = SamplingCovariance.from_matrix(A)

    return dict(Y=Y, mask=mask, noise_cov=noise_cov)
