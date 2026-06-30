"""
regression_config.py
---------------------
Single source of truth for the solver configuration used in
generate_current_behavior.py and test_current_behavior.py.

Changing anything here and re-running the generator is the intended workflow
when you deliberately modify solver behavior.
"""

# FrankWolfe kwargs shared by all FW fixtures (model is set per-fixture)
FW_CONFIG = dict(
    max_iter=2000,
    svd_method='left-gram',
    stop_criteria=('duality_gap',),
    #stop_criteria=('duality_gap', 'step_size', 'relative_loss', 'relative_dg'),
    tol=1e-4,
    step_tol=1e-4,
    rel_tol=1e-8,
    verbose=0,
)

# ProjectedGradientDescent kwargs
PGD_CONFIG = dict(
    max_iter=50,
    rel_tol=1e-8,
    verbose=0,
    stop_criteria=('relative_loss',),
)

# Constraints
R_NUC   = 100.0  # nuclear-norm radius (active: R_NUC < sv_1(Y) ≈ 146)
L1_MULT = 0.5    # l1 multiplier passed to fit() as r=(R_NUC, L1_MULT)

# Portable tolerances
# used for comparison in different machines allowing BLAS/LAPACK drift.
#
# (solver, model, is_masked, quantity) -> (rtol, atol)
# First match wins. "*" matches any value in that field.
# Convention: more specific rules first, catch-all defaults last.
#
PORTABLE_TOLERANCE_RULES = [
    # solver  model         masked  quantity     (rtol,  atol)
    (("fw",   "nnm",        True,   "step"),    (1e-7,  1e-8)),

    (("fw",   "nnm-sparse", True,   "X"),       (1e-6,  1e-8)),
    (("fw",   "nnm-sparse", True,   "gap"),     (1e-4,  1e-6)),
    (("fw",   "nnm-sparse", True,   "step"),    (1e-5,  1e-8)),

    (("fw",   "nnm-sparse", False,  "gap"),     (1e-5,  1e-8)),
    (("fw",   "nnm-sparse", False,  "step"),    (1e-6,  1e-8)),

    (("fw",   "nnm-corr",   "*",    "X"),       (1e-2,  1e-2)),
    (("fw",   "nnm-corr",   "*",    "loss"),    (1e-6,  1e-8)),

    # Catch-all: tight default for everything not explicitly loosened.
    (("*",    "*",          "*",    "*"),       (1e-8,  1e-10)),
]
