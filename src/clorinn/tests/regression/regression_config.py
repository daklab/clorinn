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
R_NUC   = 60.0   # nuclear-norm radius (active: R_NUC < sv_1(Y) ≈ 146)
L1_MULT = 0.5    # l1 multiplier passed to fit() as r=(R_NUC, L1_MULT)
