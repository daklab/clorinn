from dataclasses import dataclass

@dataclass
class SolverConfig:
    max_iter       : int   = 1000
    svd_method     : str   = 'power'
    svd_max_iter   : int | None = None
    stop_criteria  : tuple = ('duality_gap', 'step_size', 'relative_loss', 'relative_dg')
    tol            : float = 1e-3
    step_tol       : float = 1e-3
    rel_tol        : float = 1e-8
    simplex_method : str = 'sort'
