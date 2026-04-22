from dataclasses import dataclass, field
import numpy as np
from .state import StopReason



@dataclass
class History:
    loss         : list = field(default_factory=list)
    duality_gap  : list = field(default_factory=list)
    step_size    : list = field(default_factory=list)
    cpu_time     : list = field(default_factory=list)
    # sparse-only; None for nnm / nnm-corr
    loss_sparse  : list | None = None
    loss_low_rank: list | None = None



@dataclass
class FitResult:
    X           : np.ndarray
    history     : History
    n_iter      : int
    stop_reason : StopReason
    converged   : bool
    message     : str
    M           : np.ndarray | None = None  # nnm-sparse only
    metrics     : dict = field(default_factory=dict)

    @property
    def loss(self):
        return self.history.loss[-1]

    @property
    def duality_gap(self):
        return self.history.duality_gap[-1]

    @property
    def step_size(self):
        return self.history.step_size[-1]

    @property
    def cpu_time(self):
        return np.sum(self.history.cpu_time)
