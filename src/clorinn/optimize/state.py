from dataclasses import dataclass, field
import numpy as np
from enum import Enum, auto



class StopReason(Enum):
    MAX_ITER      = auto()
    DUALITY_GAP   = auto()
    STEP_SIZE     = auto()
    RELATIVE_LOSS = auto()
    RELATIVE_DG   = auto()
    BOUNDARY_ACTIVE = auto()


    @property
    def message(self):
        stop_messages = {
            self.MAX_ITER:      "Maximum number of iterations reached.",
            self.DUALITY_GAP:   "Duality gap converged below tolerance.",
            self.STEP_SIZE:     "Step size converged below tolerance.",
            self.RELATIVE_LOSS: "Relative change in loss fn converged below tolerance.",
            self.RELATIVE_DG:   "Relative change in duality gap converged below tolerance.",
            self.BOUNDARY_ACTIVE: "Nuclear norm constraint active. Ready for Frank-Wolfe."
        }
        return stop_messages[self]



@dataclass
class IterState:
    """
    Mutable iteration state for Frank-Wolfe solvers.

    Holds everything that changes between iterations and needs to be
    readable by more than one method.  The history lists (loss, duality
    gap, step sizes) are kept separately as local lists in fit() and
    collected into History at the end — they are accumulation buffers,
    not state.
    """
    X               : np.ndarray
    M               : np.ndarray | None = None
    istep           : int               = 0
    last_step_size  : float             = 1.0
    svd_u           : np.ndarray | None = None
    svd_vt          : np.ndarray | None = None
    svd_n_iter      : int | None        = None
    stop_reason     : StopReason = StopReason.MAX_ITER
    # AFW-only fields; remain None / unused under plain FW.
    # forward reference keeps state.py from needing 
    # to import active_set.py (avoids any potential circular import).
    active_set      : 'ActiveSet | None' = None
    last_step_kind  : str | None         = None
