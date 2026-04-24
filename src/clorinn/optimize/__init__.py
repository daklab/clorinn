from .frankwolfe import FrankWolfe
from .pgd import ProjectedGradientDescent
PGDWarmStart = ProjectedGradientDescent   # backward-compatible alias
from .frankwolfe_cv import FrankWolfe_CV
from .inexact_alm import IALM
from .projections import EuclideanProjection, NuclearNormProjection
