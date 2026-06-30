# Clorinn

> _Multi-trait factor analysis of GWAS Z-scores is brittle. <br> Clorinn makes it convex._

Clorinn decomposes a `trait` $\times$ `variant` matrix of GWAS Z-scores,
into a **low-rank shared signal** and a **sparse component**,
separating genetic structure from noise, outliers, and trait-specific effects.
The result is a denoised matrix suitable for stable factor analysis.
Convex optimization guarantees that
the solution is **independent of initialization**,
**reproducible across runs**, and **robust** to small changes in the input data.

Clorinn is an acronym for 
**C**onvex **Lo**w-**R**ank **I**nference via **N**uclear **N**orm constraint.

---

## Why?

    You run PCA on 2,000 GWAS traits.
    The top factor is dominated by a single artifact in one trait.
    You add a new cohort.
    Every factor changes.
    You fix a data error in one trait.
    Every factor changes again.

This is not a bug -- this is because GWAS data is not Gaussian:
outliers dominate, entries are missing,
and errors are correlated across traits.
In such cases, a single artifact can dominate the leading factors.
Clorinn is designed to handle the noise that actually appears in the data.

## When to use Clorinn?

Use Clorinn instead of PCA / SVD when:

- a few traits or SNPs dominate the top components
- results change drastically when adding/removing traits
- data has missing entries
- traits share samples (correlated errors)

If your data is clean, Gaussian, and well-behaved, 
PCA and/or truncated SVD is usually fine.

## Approach

The observed Z-score matrix $\mathbf{Z}$ is noisy.
The shared genetic signal lives in a low-rank matrix $\mathbf{X}$
that we cannot observe directly.
Clorinn recovers $\mathbf{X}$ from $\mathbf{Z}$ by solving
a convex problem of the form,

$$\min f(\mathbf{X}) \quad \text{s.t.} \quad \lVert\mathbf{X}\rVert_* \le r\ ,$$

where $`\lVert\mathbf{X}\rVert_*`$ is the nuclear norm (sum of singular values), 
a convex surrogate for matrix rank.
The constraint radius $r$ controls how much shared structure is retained.
The objective $f(\mathbf{X})$ and the constraints can be adapted
to different noise structures in the data (see Models).

Once the denoised $\mathbf{X}$ is recovered,
truncated SVD extracts the latent factors and trait loadings from
$\mathbf{X}$ rather than the raw $\mathbf{Z}$ --
so the resulting factors are stable, reproducible, and not dominated by outliers.

## Models and Solvers

Clorinn separates **what you optimize** (the model)
from **how you optimize it** (the solver).
In practice, you pick one of each:

| Models  | `minimize` | `subject to` | Use when |
|:---|:---|:---|:---|
| **NNM** | $`\dfrac{1}{2}\lVert\mathbf{Z} - \mathbf{X}\rVert_F^2`$ | $`\lVert\mathbf{X}\rVert_* \le r`$ | Independent traits, isotropic noise |
| **NNM-Sparse** | $`\dfrac{1}{2}\lVert\mathbf{Z} - \mathbf{X} - \mathbf{M}\rVert_F^2`$ | $`\lVert\mathbf{X}\rVert_* \le r`$, $`\lVert\mathbf{M}\rVert_1 \le l`$ | Sparse outliers or trait-specific effects |
| **NNM-Corr**<sup>1</sup> | $`\dfrac{1}{2} \sum_{i=1}^P \lVert \mathbf{z}_{i} - \mathbf{x}_{i} \rVert_{\mathbf{A}^{-1}}^2`$ | $`\lVert\mathbf{X}\rVert_* \le r`$ | Correlated errors <br> _e.g._ traits share samples |

<sup>1</sup>For NNM-Corr,
$`\lVert \mathbf{u} \rVert_{\mathbf{A}^{-1}}^2 = \mathbf{u}^{\mathsf{T}}\mathbf{A}^{-1}\mathbf{u}`$.

<!-- ^{\mathsf{T}} \mathbf{A}^{-1} (\mathbf{z}_i - \mathbf{x}_i)`$ -->

| Solvers | Algorithm | Models supported |
|:---|:---|:---|
| **FW** | Frank-Wolfe | NNM, NNM-Sparse, NNM-Corr |
| **AFW** | Away-step Frank-Wolfe | NNM, NNM-Sparse, NNM-Corr |
| **PGD** | Projected Gradient Descent | NNM, NNM-Sparse, NNM-Corr |

**Typical choices**

- Start with **NNM + FW** (default)
- Use **NNM-Sparse** if a few entries dominate
- Use **NNM-Corr** if traits share samples
- Use **PGD → AFW** for faster convergence on large problems

## What's new in v2.0.0

- Correlated-error model (NNM-Corr) using sampling covariance
- Explicit handling of missing data (pattern-based computation)
- Projected Gradient Descent (PGD) solver
- Hybrid PGD → Frank–Wolfe workflow
- Cleaner API


## Installation

```bash
# From GitHub
pip install git+https://github.com/daklab/clorinn.git

# Development install
git clone https://github.com/daklab/clorinn.git
cd clorinn
pip install -e .
```

**Requirements:** Python ≥ 3.10, NumPy, SciPy, scikit-learn.

=======
## Quick start

### NNM — fully observed

```python
from clorinn.optimize import FrankWolfe
from clorinn.utils import MatrixFactorization

fw = FrankWolfe(model='nnm')
fw.fit(Z, radius=r)

mf = MatrixFactorization(k=10)
mf.fit(fw.result.X)

L = mf.L   # trait loadings  (N, k)
F = mf.F   # hidden factors  (P, k)
```

### NNM-Sparse

```python
fw = FrankWolfe(model='nnm-sparse')
fw.fit(Z, radius=r, sparse_scale=0.5)

X = fw.result.X   # shared low-rank component
M = fw.result.M   # sparse trait-specific component
```

### NNM-Corr — correlated sampling errors

```python
from clorinn.utils import SamplingCovariance

# Option 1: pass a plain ndarray
fw = FrankWolfe(model='nnm-corr')
fw.fit(Z, radius=r, noise_cov=A)

# Option 2: explicitly validate before fitting
cov = SamplingCovariance.from_matrix(A)
fw.fit(Z, radius=r, noise_cov=cov)
print(cov.repair_info)
# {'n_iter': 12, 'converged': True, 'min_eig_input': -0.003, ...}
```

### Hybrid solver (PGD → Away Step FrankWolfe)

```python
from clorinn.optimize import ProjectedGradientDescent, FrankWolfe

model = 'nnm' # or 'nnm-corr', 'nnm-sparse'
pgd = ProjectedGradientDescent(
    model=model,
    stop_criteria=('relative_loss', 'boundary_active'),
    verbose=1,
)
afw = AwayStepFrankWolfe(model=model)

fit_kwargs = {}
if model == 'nnm-corr':
    fit_kwargs['noise_cov'] = A
elif model == 'nnm-sparse':
    fit_kwargs['sparse_scale'] = 0.5

pgd.fit(Z, radius=r, **fit_kwargs)
afw.fit(Z, radius=r, X0=pgd.result.X, **fit_kwargs)
```

## Sampling covariance

`SamplingCovariance` constructs and validates the $N \times N$ 
sampling covariance matrix $\mathbf{A}$ assembled from LD Score Regression intercepts
(diagonal: univariate LDSC intercepts; off-diagonal: bivariate LDSC intercepts).

```python
from clorinn.utils import SamplingCovariance

# From a pre-assembled matrix
cov = SamplingCovariance.from_matrix(A, repair=True, verbose=1)

# from_ldsc() — planned for v2.1
# cov = SamplingCovariance.from_ldsc(ldsc_intercepts, std_errors)

print(cov.A)            # validated PD matrix
print(cov.is_repaired)  # True if Higham repair was applied
print(cov.repair_info)  # n_iter, min_eig_input, min_eig_output, reg_applied
```

Because pairwise LDSC intercept estimates are noisy,
the assembled matrix is not guaranteed to be positive definite.
`repair=True` applies Higham's nearest-PD algorithm with Dykstra correction.

## Running tests

```bash
# All tests, silent
clorinn --test

# All tests, progress output
clorinn --test --verbose

# All tests, debug output
clorinn --test --vverbose

# Specific test class
clorinn --test --testmodule TestRegressionNNMCorr
```

The test suite is organised into four directories
under `src/clorinn/tests/`:

```
tests/
    unit/         Tests for individual classes and methods in isolation
    regression/   Numerical regression fixtures (freeze solver traces)
    invariants/   Solver invariants (feasibility, monotonicity, gaps)
    theory/       Mathematical correctness (exact missingness, gradients)
    integration/  Robustness on degenerate inputs
```

To regenerate regression fixtures after an intentional solver change:

```bash
python -m clorinn.tests.regression.generate_current_behavior
```

## Related work

Clorinn is related to Sparse PCA, Robust PCA,
nuclear norm minimisation for matrix completion,
and latent factor models for GWAS.
Key comparators evaluated in the paper:
truncated SVD [Tanigawa et al. 2019],
FactorGo [Zhang et al. 2023],
flashier [Willwerscheid et al. 2024],
GUIDE [Lazarev et al. 2025],
and GLEANR [Omdahl et al. 2025].

<!-- **PCA is not convex and cannot be made so.** -->

## Citation

 [1] Banerjee S, O'Connell S, Colbert SMC, Mullins N, Knowles DA.
     *Convex approaches to isolate the shared and distinct genetic components
     of complex traits.* 2025.
     [medRxiv 2025.04.15.25325870](https://doi.org/10.1101/2025.04.15.25325870)
