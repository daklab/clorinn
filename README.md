# Colormann

Colormann (**Co**nvex **Lo**w **R**ank **M**atrix **A**pproximation using **N**uclear **N**orm constraint)
extracts the shared and distinct hidden components from a noisy data matrix $\mathbf{X}$
using two steps:
  1. Decompose $\mathbf{X}$ as the sum of a low-rank component ($\mathbf{Y}$) and a sparse component ($\mathbf{M}$).
  Colormann provides a choice of several convex algorithms for the low rank matrix approximation (LRMA),
  using a combination of weighted nuclear norm and $\ell_1$ norm regularization (see below).
  2. Decompose $\mathbf{Y}$, $\mathbf{M}$ or their sum ($\mathbf{Y} + \mathbf{M}$)
  as a product of loadings and factors using singular value decomposition (SVD).

In most cases, the factors obtained from the low-rank component $\mathbf{Y}$ would be the shared hidden component
and the factors obtained from the sparse component $\mathbf{M}$ would be the distinct hidden components.
However, they may have different interpretations depending on the structure of the input data.

## LRMA algorithms

The following algorithms have been implemented:

 - [x] Rank minimization with weighted nuclear norm constraint using Frank-Wolfe algorithm.
 - [x] Rank minimization with nuclear norm and $\ell_1$ norm constraint using Frank-Wolfe algorithm.
 - [x] RobustPCA with inexact augmented Lagrange multiplier algorithm.

## Installation

The software can be installed directly from github using `pip`:
```bash
pip install git+https://github.com/banskt/colormann.git
```

For development, download this repository and install using the `-e` flag:
```bash
git clone https://github.com/banskt/colormann.git colormann # or use the SSH link
cd colormann
pip install -e .
```

## Related algorithms

Also see Sparse PCA, RobustPCA, Weighted PCA, Bayesian PCA.

**PCA is not convex and cannot be made so.**
