"""
The orthogonal projection onto the unit simplex
is a convex optimization problem.
The unit simplex is defined by,
    S = {x ∈ R^n ∣ x ⪰ 0, ∑x = 1}
Orthogonal projection onto the unit simplex is given by:
    argmin_x 0.5 || y - x ||_2^2
    subject to x ∈ S.
Several algorithms exist for this optimization problem.

Laurent Condat [1] presents a review and comparison 
of existing algorithms with a new proposal for 
projection onto the unit simplex. 
This paper lists the worst-case complexity 
and empirical complexity of those algorithms, 
and presents concise pseudo-code for many algorithms.
Condat has included C and MATLAB implementations 
of all the algorithms mentioned in his paper
here: https://lcondat.github.io/software.html

[1] Condat, L. Fast projection onto the simplex and the 𝑙1 ball. 
    Math. Program. 158, 575–585 (2016). 
    https://doi.org/10.1007/s10107-015-0946-6
"""
# Author: Saikat Banerjee

import numpy as np

class EuclideanProjection():

    def __init__(self, method = 'condat', target = 'simplex'):
        self.method_ = method
        self.target_ = target
        return


    @property
    def proj(self):
        return self.yproj_


    def fit(self, y, a = 1.0):
        #
        if self.target_ == 'simplex':
            assert np.alltrue(y >= 0)
            ypos = y
        elif self.target_ == 'l1':
            ypos = np.abs(y)
        """
        Main methods factory
        """
        if np.sum(ypos) == a:
            # best projection: itself
            yproj = ypos
        else:
            if self.method_ == 'sort':
                yproj = self._simplex_proj_sort(ypos, a)
            elif self.method_ == 'michelot':
                yproj = self._simplex_proj_michelot(ypos, a)
            elif self.method_ == 'condat':
                yproj = self._simplex_proj_condat(ypos, a)
        """
        For l1 projection, put the signs back
        """
        self.yproj_ = yproj
        if self.target_ == 'l1':
            self.yproj_ *= np.sign(y)
        return


    @staticmethod
    def l1_norm(y):
        return np.sum(np.abs(y))


    @staticmethod
    def _simplex_proj_sort(y, a):
        u = np.sort(y)[::-1]
        ukvals = (np.cumsum(u) - a) / np.arange(1, y.shape[0] + 1)
        k = np.nonzero(ukvals < u)[0][-1]
        x = np.clip(y - ukvals[k], a_min=0, a_max=None)
        return x


    @staticmethod
    def _simplex_proj_michelot(y, a):
        auxv = y.copy()
        n = y.shape[0]
        rho = (np.sum(y) - a) / n
        vnorm_last = l1_norm(auxv)
        while True:
            allowed = auxv > rho
            auxv = auxv[allowed]
            nv = np.sum(allowed)
            vnorm = l1_norm(auxv)
            if vnorm == vnorm_last:
                break
            rho = (np.sum(auxv) - a) / nv
            vnorm_last = vnorm
        x = np.clip(y - rho, a_min = 0, a_max = None)
        return x


    @staticmethod
    def _simplex_proj_condat(y, a):
        auxv = np.array([y[0]])
        vtilde = np.array([])
        rho = y[0] - a
        # Step 2
        for yn in y[1:]:
            if yn > rho:
                rho += (yn - rho) / (auxv.shape[0] + 1)
                if rho > (yn - a):
                    auxv = np.append(auxv, yn)
                else:
                    vtilde = np.append(vtilde, auxv)
                    auxv = np.array([yn])
                    rho = yn - a
        # Step 3
        if vtilde.shape[0] > 0:
            for v in vtilde:
                if v > rho:
                    auxv = np.append(auxv, v)
                    rho += (v - rho) / (auxv.shape[0])                
        # Step 4
        nv_last = auxv.shape[0]
        istep = 0
        while True:
            istep += 1
            to_remove = list()
            nv_ = auxv.shape[0]
            for i, v in enumerate(auxv):
                if v <= rho:
                    to_remove.append(i)
                    nv_ = nv_ - 1
                    rho += (rho - v) / nv_
            auxv = np.delete(auxv, to_remove)
            nv = auxv.shape[0]
            assert nv == nv_
            if nv == nv_last:
                break
            nv_last = nv
        # Step 5
        x = np.clip(y - rho, a_min=0, a_max=None)
        return x
