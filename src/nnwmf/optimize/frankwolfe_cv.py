"""
Nuclear Norm Rank Minimization using Frank-Wolfe algorithm
"""
# Author: Saikat Banerjee

import numpy as np
from .frankwolfe import NNMFW
from ..utils.logs import CustomLogger

class NNMFW_CV():

    def __init__(self, kfolds = 2, test_size = None,
            chain_init = True, reverse_path = True,
            return_fits = True, debug = False,
            **kwargs):
        
        self._kfolds = kfolds
        self._test_size = test_size
        self._return_fits = return_fits
        self._do_chain_initialize = chain_init
        self._do_reverse_path = reverse_path
        
        # Handle NNMFW options
        kwargs.setdefault('suppress_warnings', True)
        kwargs.setdefault('debug', debug)
        self._kwargs = kwargs

        self._is_debug = debug
        self.logger    = CustomLogger(__name__, is_debug = self._is_debug)
        return


    @property
    def training_error(self):
        return self._train_error


    @property
    def test_error(self):
        return self._test_error


    @property
    def nnm_dict(self):
        return self._nnm


    def optimized_rank(self):
        mean_err = {k: np.mean(v) for k,v in self.test_error.items()}
        rank = min(mean_err, key = mean_err.get)
        return rank


    def fit(self, Yin, ranks = None, weight = None, X0 = None):
        """
        Requires centered Y for cross validation.
        Can handle nan in input.
        """
        Y = Yin - np.nanmean(Yin, axis = 0, keepdims = True)
        Y = np.nan_to_num(Y, nan = 0.0)

        # Generate list of ranks for CV
        if ranks is None:
            nucnormY = np.linalg.norm(Y, 'nuc')
            ranks = self.generate_rseq(nucnormY)
        if self._do_reverse_path:
            ranks = ranks[::-1]

        self.logger.debug(f"Cross-validation over {ranks.shape[0]} ranks.")

        # Book keeping
        self._train_error = {r: list() for r in ranks}
        self._test_error  = {r: list() for r in ranks}
        self._nnm         = {r: list() for r in ranks}

        # Loop over folds and ranks for CV
        self._fold_labels = self.generate_fold_labels(Y)
        for k in range(self._kfolds):
            self.logger.debug(f"Fold {k + 1} ...")
            mask = self._fold_labels == k + 1
            Ymiss = self.generate_masked_input(Y, mask)
            Xinit = None if X0 is None else X0.copy()
            for r in ranks:
                #
                # This is the main call to NNMFW
                #
                nnm_cv = NNMFW(**self._kwargs)
                nnm_cv.fit(Ymiss, r, weight = weight, mask = mask, X0 = Xinit)
                #
                test_err_k  = self.get_error(Y, nnm_cv.X, mask)
                train_err_k = self.get_error(Y, nnm_cv.X, ~mask)
                # More bookkeeping
                if self._return_fits:
                    self._nnm[r].append(nnm_cv)
                self._test_error[r].append(test_err_k)
                self._train_error[r].append(train_err_k)
                if self._do_chain_initialize:
                    Xinit = nnm_cv.X
        return


    def generate_rseq(self, rmax):
        nseq  = int(np.floor(np.log2(rmax)) + 1) + 1
        rseq = np.logspace(0, nseq - 1, num = nseq, base = 2.0)
        return rseq


    def generate_fold_labels(self, Y, shuffle = True):
            n, p = Y.shape
            fold_labels = np.ones(n * p)
            if self._test_size is None:
                ntest = int ((n * p) / self._kfolds) 
            else:
                ntest = int(self._test_size * n * p)
            for k in range(1, self._kfolds):
                start = k * ntest
                end = (k + 1) * ntest
                fold_labels[start: end] = k + 1
            if shuffle:
                np.random.shuffle(fold_labels)
            return fold_labels.reshape(n, p)


    def generate_masked_input(self, Y, mask):
        Ymiss_nan = Y.copy()
        Ymiss_nan[mask] = np.nan
        Ymiss_nan_cent = Ymiss_nan - np.nanmean(Ymiss_nan, axis = 0, keepdims = True)
        Ymiss_nan_cent[mask] = 0.0
        return Ymiss_nan_cent


    def get_error(self, original, recovered, mask):
        n = np.sum(mask)
        mse = np.sum(np.square((original - recovered) * mask)) / n
        return np.sqrt(mse)
