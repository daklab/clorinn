"""
Nuclear Norm Rank Minimization using Frank-Wolfe algorithm
"""
# Author: Saikat Banerjee

import numpy as np
from .frankwolfe import FrankWolfe
from ..utils.logs import CustomLogger
from ..utils import model_errors as merr

class FrankWolfe_CV():

    def __init__(self, kfolds = 2, test_size = None,
            chain_init = True, reverse_path = False,
            return_fits = True, debug = False,
            **kwargs):
        
        self.kfolds_ = kfolds
        self.test_size_ = test_size
        self.return_fits_ = return_fits
        self.do_chain_initialize_ = chain_init
        self.do_reverse_path_ = reverse_path
        
        # Handle FrankWolfe options
        kwargs.setdefault('suppress_warnings', True)
        kwargs.setdefault('debug', debug)
        self.kwargs_ = kwargs

        self.is_debug_ = debug
        self.logger_    = CustomLogger(__name__, is_debug = self.is_debug_)
        return


    @property
    def training_error(self):
        return self.train_error_


    @property
    def test_error(self):
        return self.test_error_


    @property
    def cvmodels(self):
        return self.nnm_


    def _optimized_rank(self):
        mean_err = {k: np.mean(v) for k,v in self.test_error_.items()}
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
            ranks = self._generate_rseq(nucnormY)
        if self.do_reverse_path_:
            ranks = ranks[::-1]

        self.logger_.debug(f"Cross-validation over {ranks.shape[0]} ranks.")

        # Book keeping
        self.train_error_ = {r: list() for r in ranks}
        self.test_error_  = {r: list() for r in ranks}
        self.nnm_         = {r: list() for r in ranks}

        # Loop over folds and ranks for CV
        self.fold_labels_ = self._generate_fold_labels(Y)
        for k in range(self.kfolds_):
            self.logger_.debug(f"Fold {k + 1} ...")
            mask = self.fold_labels_ == k + 1
            Ymiss = self._generate_masked_input(Y, mask)
            Xinit = None if X0 is None else X0.copy()
            for r in ranks:
                #
                # Call the main algorithm
                #
                nnm_cv = FrankWolfe(**self.kwargs_)
                nnm_cv.fit(Ymiss, r, weight = weight, mask = mask, X0 = Xinit)
                #
                test_err_k  = merr.get(Y, nnm_cv.X, mask, method = 'rmse')
                train_err_k = merr.get(Y, nnm_cv.X, ~mask, method = 'rmse')
                # More bookkeeping
                if self.return_fits_:
                    self.nnm_[r].append(nnm_cv)
                self.test_error_[r].append(test_err_k)
                self.train_error_[r].append(train_err_k)
                if self.do_chain_initialize_:
                    Xinit = nnm_cv.X
        return


    def _generate_rseq(self, rmax):
        nseq  = int(np.floor(np.log2(rmax)) + 1) + 1
        rseq = np.logspace(0, nseq - 1, num = nseq, base = 2.0)
        return rseq


    def _generate_fold_labels(self, Y, shuffle = True):
            n, p = Y.shape
            fold_labels = np.ones(n * p)
            if self.test_size_ is None:
                ntest = int ((n * p) / self.kfolds_) 
            else:
                ntest = int(self.test_size_ * n * p)
            for k in range(1, self.kfolds_):
                start = k * ntest
                end = (k + 1) * ntest
                fold_labels[start: end] = k + 1
            if shuffle:
                np.random.shuffle(fold_labels)
            return fold_labels.reshape(n, p)


    def _generate_masked_input(self, Y, mask):
        Ymiss_nan = Y.copy()
        Ymiss_nan[mask] = np.nan
        Ymiss_nan_cent = Ymiss_nan - np.nanmean(Ymiss_nan, axis = 0, keepdims = True)
        Ymiss_nan_cent[mask] = 0.0
        return Ymiss_nan_cent
