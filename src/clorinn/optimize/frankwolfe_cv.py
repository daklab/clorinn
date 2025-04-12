# Author: Saikat Banerjee

import numpy as np
from .frankwolfe import FrankWolfe
from ..utils.logs import CustomLogger
from ..utils import model_errors as merr

class FrankWolfe_CV():
    """
    Cross-validation via iterative optimization using Frank-Wolfe algorithm
    along a path/sequence of constraints on the nuclear norm of the input matrix.

    Parameters
    ----------
        kfolds : integer, default=2
            Split datasets into k folds, must be at least 2.

        test_size : float, default=None
            Fraction of test data for a single split. If specified,
            then each fold contains `test_size * n * p` elements, where `(n,p)`
            is the size of the input matrix. If set to None, then it is
            automatically estimated from the number of folds.

        shuffle : boolean, default=True
            Whether to shuffle the fold indices before splitting the data
            into batches. If set to False, each fold will be consecutive.
        
        chain_init : boolean, default=True
            If set to False, each FW optimization is initialized from zero. If
            set to True, each successive FW optimization along the path of 
            constraints is initialized from the optimum low rank matrix obtained
            from the previous constraint.

        reverse_path : boolean, default=False
            Whether to use decreasing values of the constraints (reverse the path
            of constraints) on the nuclear norm.

        return_fits : boolean, default=True
            Whether to keep the FrankWolfe class for all constraints in memory.

        debug : boolean, default=False
            Whether to provide a verbose output for debugging the algorithm.

        Any parameter from the FrankWolfe class can also be specified as 
        optional inputs.

    Attributes
    ----------
        training_error : dict{r: list<float>}
            RMSE between the training data and the estimated matrix for all the `k` folds
            at each constraint `r` along the path.

        test_error : dict{r: list<float>}
            RMSE between the test data and the estimated matrix for all the `k` folds
            at each constraint `r` along the path.

        cvmodels : dict{r: list<FrankWolfe>}
            Fitted models for all the `k` folds at each constraint `r` along the path.


    Notes
    -----
    Use `.fit()` to perform the optimization along the path of constraints.
    The optimum constraint can be obtained from `training_error` or `test_error`.
    The function `._optimized_rank()` estimates the optimum constraint from the
    `test_error`. 


    Examples
    --------
    >>> import FrankWolfe_CV
    >>> nnmcv = FrankWolfe_CV(kfolds = 2, model = 'nnm')
    >>> nnmcv.fit(Y)
    >>> r_opt = nnmcv._optimized_rank()

    """

    def __init__(self, kfolds = 2, test_size = None, shuffle = True,
            chain_init = True, reverse_path = False,
            return_fits = True, debug = False,
            **kwargs):
        
        self.kfolds_ = kfolds
        self.do_shuffle_ = shuffle
        self.test_size_ = test_size
        self.return_fits_ = return_fits
        self.do_chain_initialize_ = chain_init
        self.do_reverse_path_ = reverse_path
        
        # Handle FrankWolfe options
        kwargs.setdefault('suppress_warnings', True)
        kwargs.setdefault('debug', debug)
        self.kwargs_ = kwargs

        self.is_debug_ = debug
        self.logger_   = CustomLogger(__name__, is_debug = self.is_debug_)
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


    def fit(self, Y, rseq = None, weight = None, X0 = None):
        """
        Wrapper function for performing the cross-validation.

        Parameters
        ----------
        Y : np.ndarray [size (n, p); dtype: float]
            Input data matrix with n rows and p columns. May have missing values as np.nan.

        rseq : np.ndarray [size (m,); dtype: float], default=None
            Path/sequence of nuclear norm constraints for the cross-validation. If set
            to None, a path is automatically estimated from the input data.

        weight : np.ndarray [size (n, p); dtype: float], default=None
            An array of weights for each element in the input matrix Y. If set to None,
            all weights are assumed to be 1.

        X0 : np.ndarray [size (n, p); dtype: float], default=None
            Optional initial guess for the low rank matrix in the FW algorithm. If set
            to None, the optimization is initiated from a matrix of zeros.

        Notes
        -----
        The input matrix Y is centered before cross-validation.
        
        """
        nan_mask = np.isnan(Y) # may contain NaN
        Y_cent = Y - np.nanmean(Y, axis = 0, keepdims = True)
        # initialize the missing NaN entries to zero, 
        # will be updated by the FW algorithm.
        # Y_cent = np.nan_to_num(Y_cent, nan = 0.0)

        # Generate list of rseq for CV
        if rseq is None:
            # Nuclear Norm cannot handle NaN values, set them to 0.
            nucnormY = np.linalg.norm(np.nan_to_num(Y_cent, nan = 0.0), 'nuc')
            rseq = self._generate_rseq(nucnormY)
        if self.do_reverse_path_:
            rseq = rseq[::-1]

        self.logger_.debug(f"Cross-validation over {rseq.shape[0]} rseq.")

        # Book keeping
        self.train_error_ = {r: list() for r in rseq}
        self.test_error_  = {r: list() for r in rseq}
        self.nnm_         = {r: list() for r in rseq}

        # Loop over folds and rseq for CV
        self.fold_labels_ = self._generate_fold_labels(Y_cent)
        for k in range(self.kfolds_):
            self.logger_.debug(f"Fold {k + 1} ...")
            # Mask the fold test indices: Set them to NaN.
            # Plus there could be other NaN elements in Y_train
            test_mask = self.fold_labels_ == k + 1
            Y_test  = self._generate_masked_input(Y_cent, test_mask)
            Y_train = self._generate_masked_input(Y_cent, ~test_mask)
            Xinit = None if X0 is None else X0.copy()
            for r in rseq:
                self.logger_.debug(f"Rank {r:.4f}")
                #
                # Call the main algorithm on Y_train.
                # Let FrankWolfe handle missingness.
                #
                nnm_cv = FrankWolfe(**self.kwargs_)
                nnm_cv.fit(Y_train, r, weight = weight, mask = None, X0 = Xinit)
                #
                test_err  = merr.get(Y_test, nnm_cv.X, method = 'rmse')
                train_err = merr.get(Y_train, nnm_cv.X, method = 'rmse')
                # More bookkeeping
                if self.return_fits_:
                    self.nnm_[r].append(nnm_cv)
                self.test_error_[r].append(test_err)
                self.train_error_[r].append(train_err)
                if self.do_chain_initialize_:
                    Xinit = nnm_cv.X
        return


    def _generate_rseq(self, rmax):
        """
        Generate a sequence (path) of constraints given the maximum allowed constraint.
        The lowest constraint is 1, and each constraint is `log2` spaced.
        """
        nseq  = int(np.floor(np.log2(rmax)) + 1) + 1
        rseq = np.logspace(0, nseq - 1, num = nseq, base = 2.0)
        return rseq


    def _generate_fold_labels(self, Y):
        """
        Provides train/test indices to split data in train/test sets.
        """
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
        if self.do_shuffle_:
            np.random.shuffle(fold_labels)
        return fold_labels.reshape(n, p)


    def _generate_masked_input(self, Y, mask):
        """
        Puts nan value to maked indices of the input matrix Y
        without overwriting.
        """
        Y_miss = Y.copy()
        Y_miss[mask] = np.nan
        return Y_miss
