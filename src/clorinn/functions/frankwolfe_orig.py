import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd

def f_objective(X, Y, mask = None):
    '''
    Objective function
    Y is observed, X is estimated
    '''
    Xmask = X if mask is None else X * mask
    return 0.5 * np.linalg.norm(Xmask - Y, 'fro')**2


def f_gradient(X, Y, mask = None):
    '''
    Gradient of the objective function.
    '''
    Xmask = X if mask is None else X * mask
    return Xmask - Y


def f_rmse(X, Y, mask = None):
    '''
    RMSE for masked CV
    '''
    Xmask = X if mask is None else X * mask
    return np.sqrt(np.mean(np.square(Xmask - Y)))


def linopt_oracle(grad, r = 1.0):
    '''
    Linear optimization oracle,
    where the feasible region is a nuclear norm ball for some r
    '''
    U1, V1_T = singular_vectors_power_method(grad, max_iter = 5)
    S = - r * U1 @ V1_T
    return S


def singular_vectors_randomized_method(X, max_iter = 10):
    u, s, vh = randomized_svd(X, n_components = 1, n_iter = max_iter,
                              power_iteration_normalizer = 'none',
                              random_state = 0)
    return u, vh


def singular_vectors_power_method(X, max_iter = 10):
    '''
    Power method.
        
        Computes approximate top left and right singular vector.
        
    Parameters:
    -----------
        X : array {m, n},
            input matrix
        max_iter : integer, optional
            number of steps
            
    Returns:
    --------
        u, v : (n, 1), (p, 1)
            two arrays representing approximate top left and right
            singular vectors.
    '''
    n, p = X.shape
    u = np.random.normal(0, 1, n)
    u /= np.linalg.norm(u)
    v = X.T.dot(u)
    v /= np.linalg.norm(v)
    for _ in range(max_iter):      
        u = X.dot(v)
        u /= np.linalg.norm(u)
        v = X.T.dot(u)
        v /= np.linalg.norm(v)       
    return u.reshape(-1, 1), v.reshape(1, -1)


def frank_wolfe_minimize_step(X, Y, r, istep, mask = None):
    G    = f_gradient(X, Y, mask = mask)
    S    = linopt_oracle(G, r)
    dg   = np.trace((X - S).T @ G)
    step = 2. / (2 + istep)
    #err  = old_X + S
    #step = np.sum((G * err) / np.square(err))
    #step = min(step, 1)
    Xnew = X + step * (S - X)
    return Xnew, G, dg, step


def frank_wolfe_minimize(Y, weight, r, X0 = None,
                         mask = None,
                         max_iter = 1000, tol = 1e-8,
                         return_all = True,
                         debug = False):
    
    # Step 0
    old_X = np.zeros_like(Y) if X0 is None else X0.copy()
    dg = np.inf

    if return_all:
        dg_list = [dg]
        fx_list = [f_objective(old_X, Y, mask)]
        
    # Steps 1, ..., max_iter
    for istep in range(max_iter):
        X, G, dg, step = \
            frank_wolfe_minimize_step(old_X, Y, r, istep, mask)

        if return_all:
            dg_list.append(dg)
            fx_list.append(f_objective(X, Y))
        
        if debug:
            if (istep % 10 == 0):
                print (f"Iteration {istep}. Step size {step:.3f}. Duality Gap {dg:g}")
        if np.abs(dg) <= tol:
            break
            
        old_X = X.copy()
        
    if return_all:
        return X, dg_list, fx_list
    else:
        return X


def is_monotonically_increasing(x):
    return all(x[i] >= x[i - 1] for i in range(1, x.shape[0]))


def frank_wolfe_cv_minimize(Y, weight, X0 = None,
                            r_seq = None,
                            max_iter = 1000, tol = 1e-8,
                            test_size = 0.33,
                            return_all = False,
                            debug_fw = False, debug = False):
    
    # Prepare CV masks
    n, p = Y.shape
    train_mask = np.ones(n * p)
    train_mask[:int(test_size * n * p)] = 0
    np.random.shuffle(train_mask)
    train_mask = train_mask.reshape(n, p)
    test_mask  = 1 - train_mask
    
    Ytrain = Y * train_mask
    Ytest  = Y * test_mask
    
    # Prepare rseq
    if r_seq is None:
        r_min = 1
        r_max = 4.0 * nuclear_norm(Y)
        nseq  = int(np.floor(np.log2(r_max)) + 1) + 1
        #r_seq = np.logspace(-ndec, nseq - 1, num = nseq + ndec, base = 2.0)
        r_seq = np.logspace(0, nseq - 1, num = nseq, base = 2.0)
        
    train_err_dict = dict()
    test_err_dict = dict()
    old_X = None
    
    if debug:
        print (f"Perform CV at {nseq} positions.")
        print (r_seq)
    
    for iseq, r in enumerate(r_seq):
        X = frank_wolfe_minimize(Ytrain, weight, r, 
                                 max_iter = max_iter, X0 = old_X, tol = tol,
                                 mask = train_mask, return_all = False, debug = debug_fw) 
        train_err = f_rmse(X, Ytrain, train_mask)
        test_err  = f_rmse(X, Ytest, test_mask)
        if debug:
            print(f"CV sequence {iseq + 1}, r = {r}, training error = {train_err}, test_error = {test_err}")
        train_err_dict[r] = train_err
        test_err_dict[r] = test_err
        old_X = X.copy()
        
    return r_seq, train_err_dict, test_err_dict


def nuclear_norm(X):
    '''
    Nuclear norm of input matrix
    '''
    return np.sum(np.linalg.svd(X)[1])


# Code taken from https://gist.github.com/daien/1272551
def simplex_projection(s, alpha = 1.0):
    '''
    Compute the Euclidean projection on a positive simplex
    Solves the optimisation problem (using the algorithm from [1]):
    
        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0
        
    Parameters
    ----------
    s: (n,) numpy array,
       n-dimensional vector to project
       
    alpha: int, optional, default: 1,
           radius of the simplex
           
    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex
       
    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.
    
    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf

    '''
    n, = s.shape
    # check if we are already on the simplex
    if np.sum(s) == alpha and np.alltrue(s >= 0):
        # best projection: itself!
        return s
    
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(s)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - alpha))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = float(cssv[rho] - alpha) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (s - theta).clip(min=0)
    return w

def nuclear_projection(X, r = 1.0):
    '''
    Projection onto nuclear norm ball.
    '''
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    sproj = simplex_projection(s, alpha = r)
    return U @ np.diag(sproj) @ Vt
