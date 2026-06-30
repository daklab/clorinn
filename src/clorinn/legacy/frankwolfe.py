import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.extmath import randomized_svd

def nuclear_norm(X):
    '''
    Nuclear norm of input matrix
    '''
    return np.sum(np.linalg.svd(X)[1])

def f_objective(X, Y, W = None, mask = None):
    '''
    Objective function
    Y is observed, X is estimated
    W is the weight of each observation.
    '''
    Xmask = X if mask is None else X * mask
    Wmask = W if mask is None else W * mask

    # The * operator can be used as a shorthand for np.multiply on ndarrays.
    if Wmask is None:
        f_obj = 0.5 * np.linalg.norm(Y - Xmask, 'fro')**2
    else:
        f_obj = 0.5 * np.linalg.norm(Wmask * (Y - Xmask), 'fro')**2
    return f_obj


def f_gradient(X, Y, W = None, mask = None):
    '''
    Gradient of the objective function.
    '''
    Xmask = X if mask is None else X * mask
    Wmask = W if mask is None else W * mask

    if Wmask is None:
        f_grad = Xmask - Y
    else:
        f_grad = np.square(Wmask) * (Xmask - Y)

    return f_grad


def linopt_oracle(grad, r = 1.0, max_iter = 10):
    '''
    Linear optimization oracle,
    where the feasible region is a nuclear norm ball for some r
    '''
    U1, V1_T = singular_vectors_power_method(grad, max_iter = max_iter)
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


def do_step_size(dg, D, W = None):
    if W is None:
        denom = np.linalg.norm(D, 'fro')**2
    else:
        denom = np.linalg.norm(W * D, 'fro')**2
    step_size = dg / denom
    step_size = min(step_size, 1.0)
    if step_size < 0:
        print ("Warning: Step Size is less than 0")
        step_size = 1.0
    return step_size


def frank_wolfe_minimize_step(X, Y, r, istep, W = None, mask = None):

    # 1. Gradient for X_(t-1)
    G = f_gradient(X, Y, W = W, mask = mask)
    # 2. Linear optimization subproblem
    power_iter = 10 + int(istep / 50)
    S = linopt_oracle(G, r, max_iter = power_iter)
    # 3. Define D
    D = X - S
    # 4. Duality gap
    dg = np.trace(D.T @ G)
    # 5. Step size
    step = do_step_size(dg, D, W = W)
    # 6. Update
    Xnew = X - step * D
    return Xnew, G, dg, step


def frank_wolfe_minimize(Y, r, X0 = None,
                         weight = None,
                         mask = None,
                         max_iter = 1000, tol = 1e-8,
                         return_all = True,
                         debug = False, debug_step = 10):

    # Step 0
    old_X = np.zeros_like(Y) if X0 is None else X0.copy()
    dg = np.inf

    if return_all:
        dg_list = [dg]
        fx_list = [f_objective(old_X, Y, W = weight, mask = mask)]
        st_list = [1]

    # Steps 1, ..., max_iter
    for istep in range(max_iter):
        X, G, dg, step = \
            frank_wolfe_minimize_step(old_X, Y, r, istep, W = weight, mask = mask)
        f_obj = f_objective(X, Y, W = weight, mask = mask)

        if return_all:
            dg_list.append(dg)
            fx_list.append(f_obj)
            st_list.append(step)

        if debug:
            if (istep % debug_step == 0):
                print (f"Iteration {istep}. Step size {step:.3f}. Duality Gap {dg:g}")
        if np.abs(dg) <= tol:
            break

        old_X = X.copy()

    if return_all:
        return X, dg_list, fx_list, st_list
    else:
        return X
