import numpy as np

def get(original, recovered, mask = None, method = 'rmse'):
    """
    Calculate error statistic between original and recovered matrices.

    mask : np.array [size(n,p); dtype boolean], default=None
        Whether any entry should be masked in original data.
        If True, that is excluded from calculation.

    method : string, default='rmse'
        Which statistic should be used.
    """
    if mask is None:
        mask = np.isnan(original)

    if method == 'mse':
        return get_mse(original, recovered, ~mask)

    elif method == 'rmse':
        return get_rmse(original, recovered, ~mask)

    elif method == 'psnr':
        return get_psnr(original, recovered, ~mask)

    return


def get_mse(original, recovered, mask):
    n = np.sum(mask)
    mse = np.nansum(np.square((original - recovered) * mask)) / n
    return mse


def get_rmse(original, recovered, mask):
    mse = get_mse(original, recovered, mask)
    return np.sqrt(mse)


def get_psnr(original, recovered, mask):
    omax = np.max(original[mask])
    omin = np.min(original[mask])
    maxsig2 = np.square(omax - omin)
    mse = get_mse(original, recovered, mask)
    res = 10 * np.log10(maxsig2 / mse)
    return res
