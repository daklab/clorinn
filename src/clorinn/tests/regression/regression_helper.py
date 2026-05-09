import numpy as np

def assert_allclose_print_worst_entry(testcase, actual, expected, *, rtol, atol, name="value", equal_nan = False):
    """
    np.testing.assert_allclose with additional worst-entry diagnostics.

    Suppresses exception chaining so unittest does not print
    'During handling of the above exception...'.
    """
    actual = np.asarray(actual)
    expected = np.asarray(expected)

    try:
        np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol, equal_nan = equal_nan)
    except AssertionError as err:
        # If this is a shape/broadcasting failure, NumPy's message is already best.
        if actual.shape != expected.shape:
            raise testcase.failureException(str(err)) from None

        diff = np.abs(actual - expected)
        tol = atol + rtol * np.abs(expected)
        d = diff - tol

        # Ignore non-finite entries when locating the worst finite delta.
        delta = np.where(np.isfinite(d), d, -np.inf)
        if delta.size == 0:
            raise testcase.failureException(str(err)) from None

        idx = np.unravel_index(np.argmax(delta), delta.shape)

        extra_msg = (
            f"\n\nWorst offending entry at index {idx}:\n"
            f"  actual   = {actual[idx]}\n"
            f"  expected = {expected[idx]}\n"
            f"  abs diff = {diff[idx]}\n"
            f"  allowed  = {tol[idx]}\n"
            f"  rtol     = {rtol}\n"
            f"  atol     = {atol}"
        )
        raise testcase.failureException(str(err) + extra_msg) from None
