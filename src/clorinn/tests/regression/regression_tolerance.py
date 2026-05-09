def _tol(model, solver, quantity, is_masked, is_strict):
    """
    Return (rtol, atol) for regression tests.

    In strict mode, use near-exact trajectory comparison.
    In portable mode, use tolerances calibrated to observed
    BLAS/LAPACK/NumPy/SciPy drift across machines.
    """
    if is_strict:
        return 1e-12, 1e-12

    if solver == "pgd":
        if quantity in {"X", "M"}:
            return 1e-8, 1e-10
        if quantity == "loss":
            return 1e-8, 1e-10
        return 1e-8, 1e-10

    #fixture = model
    #if is_masked:
    #    fixture = f"{model}-mask"

    if solver != "fw":
        return 1e-8, 1e-10

    # ------------------------------------------------------------------
    # Frank-Wolfe: NNM
    # ------------------------------------------------------------------
    if model == "nnm":
        if is_masked:
            # Observed:
            # X max abs ~3.9e-11, max rel ~3.1e-9
            # dg max abs ~1.4e-10, max rel ~2.8e-7
            # step max abs ~2.8e-9, max rel ~9.1e-9
            if quantity == "X":
                return 1e-8, 1e-10
            if quantity == "gap":
                return 1e-6, 1e-10
            if quantity == "step":
                return 1e-8, 1e-9
            if quantity == "loss":
                return 1e-8, 1e-8
            return 1e-8, 1e-10

        # Fully observed NNM did not appear among the failures.
        if quantity == "X":
            return 1e-10, 1e-12
        if quantity in {"loss", "gap", "step"}:
            return 1e-10, 1e-12
        return 1e-10, 1e-12

    # ------------------------------------------------------------------
    # Frank-Wolfe: NNM-Sparse
    # ------------------------------------------------------------------
    if model == "nnm-sparse":
        if is_masked:
            # Observed:
            # X max abs ~9.6e-9, max rel ~7.7e-7
            # M max abs ~3.1e-9
            # dg max abs ~4.3e-8, max rel ~4.4e-5
            # step max abs ~6.4e-7, max rel ~1.6e-6
            if quantity == "X":
                return 1e-6, 1e-8
            if quantity == "M":
                return 1e-8, 1e-8
            if quantity == "gap":
                return 1e-4, 1e-7
            if quantity == "step":
                return 1e-5, 1e-7
            if quantity == "loss":
                return 1e-8, 1e-8
            return 1e-6, 1e-8

        # Observed:
        # X max abs ~2.8e-11, max rel ~5.7e-10
        # M max abs ~1.2e-11
        # dg max abs ~7.2e-11, max rel ~2.2e-8
        # step max abs ~1.5e-8, max rel ~2.6e-8
        if quantity == "X":
            return 1e-8, 1e-10
        if quantity == "M":
            return 1e-8, 1e-10
        if quantity == "gap":
            return 1e-7, 1e-10
        if quantity == "step":
            return 1e-7, 1e-8
        if quantity == "loss":
            return 1e-8, 1e-8
        return 1e-8, 1e-10

    # ------------------------------------------------------------------
    # Frank-Wolfe: NNM-Corr
    # ------------------------------------------------------------------
    if model == "nnm-corr":
        # Important:
        # Full histories and exact n_iter are not portable here.
        # The logs show different stopping lengths:
        #   fully observed: 304 vs 325
        #   masked:         394 vs 366
        #
        # These tolerances are only for final values, not full histories.
        #
        # Observed X:
        #   fully observed max abs ~7.4e-4
        #   masked max abs ~1.35e-3
        # Max relative error is large because some entries are near zero.
        if quantity == "X":
            return 1e-3, 2e-3
        if quantity == "M":
            return 1e-3, 2e-3
        if quantity == "loss":
            return 1e-5, 1e-4
        if quantity == "gap":
            return 1e-4, 1e-6
        if quantity == "step":
            return 1e-4, 1e-6
        return 1e-4, 1e-6

    return 1e-8, 1e-10
