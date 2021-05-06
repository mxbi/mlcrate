import numpy as np
from collections import defaultdict

def rank_average(*args, **opts):
    """Return an array containing all the unique elements of the input arrays.
    The elements are sorted based on the average of their index (a rank average).
    This can be used for ensembling where you have an ordered list of elements (eg for MAP@5)

    The arrays can have different sizes - if an element does not appear in an array it is
    assumed to have index len(array)+1.

    Keyword Arguments:
    base_rank (optional): Assumed rank/index of elements that don't appear in the array
    weights (optional): Array of weights for a weighted rank average of the inputs
    """
    arrs = args
    weights = opts.get('weights', None)
    base_rank = opts.get('base_rank', None)

    # assert len(set([len(arr) for arr in arrs])) == 1,  "All input arrays must be the same length"
    assert all([arr.ndim == 1 for arr in arrs]), "All input arrays must be one-dimensional"

    if weights is None:
        weights = [1 for arr in arrs]
    else:
        assert len(weights) == len(arrs), "len(weights) must be equal to the number of input arrays"

    if base_rank is None:
        base_ranks = [len(arr) for arr in arrs]
    else:
        base_ranks = [base_rank for _ in range(len(arrs))]

    ranks = defaultdict(lambda: sum(base_ranks))

    for arr, base_rank, w in zip(arrs, base_ranks, weights):
        for i, element in enumerate(arr):
            ranks[element] -= (base_rank - i) * w

    elems, scores = zip(*ranks.items())
    result = np.array(elems)[np.argsort(scores)]

    return result

def prob_rank_average(*arrs, **opts):
    """Return a rank average across input arrays of probabilities (one array per input model)
    These are ensembled using a weighted ranking average (defaults to equal weights to each model)
    The output array of probabilities is linear in the range [0, 1], since we only care about how each
    model ranks the provided inputs.
    
    For example:
    
    prob_rank_average([0.1, 0.2, 0.5, 0.8, 0], [0.2, 0.1, 0.6, 1.5, 1000000], weights=[0.4, 0.6])
    => array([0.  , 0.25, 1.  , 0.5 , 0.75], dtype=float32)
    
    Keyword Arguments:
    weights (optional): Array of weights for a weighted rank average of the inputs
    """
    
    weights = opts.get('weights', None)
    
    arrs = [np.array(a) for a in arrs]
    
    len0 = len(arrs[0])
    for arr in arrs:
        assert arr.ndim == 1, "All input arrays must be 1-D"
        assert len(arr == len0), "All input arrays must have the same length"
    
    ranked_arrs = []
    for arr in arrs:
        ranked_arrs.append(np.argsort(arr))

    reranked = rank_average(*ranked_arrs, weights=weights).astype(np.float32)
    reranked /= len(reranked) - 1
    
    return reranked
