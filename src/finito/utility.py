from typing import Iterable
import numpy as np


# TODO: Optimize cycle. Currently O(M*N), can be O(N), where M is len(x), N is len(data)
def EmpiricalCDF(data: Iterable, x: Iterable | None = None):
    """Estimate cumulative distribution function.

    Args:
        data (Iterable): Samples of the random variable
        x (Iterable | None): Specific points to calculate distribution in

    Returns:
        _type_: _description_
    """
    # Use all possible information for estimation
    if x is None:
        x = data
    res = []
    nSamples = len(data)
    for xi in x:
        res.append(np.sum(data <= xi) / nSamples)
    res.sort()
    return np.array(res)
