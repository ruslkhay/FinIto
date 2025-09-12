"""Simulate the Ito process"""

import numpy as np
import scipy.stats as stats
from tqdm import tqdm


def generateGeneralWiener(
    a,
    b,
    dt: np.timedelta64,
    T: np.timedelta64,
    seed=42,
    x0=0,
    disable_tqdm=False,
):
    """
    Generate a random Wiener process with given parameters.

    Parameters:
        a (float): Dynamic component of the Wiener process.
        b (float): Diffusional component of the Wiener process.
        dt (np.timedelta64): The time step of the process.
        T (np.timedelta64): Duration of the process.
        seed (int): The seed used for the random number generator.
        x0 (float): The initial value of the process.
        disable_tqdm (bool): If True, disable the printing of the progress bar.

    Returns:
        A 1D array (time series).
    """
    np.random.seed(seed)
    X = np.full(T // dt, x0, dtype=float)
    delta_t = dt / np.timedelta64(1, "s")
    for i in tqdm(
        range(1, X.shape[0]), "Generating Wiener process", disable=disable_tqdm
    ):
        randomVal = stats.norm.rvs(0, 1)
        X[i] = a * delta_t + b * randomVal * np.sqrt(delta_t) + X[i - 1]
    return X
