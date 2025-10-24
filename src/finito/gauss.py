import numpy as np
from scipy import stats


class GaussianMixture:
    def _CheckGMParameters(a, b, p) -> None:
        assert a is not None and b is not None, "Means and variances must be provided"
        assert a.shape == b.shape == p.shape, "Parameters must have the same length"
        # Variances
        assert np.all(b > 0), "Variances must be positive"
        # Probabilities
        assert np.isclose(np.sum(p), 1.0), "Weights must sum to 1"
        assert np.all(p >= 0), "Weights must be non-negative"

    def __init__(self, a, b, p=None):
        """Initialize a Gaussian Mixture Model (GMM) with given parameters.

        Parameters:
            a (list[float]): means of the components
            b (list[float]): variances of the components
            p (list[float] | None): weights of the components (should sum to 1).
                If None, random weights will be generated.
        """
        a = np.asarray(a).reshape(-1, 1)  # (K, 1)
        b = np.asarray(b).reshape(-1, 1)  # (K, 1)
        if p is None:
            # Generate random weights that sum to 1
            p = np.random.dirichlet(np.ones(b.shape[0]))
        p = np.asarray(p).reshape(-1, 1)  # (K, 1)
        __class__._CheckGMParameters(a, b, p)
        self._means = a
        self._variances = b
        self._weights = p
        self.n_components = a.shape[0]
        self._gaussians = stats.norm(loc=self._means, scale=np.sqrt(self._variances))

    def cdf(self, x):
        """Compute the cumulative distribution function for the GMM at points x."""
        return np.sum(self._weights * self._gaussians.cdf(x), axis=0)

    def pdf(self, x):
        """Compute the probability density function for the GMM at points x."""
        return np.sum(self._weights * self._gaussians.pdf(x), axis=0)

    def sample(self, size, random_state=None):
        """Generate random samples from the GMM."""
        # Choose what gaussian to use for generation based on components weights
        component_indices = np.random.choice(
            self.n_components, size=size, p=self._weights.flatten()
        )
        # Generate samples from the chosen components
        samples = np.array(
            [
                self._gaussians.rvs(
                    size=(self.n_components, 1), random_state=random_state
                )[idx][0]
                for idx in component_indices
            ]
        )
        return samples

    def get_params(self):
        """Return the parameters of the GMM as a tuple (means, variances, weights)."""
        return self._means.flatten(), self._variances.flatten(), self._weights.flatten()
