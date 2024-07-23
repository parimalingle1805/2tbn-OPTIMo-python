import numpy as np
from scipy.stats import norm

def boundedLinearGaussianCPD(x, mu, sigma, num_bins):
    """
    Computes the bounded linear Gaussian CPD.

    This function calculates the probability of x given a normal
    distribution with mean mu and standard deviation sigma,
    considering the boundaries [0, 1].

    Args:
        x (np.array): The target values (bounded between 0 and 1).
        mu (float or np.array): The mean(s) of the Gaussian distribution.
                               If a float, it's used for all x values.
                               If an array, it should have the same length as x.
        sigma (float): The standard deviation of the Gaussian distribution.
        num_bins (int): Number of bins for histogram approximation.

    Returns:
        np.array: The normalized probability mass function (PMF).
    """

    # Handle scalar mu input (broadcast to match x)
    if np.isscalar(mu):
        mu = np.full_like(x, mu)

    mu = np.clip(mu, 0, 1)  # Bound mean within [0, 1]

    # Calculate unnormalized probabilities using normpdf
    probs_unnorm = norm.pdf(x, mu, sigma) / num_bins

    # Fold left and right tails of the Gaussian CDF onto boundaries
    probs_unnorm[0] += (norm.cdf(0, mu[0], sigma) - 0)  # For x = 0
    probs_unnorm[-1] += (1 - norm.cdf(1, mu[-1], sigma))  # For x = 1

    # Normalize the PMF
    norm_const = np.sum(probs_unnorm)
    if norm_const == 0:
        norm_const = 1  # Avoid division by zero
    probs = probs_unnorm / norm_const

    return probs