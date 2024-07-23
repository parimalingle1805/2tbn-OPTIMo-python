import numpy as np

def loss_norm_cpd(params, x, U, enable_param_grad=None):
    """
    Computes the negative log-likelihood loss and gradients for a linear Gaussian CPD.

    Args:
        params (np.array): Model parameters [w_1, ..., w_N, sigma].
        x (np.array): Observed data points.
        U (np.array): Design matrix (features/inputs).
        enable_param_grad (np.array, optional): Boolean array to enable/disable
            gradient calculation for specific parameters. Defaults to None (all enabled).

    Returns:
        tuple: The negative log-likelihood loss and gradient.
    """
    sigma = params[-1]
    w = params[:-1]
    mu = U @ w 
    x_norm = (x - mu) / sigma

    # Calculate the negative log-likelihood loss
    logprobs = -0.5 * np.log(2 * np.pi) - np.log(sigma) - 0.5 * x_norm**2
    loss = -np.sum(logprobs)

    # Calculate the gradient if requested
    if enable_param_grad is not None:
        grad_w = (x_norm[:, None] * U).sum(axis=0)
        grad_sigma = (-1 / sigma * (1 - x_norm**2)).sum()
        gradloss = np.concatenate([-grad_w, [-grad_sigma]])

        if enable_param_grad is not None:
            gradloss = gradloss[enable_param_grad]
        return loss, gradloss
    else:
        return loss