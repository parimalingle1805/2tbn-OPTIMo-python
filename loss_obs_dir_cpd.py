import numpy as np

def loss_obs_dir_cpd(params, dx, obs_dir, enable_param_grad=None):
    """
    Computes the negative log-likelihood loss and gradients for the directional observation CPD.

    Args:
        params (np.array): Model parameters [kd, od, bc].
        dx (np.array): The change in state (x_curr - x_past).
        obs_dir (np.array): The observed directions.
        enable_param_grad (np.array, optional): Boolean array to enable/disable gradient
            calculation for specific parameters. Defaults to None (all enabled).

    Returns:
        tuple: The negative log-likelihood loss and gradient.
    """
    kd, od, bc = params

    # Create logical indices for the different observation cases
    idx_plus = obs_dir > 0
    idx_minus = obs_dir < 0
    idx_same = obs_dir == 0

    # Calculate sigmoid values
    sigmoid_plus = 1.0 / (1.0 + np.exp(-kd * (dx - od)))
    sigmoid_minus = 1.0 / (1.0 + np.exp(-kd * (-dx - od)))

    # Calculate probabilities for each observation case
    probs = np.zeros_like(dx)
    probs[idx_plus] = sigmoid_plus[idx_plus] * (1 - 3 * bc) + bc
    probs[idx_minus] = sigmoid_minus[idx_minus] * (1 - 3 * bc) + bc
    probs[idx_same] = 1 - 2 * bc - (1 - 3 * bc) * (
        sigmoid_plus[idx_same] + sigmoid_minus[idx_same])
    probs = np.clip(probs, 1e-12, 1)  # Ensure probs are not zero (avoid log(0))

    # Calculate the negative log-likelihood loss
    loss = -np.sum(np.log(probs))

    # Calculate gradient if requested
    if enable_param_grad is not None:
        grad_kd_probs = np.zeros_like(dx)
        grad_kd_probs[idx_plus] = (1 - 3 * bc) * sigmoid_plus[idx_plus] * (
            1 - sigmoid_plus[idx_plus]) * (dx[idx_plus] - od)
        grad_kd_probs[idx_minus] = (1 - 3 * bc) * sigmoid_minus[idx_minus] * (
            1 - sigmoid_minus[idx_minus]) * (-dx[idx_minus] - od)
        grad_kd_probs[idx_same] = (3 * bc - 1) * (
            sigmoid_plus[idx_same] * (1 - sigmoid_plus[idx_same]) *
            (dx[idx_same] - od) + sigmoid_minus[idx_same] *
            (1 - sigmoid_minus[idx_same]) * (-dx[idx_same] - od))

        grad_od_probs = np.zeros_like(dx)
        grad_od_probs[idx_plus] = (1 - 3 * bc) * sigmoid_plus[idx_plus] * (
            1 - sigmoid_plus[idx_plus]) * -kd
        grad_od_probs[idx_minus] = (1 - 3 * bc) * sigmoid_minus[idx_minus] * (
            1 - sigmoid_minus[idx_minus]) * -kd
        grad_od_probs[idx_same] = (1 - 3 * bc) * kd * (
            sigmoid_plus[idx_same] * (1 - sigmoid_plus[idx_same]) +
            sigmoid_minus[idx_same] * (1 - sigmoid_minus[idx_same]))

        grad_bc_probs = np.zeros_like(dx)
        grad_bc_probs[idx_plus] = 1 - 3 * sigmoid_plus[idx_plus]
        grad_bc_probs[idx_minus] = 1 - 3 * sigmoid_minus[idx_minus]
        grad_bc_probs[idx_same] = -2 + 3 * (
            sigmoid_plus[idx_same] + sigmoid_minus[idx_same])

        gradloss = np.array([
            -np.sum(grad_kd_probs / probs),
            -np.sum(grad_od_probs / probs),
            -np.sum(grad_bc_probs / probs)
        ])

        if enable_param_grad is not None:
            gradloss = gradloss[enable_param_grad]
        return loss, gradloss
    else:
        return loss