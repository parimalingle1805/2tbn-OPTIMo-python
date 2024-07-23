import numpy as np

def sigmoidCPD(xprime, x, d, kd, od, bd):
    """
    Computes the probability of observing direction d given state change.

    This function calculates the conditional probability distribution
    (CPD) for observing a direction of motion/change (d) given the
    current state (x) and the previous state (xprime). It uses a sigmoid
    function parameterized by kd, od, and bd to model the probability.

    Args:
        xprime (np.array): The previous state values.
        x (np.array): The current state values.
        d (int): The observed direction (-1, 0, or 1).
        kd (float): Sigmoid gain (controls steepness of the sigmoid slope).
        od (float): Center offset (defines point where prob = 0.5).
        bd (float): Bias probability (probability of erroneous signaling).

    Returns:
        np.array: The probabilities of observing the given direction
                  for each pair of xprime and x.
    """
    dx = x - xprime
    scale = 1 - 3 * bd

    if d > 0:
        probs = bd + scale / (1 + np.exp(-kd * (+dx - od)))
    elif d < 0:
        probs = bd + scale / (1 + np.exp(-kd * (-dx - od)))
    else:  # d == 0
        probs = 1 - 2 * bd - scale * (1 / (1 + np.exp(-kd * (+dx - od))) +
                                       1 / (1 + np.exp(-kd * (-dx - od))))
        probs = np.clip(probs, 0, 1)  # Clip probabilities to [0, 1]

    return probs