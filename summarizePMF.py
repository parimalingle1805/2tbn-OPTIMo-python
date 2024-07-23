import numpy as np


def summarizePMF(pmfs, states, bin_widths, alpha=0.05):
    """
    Computes summary statistics for probability mass functions (PMFs).

    This function calculates the expected value, mode, lower confidence
    interval, and upper confidence interval for each PMF provided in
    the input matrix `pmfs`.

    Args:
        pmfs (np.array): A 2D array (TxN) where each row represents a
            normalized PMF with N bins.
        states (np.array): A 1D array (1xN) containing the center values
            for each state bin.
        bin_widths (np.array): A 1D array (1xN) containing the width of
            each state bin.
        alpha (float, optional): The significance level for the
            confidence interval calculation. Defaults to 0.05.

    Returns:
        np.array: A 2D array (Tx4) containing the expected value, mode,
                  lower confidence interval, and upper confidence interval
                  for each input PMF.
    """
    num_obs = pmfs.shape[0]
    stats = np.zeros((num_obs, 4))

    # Calculate expected value
    stats[:, 0] = np.sum(pmfs * states, axis=1)

    # Calculate mode
    mode_idxes = np.argmax(pmfs, axis=1)
    stats[:, 1] = states[mode_idxes]

    # Calculate cumulative mass function (CMF)
    cmfs = np.cumsum(pmfs, axis=1)

    # Calculate confidence interval bounds
    if alpha > 0.5:
        alpha = 1 - alpha
    conf_int_low = alpha / 2
    conf_int_high = 1 - conf_int_low

    for i in range(num_obs):
        # Lower confidence interval
        idx_upper_ci_low = np.argmax(cmfs[i, :] >= conf_int_low)
        prob_upper_ci_low = cmfs[i, idx_upper_ci_low]
        prob_lower_ci_low = cmfs[i, idx_upper_ci_low - 1] if idx_upper_ci_low > 0 else 0
        stats[i, 2] = (
            states[idx_upper_ci_low] - bin_widths[idx_upper_ci_low] / 2 +
            bin_widths[idx_upper_ci_low] *
            (conf_int_low - prob_lower_ci_low) /
            (prob_upper_ci_low - prob_lower_ci_low))

        # Upper confidence interval
        idx_upper_ci_high = np.argmax(cmfs[i, :] >= conf_int_high)
        prob_upper_ci_high = cmfs[i, idx_upper_ci_high]
        prob_lower_ci_high = cmfs[i, idx_upper_ci_high - 1] if idx_upper_ci_high > 0 else 0
        stats[i, 3] = (
            states[idx_upper_ci_high] - bin_widths[idx_upper_ci_high] / 2 +
            bin_widths[idx_upper_ci_high] *
            (conf_int_high - prob_lower_ci_high) /
            (prob_upper_ci_high - prob_lower_ci_high))

    return stats