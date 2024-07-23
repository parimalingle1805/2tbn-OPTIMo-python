import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize, Bounds
import matplotlib.pyplot as plt
from boundedLinearGaussianCPD import boundedLinearGaussianCPD
from sigmoidCPD import sigmoidCPD
from loss_norm_cpd import loss_norm_cpd
from loss_obs_dir_cpd import loss_obs_dir_cpd
from summarizePMF import summarizePMF


def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))

# def boundedLinearGaussianCPD(x, mu, sigma, num_bins):
#     """
#     Compute the bounded linear Gaussian CPD.
#     """
#     probs = norm.pdf(x, loc=mu, scale=sigma)
#     probs[0] += norm.cdf(0, loc=mu, scale=sigma)
#     probs[-1] += 1 - norm.cdf(1, loc=mu, scale=sigma)
#     return probs

# def sigmoidCPD(x_prev, x_curr, dir_sign, kd, od, bd):
#     """
#     Sigmoid CPD function for direction observation.
#     """
#     return bd + (1 - 3 * bd) * sigmoid(kd * (dir_sign * (x_curr - x_prev) - od))

# def loss_norm_cpd(x_diff, At, params):
#     """Loss function for propagate() parameters (negative log-likelihood)."""
#     wx_bias, wx_spd, sx = params
#     mu = At @ np.array([wx_bias, wx_spd])
#     return -np.sum(norm.logpdf(x_diff, loc=mu, scale=sx))

# def loss_obs_dir_cpd(dx_instances, obs_dir_instances, params):
#     """Loss function for observe_dir() parameters (negative log-likelihood)."""
#     kd, od, bd = params
#     probs = sigmoidCPD(0, dx_instances, obs_dir_instances, kd, od, bd) 
#     return -np.sum(np.log(probs)) 

# def summarizePMF(pmf, x_vec, bin_sizes, alpha=None):
#     """Summarize PMF to get mean, std, and confidence interval."""
#     if alpha is None:
#         alpha = 0.05  # Default 95% confidence interval
#     cdf = np.cumsum(pmf, axis=0)
#     mean_pmf = np.sum(pmf * x_vec * bin_sizes, axis=0)
#     var_pmf = np.sum(pmf * (x_vec ** 2) * bin_sizes, axis=0) - mean_pmf ** 2
#     std_pmf = np.sqrt(var_pmf)
#     ci_lower = np.interp(alpha / 2, cdf, x_vec)
#     ci_upper = np.interp(1 - alpha / 2, cdf, x_vec)
#     return np.array([mean_pmf, std_pmf, ci_lower, ci_upper]).T

class RobotModel:
    optimize_prop_cpd_using_fmincon = False
    prop_optim_options = {'method': 'trust-constr'}
    prop_optim_params_lb = np.array([-1, -1, 0])
    prop_optim_params_ub = np.array([1, 1, np.inf])
    
    obs_dir_optim_options = {'method': 'trust-constr'}
    obs_dir_optim_params_lb = np.array([1, 0, 0])
    obs_dir_optim_params_ub = np.array([1e4, 1, 1/3])

    eps_wx_bias = 1e-8
    eps_wx_spd = 1e-8
    eps_sx = 1e-10
    eps_kd = 1e-9
    eps_od = 1e-9
    eps_bd = 1e-9
    eps_sz = 1e-10

    def __init__(self, params):
        self.params = params

    def updateParams(self, new_params):
        self.params = new_params

    def propagate(self, cache, data_curr, data_past=None):
        """Propagates belief on latent state to next time step."""
        if 'x' not in data_curr or data_curr['x'] is None:
            return np.eye(cache['num_bins'])

        mu = cache['x_past'] + self.params['wx_bias'] + self.params['wx_spd'] * data_curr['u']
        probs = boundedLinearGaussianCPD(cache['x_curr'], mu, self.params['sx'], cache['num_bins'])
        return probs

    def observe(self, cache, data_curr, data_past=None):
        """Applies evidence to update belief on latent state."""
        probs = np.ones_like(cache['x_curr'])

        if 'd' in data_curr and data_curr['d'] is not None:
            probs *= sigmoidCPD(
                cache['x_past'], cache['x_curr'], data_curr['d'],
                self.params['kd'], self.params['od'], self.params['bd']
            )

        if 'z' in data_curr and data_curr['z'] is not None:
            probs *= boundedLinearGaussianCPD(
                cache['x_curr'], data_curr['z'], self.params['sz'], len(cache['x_curr'])
            )
        return probs

    @staticmethod
    def plotHistogram(data_all, cache, map_states, filtered_probs=None,
                      smoothed_probs=None, fig_id=0, alpha=None):
        """Visualizes observed/controls data set and PGM inference outputs."""
        show_map = map_states is not None
        show_filter = filtered_probs is not None
        show_smooth = smoothed_probs is not None

        if show_filter:
            filtered_stats = summarizePMF(filtered_probs / cache['num_bins'],
                                        cache['x_vec'],
                                        1.0 / cache['num_bins'] * np.ones(cache['num_bins']),
                                        alpha)
        if show_smooth:
            smoothed_stats = summarizePMF(smoothed_probs / cache['num_bins'],
                                        cache['x_vec'],
                                        1.0 / cache['num_bins'] * np.ones(cache['num_bins']),
                                        alpha)

        time_indices = np.arange(1, len(data_all) + 1)
        time_indices_w_zero = np.arange(0, len(data_all) + 1)

        plt.figure(fig_id)
        plt.clf()

        plt.subplot(3, 1, 1)
        #plt.hold(True)

        if show_filter:
            plt.errorbar(time_indices_w_zero, filtered_stats[:, 0],
                         yerr=[
    np.clip(filtered_stats[:, 0] - filtered_stats[:, 2], 0, None),  # Clip lower errors
    np.clip(filtered_stats[:, 3] - filtered_stats[:, 0], 0, None)   # Clip upper errors
],
                         fmt='-g', label='Exp(filter)')

        if show_smooth:
            plt.errorbar(time_indices_w_zero, smoothed_stats[:, 0],
                         yerr=[
    np.clip(smoothed_stats[:, 0] - smoothed_stats[:, 2], 0, None),  # Clip lower errors
    np.clip(smoothed_stats[:, 3] - smoothed_stats[:, 0], 0, None)   # Clip upper errors
],
                         fmt='-b', label='Exp(smooth)')

        if show_map:
            plt.plot(time_indices_w_zero, map_states, '-k', linewidth=2, label='MAP')

        valid_z_idx = np.array([d['z'] is not None for d in data_all])
        if np.any(valid_z_idx):
            plt.plot(time_indices[valid_z_idx],
                     np.array([d['z'] for d in data_all])[valid_z_idx],
                     'or', linewidth=2, label='Observed Pose')

        plt.xlabel('time (sec)     MAP (black -) | Exp(smooth) (blue -) | Exp(filter) (green -) | Observed Pose (red o)')
        plt.ylabel('Latent state (robot pose)')
        plt.xlim([0, time_indices[-1]])
        plt.ylim([0, 1])
        plt.legend()

        plt.subplot(3, 1, 2)
        #plt.hold(True)

        map_diff = []
        diff_filter = []
        diff_smooth = []

        if show_map:
            map_diff = map_states[1:] - map_states[:-1]
        if show_filter:
            diff_filter = filtered_stats[1:, 0] - filtered_stats[:-1, 0]
            plt.plot(time_indices, diff_filter, '-vg', label='Diff in Exp(filter)')
        if show_smooth:
            diff_smooth = smoothed_stats[1:, 0] - smoothed_stats[:-1, 0]
            plt.plot(time_indices, diff_smooth, '--^b', label='Diff in Exp(smooth)')
        if show_map:
            plt.plot(time_indices, map_diff, ':xk', label='Diff in MAP')

        max_abs_diff = np.max(np.abs(np.concatenate(([0], map_diff, diff_filter, diff_smooth))))
        if max_abs_diff == 0:
            max_abs_diff = 1

        valid_d_idx = np.array([d['d'] is not None for d in data_all])
        if np.any(valid_d_idx):
            plt.stem(time_indices[valid_d_idx],
                     max_abs_diff * np.array([d['d'] for d in data_all])[valid_d_idx],
                     'ro', linefmt='r-', label='Observed Dir')

        plt.xlabel('time (sec)     Diff in: MAP (black -) | Exp(smooth) (blue -) | Exp(filter) (green -) | Observed Dir (red o)')
        plt.ylabel('Change in latent state (robot pose)')
        plt.xlim([0, time_indices[-1]])
        plt.ylim([-max_abs_diff, max_abs_diff])
        plt.legend()

        active_u_idx = np.array([d['u'] is not None for d in data_all])
        plt.subplot(3, 1, 3)
        #plt.hold(True)
        plt.stem(time_indices[active_u_idx],
                 np.array([d['u'] for d in data_all])[active_u_idx],
                 '-xb', label='Control Input')
        plt.xlabel('time (sec)     Control Input (blue x)')
        plt.xlim([0, time_indices[-1]])
        plt.ylim([-1, 1])
        plt.legend()

        plt.tight_layout()
        plt.show()

    @staticmethod
    def compareParams(params_a, params_b, epsilon=0):
        """Assesses whether 2 sets of parameters are sufficiently similar."""
        return (
            (abs(params_a['wx_bias'] - params_b['wx_bias']) < epsilon * RobotModel.eps_wx_bias) and
            (abs(params_a['wx_spd'] - params_b['wx_spd']) < epsilon * RobotModel.eps_wx_spd) and
            (abs(params_a['sx'] - params_b['sx']) < epsilon * RobotModel.eps_sx) and
            (abs(params_a['kd'] - params_b['kd']) < epsilon * RobotModel.eps_kd) and
            (abs(params_a['od'] - params_b['od']) < epsilon * RobotModel.eps_od) and
            (abs(params_a['bd'] - params_b['bd']) < epsilon * RobotModel.eps_bd) and
            (abs(params_a['sz'] - params_b['sz']) < epsilon * RobotModel.eps_sz)
        )

    def optimizeParams(self, data_all, x_states, params_old):
        """Optimizes model parameters given data and estimated states."""
        if x_states.ndim > 1:
            raise NotImplementedError("Soft EM has not been implemented")

        params_new = params_old.copy()
        num_time_steps = len(data_all)
        if num_time_steps == 0:
            raise ValueError("data_all has 0 samples")

        x_past = x_states[:-1]
        x_curr = x_states[1:]
        x_diff = x_curr - x_past

        # Optimize wx_bias, wx_spd (,sx)
        u_curr = np.array([d['u'] if d['u'] is not None else np.nan for d in data_all])
        valid_u_idx = np.logical_not(np.isnan(u_curr))
        At = np.column_stack((np.ones(np.sum(valid_u_idx)), u_curr[valid_u_idx]))

        if np.linalg.cond(At) > 10000:
            print("Warning: Cannot optimize params for propagate() since design matrix A is near-singular")
        else:
            if RobotModel.optimize_prop_cpd_using_fmincon:
                # Optimize using constrained minimization
                result = minimize(
                    loss_norm_cpd,
                    [params_old['wx_bias'], params_old['wx_spd'], params_old['sx']],
                    args=(x_diff[valid_u_idx], At),
                    bounds=Bounds(RobotModel.prop_optim_params_lb, RobotModel.prop_optim_params_ub),
                    options=RobotModel.prop_optim_options,
                )
                if not result.success:
                    print("Warning: Bounded optimization of propagate() params failed")
                else:
                    params_new['wx_bias'], params_new['wx_spd'], params_new['sx'] = result.x
            else:
                # Optimize using linear least squares
                prop_optim_newparams = np.linalg.lstsq(At, x_diff[valid_u_idx], rcond=None)[0]
                params_new['wx_bias'] = prop_optim_newparams[0]
                params_new['wx_spd'] = prop_optim_newparams[1]

        # Optimize kd, od, bd via constrained minimization
        valid_d_idx = np.array([d['d'] is not None for d in data_all])
        dx_instances = x_diff[valid_d_idx]
        obs_dir_instances = np.array([d['d'] for d in data_all])[valid_d_idx]
        result = minimize(
            loss_obs_dir_cpd,
            [params_old['kd'], params_old['od'], params_old['bd']],
            args=(dx_instances, obs_dir_instances),
            bounds=Bounds(RobotModel.obs_dir_optim_params_lb, RobotModel.obs_dir_optim_params_ub),
            options=RobotModel.obs_dir_optim_options,
        )
        if not result.success:
            print("Warning: Bounded optimization of observe_dir() params failed")
        else:
            params_new['kd'], params_new['od'], params_new['bd'] = result.x

        return params_new