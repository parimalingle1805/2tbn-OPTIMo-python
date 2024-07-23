import numpy as np
from scipy.stats import norm

def sigmoidCPD(x_prev, x_curr, dir_sign, kd, od, bd):
    """
    Sigmoid CPD function for direction observation.

    Args:
        x_prev: Previous position.
        x_curr: Current position.
        dir_sign: Sign of the direction (+1 or -1).
        kd: Steepness of the sigmoid.
        od: Offset of the sigmoid.
        bd: Bias term.

    Returns:
        Probability of observing the given direction.
    """
    return od / (1 + np.exp(-kd * dir_sign * (x_curr - x_prev))) + bd

def genRobotPathData(num_time_steps, init_pos,
                     prop_behavior, prop_step_size, prop_bias, prop_stdev,
                     observe_pose_stdev, observe_pose_freq=1, observe_dir_od=0.1,
                     observe_dir_kd=1000, observe_dir_bd=0, observe_dir_freq=np.inf, 
                     sweep_tight_lower_cap=0.4):
    """
    Generates time-series data for toy robot example.

    Args:
        num_time_steps: Number of time steps.
        init_pos: Initial position.
        prop_behavior: Propagation behavior ('stationary', 'brownian', 'sweep', 'sweep_tight').
        prop_step_size: Step size for propagation.
        prop_bias: Bias term for propagation.
        prop_stdev: Standard deviation for propagation noise.
        observe_pose_stdev: Standard deviation for pose observation noise.
        observe_pose_freq: Frequency of pose observations.
        observe_dir_od: Offset for direction observation sigmoid.
        observe_dir_kd: Steepness for direction observation sigmoid.
        observe_dir_bd: Bias for direction observation sigmoid.
        observe_dir_freq: Frequency of direction observations.
        sweep_tight_lower_cap: Lower cap for 'sweep_tight' behavior.

    Returns:
        Dictionary containing parameters and data:
            - 'params': Dictionary of parameters used.
            - 'data_all': List of dictionaries, each containing:
                - 'x': Ground truth pose.
                - 'u': Steering direction.
                - 'd': Observed direction.
                - 'z': Observed pose.
    """

    # Parse arguments
    sweep_tight_higher_cap = 1.0 - sweep_tight_lower_cap

    if prop_behavior not in ['stationary', 'brownian', 'sweep', 'sweep_tight']:
        print(f"Warning: Specified behavior [{prop_behavior}] is not recognized, setting to stationary")
        prop_behavior = 'stationary'

    ds = {}
    ds['params'] = {
        'num_time_steps': num_time_steps,
        'init_pos': min(max(init_pos, 0.0), 1.0),
        'propagate_behavior': prop_behavior,
        'propagate_step_size': prop_step_size,
        'propagate_bias': prop_bias,
        'propagate_stdev': prop_stdev,
        'observe_pose_stdev': observe_pose_stdev,
        'observe_pose_freq': observe_pose_freq,
        'observe_dir_od': observe_dir_od,
        'observe_dir_kd': observe_dir_kd,
        'observe_dir_bd': observe_dir_bd,
        'observe_dir_freq': observe_dir_freq
    }
    
    ds['data_all'] = []

    pos = init_pos
    steering_dir = 1

    for i in range(1, ds['params']['num_time_steps'] + 1):
        # Propagate
        if prop_behavior == 'stationary':
            u = 0
        elif prop_behavior == 'brownian':
            u = np.random.randint(3) - 1  # -1, 0, or 1
        elif prop_behavior == 'sweep':
            u = steering_dir
        elif prop_behavior == 'sweep_tight':
            u = steering_dir

        # Update pose
        prev_pos = pos
        pos = pos + u * prop_step_size + np.random.normal(prop_bias, prop_stdev)

        if prop_behavior == 'sweep_tight':
            if pos >= sweep_tight_higher_cap:
                pos = sweep_tight_higher_cap
                steering_dir = -1
            elif pos <= sweep_tight_lower_cap:
                pos = sweep_tight_lower_cap
                steering_dir = 1
        else:
            if pos >= 1:
                pos = 1
                steering_dir = -1
            elif pos <= 0:
                pos = 0
                steering_dir = 1

        # Observe direction
        if (i - 1) % int(ds['params']['observe_dir_freq']) == 0:
            prob_obs_pos_dir = sigmoidCPD(prev_pos, pos, 1,
                                        ds['params']['observe_dir_kd'],
                                        ds['params']['observe_dir_od'],
                                        ds['params']['observe_dir_bd'])
            prob_obs_neg_dir = sigmoidCPD(prev_pos, pos, -1,
                                        ds['params']['observe_dir_kd'],
                                        ds['params']['observe_dir_od'],
                                        ds['params']['observe_dir_bd'])
            r = np.random.rand()
            if r < prob_obs_pos_dir:
                d = 1
            elif r < prob_obs_pos_dir + prob_obs_neg_dir:
                d = -1
            else:
                d = 0
        else:
            d = None

        # Observe pose
        if (i - 1) % int(ds['params']['observe_pose_freq']) == 0:
            z = np.random.normal(pos, observe_pose_stdev)
            if z > 1:
                z = 1
            elif z < 0:
                z = 0
        else:
            z = None

        # Collect data
        ds['data_all'].append({'x': pos, 'u': u, 'd': d, 'z': z})

    return ds