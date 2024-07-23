# TEST_ROBOT.py

import numpy as np
import matplotlib.pyplot as plt
# from scipy.stats import norm, vonmises
import time

from robotModel import RobotModel
from histoEngineEM import HistoEngineEM
from genRobotPathData import genRobotPathData



# %% Set up environment
np.random.seed(1729)

# %% Generate simulated observed data

num_time_steps = 201
init_pos = 0.3
# prop_behavior = 'stationary'
# prop_behavior = 'brownian'
prop_behavior = 'sweep'
# prop_behavior = 'sweep_tight'
prop_step_size = 0.01
prop_bias = 0.0
prop_stdev = 0.01
observe_pose_stdev = 0.1
observe_pose_freq = 20
observe_dir_od = 0.02
observe_dir_kd = 150
observe_dir_bd = 0.1
observe_dir_freq = 5
sweep_tight_lower_cap = 0.2

dataset = genRobotPathData(num_time_steps, init_pos,
                         prop_behavior, prop_step_size,
                         prop_bias, prop_stdev, observe_pose_stdev, observe_pose_freq,
                         observe_dir_od, observe_dir_kd, observe_dir_bd, observe_dir_freq,
                         sweep_tight_lower_cap)
# ds = dataset['data_all']
ds = np.array(dataset['data_all']) 

# %% Initialize engine

# Set to 0 to only run filtering+smoothing+MAP
num_em_steps = 5

num_bins = 300
hard_em_type = 'smooth'
em_params_eps_gain = 1
params = {
    'wx_bias': 0.1,
    'wx_spd': 0.05,
    'sx': 0.05,
    'od': 0.5,
    'kd': 25,
    'bd': 0.2,
    'sz': 0.1
}

model = RobotModel(params)
model_dup = RobotModel(params)
engine = HistoEngineEM(model, params, num_bins, [], ds, hard_em_type)
engine_dup = HistoEngineEM(model_dup, params, num_bins, [], ds, hard_em_type)

# %% Run smoothing/filtering/MAP

inference_t = -1
paramfit_t = -1
ljp_t = -1
if num_em_steps <= 0:
    tic = time.time()
    smoothing_probs, filtering_probs = engine.batchSmooth(ds, True)
    map_states = engine.extractMAP()
    inference_t = time.time() - tic

    # Run log joint prob
    tic = time.time()
    state_traj = map_states
    logjointprob = engine.logJointProb(ds, state_traj)
    ljp_t = time.time() - tic

    # Run param fitting
    tic = time.time()
    params_new = model.optimizeParams(ds, state_traj, params)
    paramfit_t = time.time() - tic

    # Plot results
    model.plotHistogram(ds, engine.cache, state_traj,
                        filtering_probs, smoothing_probs)
    plt.show()

# %% Run EM
em_t = -1
if num_em_steps > 0:
    engine.reset()
    tic = time.time()
    engine.runEM(num_em_steps, False, em_params_eps_gain, True)
    em_t = time.time() - tic

# %% Report results
if num_em_steps > 0:
    state_vec_matrix = np.tile(engine.cache['x_vec'], (ds.shape[0] + 1, 1))

    init_params = engine.em['params_list'][0]
    final_params = engine.em['params_list'][-1]

    engine_dup.reset()
    engine_dup.updateParams(init_params)
    cache = engine_dup.cache
    cache['ds'] = dataset
    smoothing_probs_first, filtering_probs_first = engine_dup.batchSmooth(ds, True)
    map_states_first = engine_dup.extractMAP()
    exp_smoothed_states_first = np.mean(smoothing_probs_first * state_vec_matrix, axis=1)
    ljp_first = engine_dup.logJointProb(ds, exp_smoothed_states_first)
    model_dup.plotHistogram(ds, cache, map_states_first,
                           filtering_probs_first, smoothing_probs_first, 1, 0.05)
    plt.title(f'First EM iter (Log Joint Prob w/ Expected Smoothed States: {ljp_first:.4e})')
    plt.show()

    engine_dup.reset()
    engine_dup.updateParams(final_params)
    cache = engine_dup.cache
    cache['ds'] = dataset
    smoothing_probs_last, filtering_probs_last = engine_dup.batchSmooth(ds, True)
    map_states_last = engine_dup.extractMAP()
    exp_smoothed_states_last = np.mean(smoothing_probs_last * state_vec_matrix, axis=1)
    ljp_last = engine_dup.logJointProb(ds, exp_smoothed_states_last)
    model_dup.plotHistogram(ds, cache, map_states_last,
                           filtering_probs_last, smoothing_probs_last, 2, [])
    plt.title(f'Final EM iter (Log Joint Prob w/ Expected Smoothed States: {ljp_last:.4e})')
    plt.show()

    curr_params = init_params
    print('\n')
    print('First Iter Params (vs GT | diff):')
    print(f'- wx_bias: {curr_params["wx_bias"]:.4f} ({prop_bias:.4f} | {abs(curr_params["wx_bias"] - prop_bias):.4f})')
    print(f'- wx_spd: {curr_params["wx_spd"]:.4f} ({prop_step_size:.4f} | {abs(curr_params["wx_spd"] - prop_step_size):.4f})')
    print(f'- sx: {curr_params["sx"]:.4f} ({prop_stdev:.4f} | {abs(curr_params["sx"] - prop_stdev):.4f})')
    print(f'- od: {curr_params["od"]:.4f} ({observe_dir_od:.4f} | {abs(curr_params["od"] - observe_dir_od):.4f})')
    print(f'- kd: {curr_params["kd"]:.4f} ({observe_dir_kd:.4f} | {abs(curr_params["kd"] - observe_dir_kd):.4f})')
    print(f'- bd: {curr_params["bd"]:.4f} ({observe_dir_bd:.4f} | {abs(curr_params["bd"] - observe_dir_bd):.4f})')
    print(f'- sz: {curr_params["sz"]:.4f} ({observe_pose_stdev:.4f} | {abs(curr_params["sz"] - observe_pose_stdev):.4f})')
    print('\n')

    curr_params = final_params
    print('Last Iter Params (vs GT | diff):')
    print(f'- wx_bias: {curr_params["wx_bias"]:.4f} ({prop_bias:.4f} | {abs(curr_params["wx_bias"] - prop_bias):.4f})')
    print(f'- wx_spd: {curr_params["wx_spd"]:.4f} ({prop_step_size:.4f} | {abs(curr_params["wx_spd"] - prop_step_size):.4f})')
    print(f'- sx: {curr_params["sx"]:.4f} ({prop_stdev:.4f} | {abs(curr_params["sx"] - prop_stdev):.4f})')
    print(f'- oc: {curr_params["od"]:.4f} ({observe_dir_od:.4f} | {abs(curr_params["od"] - observe_dir_od):.4f})')
    print(f'- kc: {curr_params["kd"]:.4f} ({observe_dir_kd:.4f} | {abs(curr_params["kd"] - observe_dir_kd):.4f})')
    print(f'- bc: {curr_params["bd"]:.4f} ({observe_dir_bd:.4f} | {abs(curr_params["bd"] - observe_dir_bd):.4f})')
    print(f'- sz: {curr_params["sz"]:.4f} ({observe_pose_stdev:.4f} | {abs(curr_params["sz"] - observe_pose_stdev):.4f})')
    print('\n')

print(f'TEST_ROBOT.py\n- inference: {inference_t:.4f} sec\n- log joint prob: {ljp_t:.4f} sec\n- param fit: {paramfit_t:.4f} sec\n- em: {em_t:.4f} sec\n\n')