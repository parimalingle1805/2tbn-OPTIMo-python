# GEN_TECH_REPORT_PLOTS.py

import numpy as np
import matplotlib.pyplot as plt

from robotModel import RobotModel
from histoEngineEM import HistoEngineEM
from genRobotPathData import genRobotPathData
from errorbar_tick import errorbar_tick

# %% Set up environment
np.random.seed(1729)

# %% Generate simulated observed data

num_time_steps = 201
init_pos = 0.3
# prop_behavior = 'stationary'
# prop_behavior = 'brownian'
# prop_behavior = 'sweep'
prop_behavior = 'sweep_tight'
prop_step_size = 0.01
prop_bias = 0.0
prop_stdev = 0.01
observe_pose_stdev = 0.1
observe_pose_freq = 15
observe_dir_od = 0.02
observe_dir_kd = 150
observe_dir_bd = 0.1
observe_dir_freq = 5
sweep_tight_lower_cap = 0.3

dataset = genRobotPathData(num_time_steps, init_pos,
                           prop_behavior, prop_step_size,
                           prop_bias, prop_stdev, observe_pose_stdev, observe_pose_freq,
                           observe_dir_od, observe_dir_kd, observe_dir_bd, observe_dir_freq,
                           sweep_tight_lower_cap)
ds = dataset['data_all']

# %% Initialize engine and run filtering through first N time steps

N_pre_filter_steps = 45
num_bins = 100
hard_em_type = 'smooth'
em_params_eps_gain = 1
params = {
    'wx_bias': 0.05,
    'wx_spd': 0.15,
    'sx': 0.05,
    'od': 0.3,
    'kd': 25,
    'bd': 0.1,
    'sz': 0.1
}

model = RobotModel(params)
engine = HistoEngineEM(model, params, num_bins, [], ds[:N_pre_filter_steps],
                      hard_em_type)
pre_filtered_pmfs = engine.batch_filter(ds[:N_pre_filter_steps], False)

# %% Plot propagation matrices for different values of u

data_curr = ds[N_pre_filter_steps].copy()  # Copy to avoid modifying original data
data_past = ds[N_pre_filter_steps - 1]
propCPDs = []
for u in [-1, 0, 1]:
    data_curr['u'] = u
    propCPDs.append(model.propagate(engine.cache, data_curr, data_past))
propCPDTitles = ['u = -1', 'u = 0', 'u = +1']
probCPDFNames = ['prop_cpd_neg', 'prop_cpd_same', 'prop_cpd_pos']

for i, propCPD in enumerate(propCPDs):
    fig = plt.figure(i)
    fig.set_size_inches(2.25, 2)
    plt.clf()
    plt.title(propCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    max_prob_cap = np.max(propCPD) / 15
    plt.imshow(propCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Prob(x_k|x_{k-1}, u_k)')
    cbar.set_ticks([])  # Remove ticks from colorbar

    plt.savefig(f'figures/{probCPDFNames[i]}.png', dpi=300)
    plt.savefig(f'figures/{probCPDFNames[i]}.emf')

    # Save raw image without colorbar
    fig.set_size_inches(2.10, 2)
    plt.clf()
    plt.title(propCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.imshow(propCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')
    plt.savefig(f'figures/{probCPDFNames[i]}.raw.png', dpi=300)
    plt.savefig(f'figures/{probCPDFNames[i]}.raw.emf')

# %% Plot direction observation matrices for different values of d

data_curr = ds[N_pre_filter_steps].copy()
data_past = ds[N_pre_filter_steps - 1]
obsDirCPDs = []
for d in [-1, 0, 1]:
    data_curr['d'] = d
    data_curr['z'] = None
    obsDirCPDs.append(model.observe(engine.cache, data_curr, data_past))
obsDirCPDTitles = ['d = -1', 'd = 0', 'd = +1']
obsDirCPDFNames = ['obs_dir_cpd_neg', 'obs_dir_cpd_same', 'obs_dir_cpd_pos']

for i, obsDirCPD in enumerate(obsDirCPDs):
    fig = plt.figure(i + 6)
    fig.set_size_inches(2.25, 2)
    plt.clf()
    plt.title(obsDirCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    max_prob_cap = np.max(obsDirCPD) / 1
    plt.imshow(obsDirCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Prob(d_k|x_{k-1}, x_k)')
    cbar.set_ticks([])

    plt.savefig(f'figures/{obsDirCPDFNames[i]}.png', dpi=300)
    plt.savefig(f'figures/{obsDirCPDFNames[i]}.emf')

    # Save raw image without colorbar
    fig.set_size_inches(2.10, 2)
    plt.clf()
    plt.title(obsDirCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.imshow(obsDirCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')
    plt.savefig(f'figures/{obsDirCPDFNames[i]}.raw.png', dpi=300)
    plt.savefig(f'figures/{obsDirCPDFNames[i]}.raw.emf')

# %% Plot pose observation matrices for different values of z

data_curr = ds[N_pre_filter_steps].copy()
data_past = ds[N_pre_filter_steps - 1]
obsPoseCPDs = []
for z in [0.1, 0.5, 0.9]:
    data_curr['d'] = None
    data_curr['z'] = z
    obsPoseCPDs.append(model.observe(engine.cache, data_curr, data_past))
obsPoseCPDTitles = ['z = 0.1', 'z = 0.5', 'z = 0.9']
obsPoseCPDFNames = ['obs_pose_cpd_01', 'obs_pose_cpd_05', 'obs_pose_cpd_09']

for i, obsPoseCPD in enumerate(obsPoseCPDs):
    fig = plt.figure(i + 12)
    fig.set_size_inches(2.25, 2)
    plt.clf()
    plt.title(obsPoseCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    max_prob_cap = np.max(obsPoseCPD) / 2
    plt.imshow(obsPoseCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')
    cbar = plt.colorbar()
    cbar.set_label('Prob(z_k|x_{k-1}, x_k)')
    cbar.set_ticks([])

    plt.savefig(f'figures/{obsPoseCPDFNames[i]}.png', dpi=300)
    plt.savefig(f'figures/{obsPoseCPDFNames[i]}.emf')

    # Save raw image without colorbar
    fig.set_size_inches(2.10, 2)
    plt.clf()
    plt.title(obsPoseCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.imshow(obsPoseCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')
    plt.savefig(f'figures/{obsPoseCPDFNames[i]}.raw.png', dpi=300)
    plt.savefig(f'figures/{obsPoseCPDFNames[i]}.raw.emf')

# %% Plot prior state matrix at N+1'th time step

priorCPD = pre_filtered_pmfs[-1, :]
priorCPDMat = np.tile(priorCPD, (num_bins, 1))
max_prob_cap = np.max(priorCPDMat)
prior_fnames = ['filter_prior', 'filter_prior_repmat']

for i in range(2):
    fig = plt.figure(i + 19)
    fig.set_size_inches(2.10, 2)
    plt.clf()
    plt.title('Prior Filtered Belief')
    plt.xlabel('x_{k-1}')
    if i > 0:
        plt.ylabel('x_k')
        plt.imshow(priorCPDMat,
                   cmap='hot',
                   extent=[0, 1, 0, 1],
                   origin='lower',
                   vmin=0,
                   vmax=max_prob_cap,
                   aspect='auto')
    else:
        plt.ylabel('b_f(x_{k-1})')
        plt.yticks([0, 1], ["", ""])  # Hide y-ticks

    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))
    plt.plot(engine.cache['x_vec'],
             priorCPD / 20,
             '-w',
             linewidth=8)
    plt.plot(engine.cache['x_vec'],
             priorCPD / 20,
             '-k',
             linewidth=4)
    plt.xlim([0, 1])
    plt.ylim([0, 1])

    plt.savefig(f'figures/{prior_fnames[i]}.png', dpi=300)
    plt.savefig(f'figures/{prior_fnames[i]}.emf')

# %% Plot propagation and observation matrices for N+1'th time step

data_curr = ds[N_pre_filter_steps].copy()
data_past = ds[N_pre_filter_steps - 1]
filterCPDs = []
filterCPDs.append(model.propagate(engine.cache, data_curr, data_past))
data_curr['z'] = None
filterCPDs.append(model.observe(engine.cache, data_curr, data_past))
data_curr = ds[N_pre_filter_steps].copy()
data_curr['d'] = None
filterCPDs.append(model.observe(engine.cache, data_curr, data_past))
data_curr = ds[N_pre_filter_steps].copy()
filterCPDTitles = [
    f'u = {data_curr["u"]}', f'd = {data_curr["d"]}',
    f'z = {data_curr["z"]:.1f}'
]
filterCPDFNames = [
    f'filter_prop_u{data_curr["u"]}',
    f'filter_obs_dir_d{data_curr["d"]}',
    f'filter_obs_pose_z{data_curr["z"]:.1f}'
]

for i, filterCPD in enumerate(filterCPDs):
    fig = plt.figure(i + 20)
    fig.set_size_inches(2.10, 2)
    plt.clf()
    plt.title(filterCPDTitles[i])
    plt.xlabel('x_{k-1}')
    plt.ylabel('x_k')
    plt.xticks(np.arange(0, 1.1, 0.2))
    plt.yticks(np.arange(0, 1.1, 0.2))

    plt.imshow(filterCPD,
               cmap='hot',
               extent=[0, 1, 0, 1],
               origin='lower',
               vmin=0,
               vmax=max_prob_cap,
               aspect='auto')

    plt.savefig(f'figures/{filterCPDFNames[i]}.png', dpi=300)
    plt.savefig(f'figures/{filterCPDFNames[i]}.emf')

# %% Plot joint belief, and filtered state matrix at N+1'th time step

# TODO: Implement joint belief plotting

# %% Plot smooth-prior state matrix at N'th time step

# TODO: Implement smooth-prior plotting

# %% Plot smoothed state matrix at N'th time step

# TODO: Implement smoothed state matrix plotting

# %% TODO: prediction illustrations

# %% TODO: MAP inference limitation illustrations

plt.show()