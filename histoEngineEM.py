from histoEngine import HistoEngine

import numpy as np

class HistoEngineEM(HistoEngine):
    def __init__(self, model, init_params, num_bins, prior_x_pmf, data_all, hard_em_type='map'):
        super().__init__(model, init_params, num_bins, prior_x_pmf)
        self.data_all = data_all
        self.reset(init_params, hard_em_type)

    def reset(self, init_params=None, hard_em_type=None):
        if init_params is None:
            init_params = self.em['params_list'][0]
        if hard_em_type is None:
            hard_em_type = self.em['hard_em_type']

        self.em = {
            'hard_em_type': hard_em_type,
            'iter': 0,
            'params_list': [init_params],
            'E_states_list': [],
            'E_LJP_list': [],
            'M_LJP_list': [],
            'converged': False
        }

    def runEM(self, num_iters, iter_verbose=False, param_eps_gain=1, em_verbose=False):
        """Runs the Expectation-Maximization (EM) algorithm."""
        err = 0
        state_vec_matrix = np.tile(self.cache['x_vec'], (len(self.data_all) + 1, 1))

        for i in range(1, num_iters + 1):
            if self.em['converged']:
                break

            if em_verbose:
                print(f"> EM loop {i:3d} / {num_iters:3d}, cur_iter = {self.em['iter']:3d}")

            self.em['iter'] += 1

            # E step: infer latent states for current model
            try:
                if self.em['hard_em_type'] == 'filter':
                    filtered_pmfs = self.batchFilter(self.data_all, iter_verbose)
                    self.em['E_states_list'].append(
                        np.mean(filtered_pmfs * state_vec_matrix, axis=1)
                    )
                elif self.em['hard_em_type'] == 'smooth':
                    smoothed_pmfs, _ = self.batchSmooth(self.data_all, iter_verbose, False)
                    self.em['E_states_list'].append(
                        np.mean(smoothed_pmfs * state_vec_matrix, axis=1)
                    )
                else:
                    self.batchFilter(self.data_all, iter_verbose)
                    self.em['E_states_list'].append(self.extractMAP())

            except Exception as err:
                if str(err) == "PMF became all-zero":
                    print(f"Warning: EM terminated prematurely on iter {self.em['iter']} "
                          f"due to error: {str(err)}")
                    break
                else:
                    raise err

            self.em['E_LJP_list'].append(
                self.logJointProb(self.data_all, self.em['E_states_list'][-1])
            )

            # M step: fit model parameters
            self.em['params_list'].append(
                self.model.optimizeParams(self.data_all,
                                        self.em['E_states_list'][-1],
                                        self.em['params_list'][-1])
            )

            if self.model.compareParams(self.em['params_list'][-2],
                                      self.em['params_list'][-1],
                                      param_eps_gain):
                self.em['converged'] = True

            self.updateParams(self.em['params_list'][-1])
            self.em['M_LJP_list'].append(
                self.logJointProb(self.data_all, self.em['E_states_list'][-1])
            )

        if iter_verbose:
            print(f"-> Total EM iters: {self.em['iter']} (converged: {self.em['converged']})")

        return err