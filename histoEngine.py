import numpy as np
from robotModel import RobotModel

class HistoEngine:
    def __init__(self, model, params, num_bins, prior_x_pmf=None):
        self.prior_x_pmf = prior_x_pmf if prior_x_pmf is not None else []
        self.model = model
        self.settings = {'num_bins': num_bins}
        self.buildCache()
        self.updateParams(params)

    def buildCache(self):
        """(Re-)builds temporary cache data."""
        self.cache = {}
        self.cache['num_bins'] = self.settings['num_bins']
        self.cache['bin_width'] = (1.0 - 0.0) / self.settings['num_bins']
        self.cache['eye_num_bins'] = np.eye(self.settings['num_bins'])

        self.cache['x_vec'] = (self.cache['bin_width'] / 2 +
                               np.arange(self.settings['num_bins']) * self.cache['bin_width'])
        self.cache['x_past'] = np.tile(self.cache['x_vec'][:, None], (1, self.settings['num_bins']))
        self.cache['x_curr'] = self.cache['x_past'].T

    def updatePriorPMF(self, new_prior_x_pmf):
        """Initiates prior belief on latent state."""
        self.prior_x_pmf = new_prior_x_pmf
        self.resetState()

    def updateParams(self, new_params):
        """Updates parameters for PGM."""
        self.model.updateParams(new_params)
        self.resetState()

    def resetState(self):
        """Restores default state and removes previous inference results."""
        if not hasattr(self, 'cache') or self.cache is None:
            self.buildCache()

        self.state = {'curr_time': 0}  # Initialize state dictionary

        # Initialize 'x_latest_pmf' BEFORE using it
        if self.prior_x_pmf is not None and len(self.prior_x_pmf) > 0:
            if len(self.prior_x_pmf) != self.cache['num_bins']:
                raise ValueError(
                    f"User-specified prior PMF does not have the correct number of bins: "
                    f"found {len(self.prior_x_pmf)}, expecting {self.cache['num_bins']}"
                )
            pmf_norm = np.sum(self.prior_x_pmf)
            if pmf_norm == 0:
                raise ValueError("User-specified prior PMF sums to zero")
            elif abs(pmf_norm - self.cache['num_bins']) > 1e-10:
                print(
                    f"Warning: User-specified prior PMF does not add up to num_bins: "
                    f"found {np.sum(self.prior_x_pmf):.4f}, expecting {self.cache['num_bins']}"
                )
                self.prior_x_pmf = self.prior_x_pmf / pmf_norm * self.cache['num_bins']
            self.state['x_latest_pmf'] = self.prior_x_pmf
        else:
            self.state['x_latest_pmf'] = np.ones(self.cache['num_bins'])

        # Now you can safely use self.state['x_latest_pmf']
        self.state['x_filtered_pmfs'] = np.zeros((self.settings['num_bins'] + 1, self.cache['num_bins']))
        self.state['x_filtered_pmfs'][0, :] = self.state['x_latest_pmf']   

        if self.prior_x_pmf is not None and len(self.prior_x_pmf) > 0:
            if len(self.prior_x_pmf) != self.cache['num_bins']:
                raise ValueError(
                    f"User-specified prior PMF does not have the correct number of bins: "
                    f"found {len(self.prior_x_pmf)}, expecting {self.cache['num_bins']}"
                )
            pmf_norm = np.sum(self.prior_x_pmf)
            if pmf_norm == 0:
                raise ValueError("User-specified prior PMF sums to zero")
            elif abs(pmf_norm - self.cache['num_bins']) > 1e-10:
                print(
                    f"Warning: User-specified prior PMF does not add up to num_bins: "
                    f"found {np.sum(self.prior_x_pmf):.4f}, expecting {self.cache['num_bins']}"
                )
                self.prior_x_pmf = self.prior_x_pmf / pmf_norm * self.cache['num_bins']
            self.state['x_latest_pmf'] = self.prior_x_pmf
        else:
            self.state['x_latest_pmf'] = np.ones(self.cache['num_bins'])

        self.state['x_filtered_pmfs'] = [self.state['x_latest_pmf']]
        self.state['latest_max_x_logpmf'] = np.log(self.state['x_latest_pmf'])
        self.state['max_x_idxs'] = np.ones_like(self.state['x_latest_pmf']) * -1

    def stepFilter(self, data_curr, data_past=None):
        """Iteratively updates the filtering belief."""
        if self.state['curr_time'] <= 0:
            data_past = {}  # Equivalent to empty struct in MATLAB
        self.state['curr_time'] += 1

        # Compute posterior using log probabilities for numerical stability
        logprobs_mat_prior = np.tile(np.log(self.state['x_latest_pmf'] + 1e-12)[:, None],
                                     (1, self.cache['num_bins']))
        logprobs_mat_propagate = np.log(self.model.propagate(self.cache, data_curr, data_past) + 1e-12)
        logprobs_mat_observe = np.log(self.model.observe(self.cache, data_curr, data_past) + 1e-12)
        logprobs_mat_local_factors = logprobs_mat_propagate + logprobs_mat_observe
        
        # Sum in log space
        log_prob_mat_posterior = logprobs_mat_prior + logprobs_mat_local_factors
        log_probs_posterior = np.logaddexp.reduce(log_prob_mat_posterior, axis=0)

        # Convert back to probabilities for normalization
        probs_posterior = np.exp(log_probs_posterior)

        # Marginalize and normalize
        norm_posterior = np.sum(probs_posterior)
        if norm_posterior == 0:
            raise ValueError(
                f"Filtering failed: sum p(t_{self.state['curr_time']} | "
                f"obs_1:{self.state['curr_time']}) == 0!"
            )
        norm_posterior /= self.cache['num_bins']
        self.state['x_latest_pmf'] = probs_posterior / norm_posterior

        # Use np.vstack to add the new filtered PMF as a row
        self.state['x_filtered_pmfs'] = np.vstack((
            self.state['x_filtered_pmfs'], self.state['x_latest_pmf']
        ))

        if np.any(np.isnan(self.state['x_latest_pmf'])):
            raise ValueError(
                f"Filtering failed: p(t_{self.state['curr_time']} | "
                f"obs_1:{self.state['curr_time']}) has NaN term!"
            )

        # Compute MAP-related quantities
        logprobs_mat_map_prior = np.tile(self.state['latest_max_x_logpmf'][:, None],
                                         (1, self.cache['num_bins']))
        logprobs_mat_map_posterior = logprobs_mat_map_prior + logprobs_mat_local_factors
        self.state['latest_max_x_logpmf'] = np.max(logprobs_mat_map_posterior, axis=0)
        latest_max_x_logpmf_idxs = np.argmax(logprobs_mat_map_posterior, axis=0)
        ambiguous_map_idx = (np.max(logprobs_mat_map_posterior, axis=0) ==
                             np.min(logprobs_mat_map_posterior, axis=0))
        latest_max_x_logpmf_idxs[ambiguous_map_idx] = np.arange(self.cache['num_bins'])[
            ambiguous_map_idx
        ]
        self.state['max_x_idxs'] = np.vstack(
            (self.state['max_x_idxs'], latest_max_x_logpmf_idxs)
        )

    def batchFilter(self, data_all, verbose=False):
        """Convenience function for batch filtering."""
        _, filtered_pmfs = self.batchSmooth(data_all, verbose, filter_only=True)
        return filtered_pmfs

    def batchSmooth(self, data_all, verbose=False, filter_only=False):
        """Batch computation of filtered and smoothed beliefs."""
        num_time_steps = len(data_all)
        self.resetState()
        smoothed_pmfs = []

        if verbose:
            for time in range(num_time_steps):
                print(f"Filtering+MAP {time + 1:3d} / {num_time_steps:3d} steps...")
                if time == 0:
                    self.stepFilter(data_all[time])
                else:
                    self.stepFilter(data_all[time], data_all[time - 1])
        else:
            for time in range(num_time_steps):
                if time == 0:
                    self.stepFilter(data_all[time])
                else:
                    self.stepFilter(data_all[time], data_all[time - 1])

        filtered_pmfs = np.array(self.state['x_filtered_pmfs'])

        if not filter_only:
            smoothed_pmfs = np.zeros((num_time_steps + 1, self.cache['num_bins']))
            latest_smoothed_x_pmf = self.state['x_latest_pmf']
            smoothed_pmfs[-1, :] = latest_smoothed_x_pmf

            for time in range(num_time_steps - 1, 0, -1):
                if verbose:
                    print(f"Smoothing {time + 1:3d} / {num_time_steps:3d} steps...")

                data_curr = data_all[time]
                data_past = data_all[time - 1] if time > 1 else {}
                logprobs_mat_prior = np.tile(np.log(self.state['x_filtered_pmfs'][time, :])[:, None],
                                             (1, self.cache['num_bins']))
                logprobs_mat_propagate = np.log(self.model.propagate(self.cache, data_curr, data_past))
                logprobs_mat_observe = np.log(self.model.observe(self.cache, data_curr, data_past))
                filtered_POF = np.exp(logprobs_mat_prior + logprobs_mat_propagate + logprobs_mat_observe)

                filtered_probs_posterior = np.sum(filtered_POF, axis=0)

                log_smoothed_posterior_over_filtered_posterior = (
                        np.log(latest_smoothed_x_pmf) - np.log(filtered_probs_posterior)
                )
                log_smoothed_posterior_over_filtered_posterior[filtered_probs_posterior == 0] = -np.inf
                smoothed_probs_backprior = np.exp(
                    np.log(filtered_POF) +
                    np.tile(log_smoothed_posterior_over_filtered_posterior[:, None], (1, self.cache['num_bins']))
                )

                latest_smoothed_x_pmf = np.sum(smoothed_probs_backprior, axis=0)
                norm_smoothed_x_pdf = np.sum(latest_smoothed_x_pmf)
                if norm_smoothed_x_pdf == 0:
                    raise ValueError(
                        f"Smoothing failed: sum p(x_{time} | obs_1:{num_time_steps}) == 0!"
                    )
                norm_smoothed_x_pdf /= self.cache['num_bins']
                latest_smoothed_x_pmf /= norm_smoothed_x_pdf
                if np.any(np.isnan(latest_smoothed_x_pmf)):
                    raise ValueError(
                        f"Smoothing failed: p(t_{time} | obs_1:{num_time_steps}) has NaN term!"
                    )

                smoothed_pmfs[time, :] = latest_smoothed_x_pmf

        return smoothed_pmfs, filtered_pmfs

    def extractMAP(self):
        """Extracts the MAP state sequence."""
        if not hasattr(self, 'cache') or self.cache is None:
            self.buildCache()

        if self.state['curr_time'] <= 0:
            return []

        map_states = np.zeros(self.state['curr_time'] + 1)
        latest_map_idx = np.argmax(self.state['latest_max_x_logpmf'])
        map_states[-1] = self.cache['x_vec'][latest_map_idx]
        for time in range(self.state['curr_time'] - 1, -1, -1):
            latest_map_idx = int(self.state['max_x_idxs'][time + 1, latest_map_idx])
            map_states[time] = self.cache['x_vec'][latest_map_idx]
        return map_states

    def logJointProb(self, data_all, states_all):
        """Computes the log joint probability of data and states."""
        if not hasattr(self, 'cache') or self.cache is None:
            self.buildCache()

        num_time_steps = len(data_all)

        state_diffs = np.abs(
            np.tile(states_all[:, None], (1, self.cache['num_bins'])) -
            np.tile(self.cache['x_vec'], (num_time_steps + 1, 1))
        )
        state_bin_idx = np.argmin(state_diffs, axis=1)
        state_binctr_all = self.cache['x_vec'][state_bin_idx]

        logprobs = 0  # log(1) for initial probability

        query = {
            'x_past': state_binctr_all[0],
            'x_curr': self.cache['x_vec'],
            'num_bins': self.cache['num_bins']
        }
        data_past = {}
        for time in range(num_time_steps):
            data_curr = data_all[time]
            query['eye_num_bins'] = self.cache['eye_num_bins'][:, state_bin_idx[time]]

            logprobs_propagate = np.log(self.model.propagate(query, data_curr, data_past))
            logprobs_observe = np.log(self.model.observe(query, data_curr, data_past))
            logprobs_propobs = logprobs_propagate + logprobs_observe
            lognorm_propobs = np.log(np.sum(np.exp(logprobs_propobs)))
            if np.isinf(lognorm_propobs):
                raise ValueError("Evaluation of logJointProb failed: sum p(...) == 0!")

            logprobs_propobs -= lognorm_propobs  # Normalize
            logprobs_propobs = logprobs_propobs[state_bin_idx[time + 1]]
            logprobs += logprobs_propobs

            if np.any(np.isnan(np.exp(logprobs_propobs))):
                raise ValueError("Evaluation of logJointProb failed: p(...) has NaN term!")

            query['x_past'] = state_binctr_all[time + 1]

        return logprobs

    @staticmethod
    def runModel(model, params, num_bins, data_all, prior_x_pmf=None, fig_id=1,
                 conf_int_alpha=None):
        """Plots filtering+smoothing+MAP inference on observed data seq."""
        engine = HistoEngine(model, params, num_bins, prior_x_pmf)
        smoothed_pmfs, filtered_pmfs = engine.batchSmooth(data_all, verbose=False, filter_only=False)
        map_states = engine.extractMAP()
        model.plotHistogram(data_all, engine.cache, map_states, filtered_pmfs,
                            smoothed_pmfs, fig_id, conf_int_alpha)
        return engine, filtered_pmfs, smoothed_pmfs, map_states

    @staticmethod
    def showTrainedModel(trial_obj, fig_id=1, conf_int_alpha=None):
        """Convenience function: plots model output on training set."""
        params = trial_obj['train']['opt_params']
        trained_model = RobotModel(params)  # Assuming RobotModel is defined elsewhere
        num_bins = trial_obj['settings']['num_bins']
        data_all = trial_obj['data']['ds_train']
        trained_prior_x_pmf = []  # Or adjust based on your needs

        return HistoEngine.runModel(
            trained_model, params, num_bins, data_all, trained_prior_x_pmf, fig_id,
            conf_int_alpha
        )

    @staticmethod
    def showRunModel(trial_obj, fig_id=1, conf_int_alpha=None):
        """Convenience function: plots model output on test set."""
        params = trial_obj['test']['params']
        test_model = RobotModel(params)  # Assuming RobotModel is defined elsewhere
        num_bins = trial_obj['settings']['num_bins']
        data_all = trial_obj['data']['ds_test']
        test_prior_x_pmf = trial_obj['test']['prior_x_pmf']

        return HistoEngine.runModel(
            test_model, params, num_bins, data_all, test_prior_x_pmf, fig_id,
            conf_int_alpha
        )