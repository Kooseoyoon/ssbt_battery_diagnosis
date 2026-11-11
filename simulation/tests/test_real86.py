import itertools
import unittest

import numpy as np

from battery_sim.real86 import ObservationState, allocation, choose, rul_samples, score, run_policy
from test_core import pairings


class RealPilotChecks(unittest.TestCase):
    def test_subset_oracle_matches_exhaustive_search(self):
        values = np.array([90., 10., 100., 20., 45.])
        possible = [score(values, packs)
                    for subset in itertools.combinations(range(5), 4)
                    for packs in pairings(list(subset))]
        self.assertEqual(float(score(values, allocation(values, 2))), float(max(possible)))

    def test_86_candidates_yield_80_unique_selected_cells(self):
        chosen = allocation(np.arange(86), 8)
        self.assertEqual(chosen.shape, (10, 8))
        self.assertEqual(len(np.unique(chosen)), 80)
        self.assertEqual(set(range(86)) - set(chosen.ravel()), set(range(6)))

    def test_inspection_consumes_remaining_life(self):
        model = {'minimum_decay_Ah_per_100_cycles': .005, 'maximum_RUL_cycles': 2000}
        mu, cov, z = np.array([2., -.2]), np.eye(2) * 1e-12, np.zeros((3, 2))
        old = rul_samples(mu, cov, 21, z, model, 1.44)
        new = rul_samples(mu, cov, 26, z, model, 1.44)
        np.testing.assert_allclose(old - new, 5)

    def test_observation_state_has_no_hidden_labels_or_future_capacity(self):
        config = {'model': {'window_observations': 3, 'noise_Ah': .01,
                           'prior_mean': [2., -.15], 'prior_sd': [.3, .3],
                           'minimum_decay_Ah_per_100_cycles': .005, 'maximum_RUL_cycles': 2000},
                  'posterior_samples': 32, 'eol_capacity_Ah': 1.44}
        state = ObservationState.create(np.tile([1., 6., 11.], (4, 1)),
                                        np.tile([1.9, 1.88, 1.87], (4, 1)),
                                        np.full(4, 16.), config, 0)
        self.assertEqual(len(state.history_q[0]), 3)
        self.assertNotIn('eol', vars(state))
        self.assertNotIn('capacities', vars(state))
        self.assertEqual(choose('uniform', state, np.arange(4), np.random.default_rng(0))[0], 0)
        state.reveal(0, 16., 1.86, np.nan)
        self.assertTrue(np.isnan(state.next_cycles[0]))

    def test_hidden_eol_permutation_does_not_change_inspection_decisions(self):
        config = {'model': {'window_observations': 3, 'noise_Ah': .01,
                           'prior_mean': [2., -.15], 'prior_sd': [.3, .3],
                           'minimum_decay_Ah_per_100_cycles': .005, 'maximum_RUL_cycles': 2000},
                  'posterior_samples': 32, 'eol_capacity_Ah': 1.44,
                  'initial_observations': 3, 'maximum_observations': 5,
                  'additional_budgets_cell_cycles': [0, 5, 10], 'pack_size': 2,
                  'decision_candidate_limit': 4, 'decision_fantasies': 3,
                  'decision_evaluation_samples': 16}
        x = np.array([1, 6, 11, 16, 21])
        q = np.array([[1.9, 1.88, 1.87, 1.86, 1.85],
                      [1.92, 1.90, 1.89, 1.88, 1.87],
                      [1.89, 1.87, 1.85, 1.84, 1.83],
                      [1.95, 1.94, 1.93, 1.91, 1.9]])
        eol = np.array([200, 300, 400, 500])
        _, trace_a = run_policy(q, x, np.arange(4), eol, config, 'decision_value', 0)
        _, trace_b = run_policy(q, x, np.arange(4), eol[::-1], config, 'decision_value', 0)
        self.assertEqual(trace_a, trace_b)


if __name__ == '__main__':
    unittest.main()
