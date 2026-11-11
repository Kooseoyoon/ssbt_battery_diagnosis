import json
import unittest
from pathlib import Path

import numpy as np

from battery_sim.compare_models import (add_case, calibration_scale, features,
                                         optimize_packs, sample_rul)
from battery_sim.real86 import allocation, score


class ComparisonChecks(unittest.TestCase):
    def test_future_capacity_cannot_change_prefix_features(self):
        x = np.arange(1, 27, 5)
        capacity = np.array([[1.9, 1.88, 1.86, 1.84, 1.82, 1.8]])
        first = features(x[:3], capacity[:, :3])
        capacity[:, 3:] = 1000
        np.testing.assert_array_equal(first, features(x[:3], capacity[:, :3]))

    def test_separate_calibration_uses_finite_sample_quantile(self):
        residual = np.arange(1., 12.)
        factor, radius = calibration_scale(residual, np.zeros(11), np.ones(11), .8)
        self.assertEqual(radius, 10.)
        self.assertGreater(factor, 1)

    def test_swap_improves_expected_minimum_not_just_mean_rank(self):
        scenarios = np.array([[1., 2., 90., 100.], [100., 90., 2., 1.]])
        initial = allocation(scenarios.mean(axis=0), 2)
        optimized, objectives = optimize_packs(scenarios, initial, 10)
        self.assertTrue(np.all(np.diff(objectives) > 0))
        self.assertEqual(float(score(scenarios, optimized).mean()), 91.)
        self.assertEqual(len(np.unique(optimized)), 4)

    def test_joint_scenarios_preserve_cell_order_and_cap(self):
        log_mean = np.log([100., 300.])
        samples = sample_rul(log_mean, np.eye(2) * .1, np.zeros((4, 2)), 2000)
        np.testing.assert_allclose(samples, np.tile([100., 300.], (4, 1)))

    def test_test_label_permutation_does_not_change_models_or_assignments(self):
        cfg = json.loads((Path(__file__).resolve().parents[1] / 'configs/model_comparison_v2.json').read_text())
        cfg.update({'inner_folds': 2, 'ridge_alphas': [1.],
                    'gp_candidates': [{'length_scale': 3., 'noise_level': .2}],
                    'optimization_scenarios': 16, 'validation_scenarios': 32,
                    'maximum_swaps': 2, 'pack_size': 2, 'observation_counts': [3]})
        x = np.array([1., 6., 11.])
        q = np.array([[1.85 + .01 * i, 1.83 + .01 * i, 1.81 + .01 * i] for i in range(16)])
        ids = np.arange(1, 17)
        eol = np.linspace(170, 600, 16)
        proper, cal, test = np.arange(8), np.arange(8, 13), np.arange(13, 16)
        rows_a, predictions_a, fits_a = add_case(q, x, ids, eol, proper, cal, test, 3, 11, 0, cfg)
        eol_b = eol.copy(); eol_b[test] = eol[test][::-1]
        rows_b, predictions_b, fits_b = add_case(q, x, ids, eol_b, proper, cal, test, 3, 11, 0, cfg)
        self.assertEqual(fits_a, fits_b)
        self.assertEqual([r['assignment_cell_ids'] for r in rows_a], [r['assignment_cell_ids'] for r in rows_b])
        self.assertEqual([p['predicted_RUL_median'] for p in predictions_a], [p['predicted_RUL_median'] for p in predictions_b])


if __name__ == '__main__':
    unittest.main()
