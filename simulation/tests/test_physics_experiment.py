import unittest

import numpy as np

from battery_sim.physics_experiment import (
    early_features,
    predicted_rul,
    query_rows,
    rank_predicted_rul,
)


class PhysicsExperimentChecks(unittest.TestCase):
    def test_features_use_only_the_two_revealed_capacities(self):
        capacities = np.array([[2.0, 1.9, 1.4, 1.1]])
        before = early_features(capacities)
        capacities[:, 2:] = 900.0
        np.testing.assert_array_equal(before, early_features(capacities))
        np.testing.assert_allclose(before, [[2.0, 1.9, -0.02]])

    def test_query_rows_use_time_and_only_early_cell_features(self):
        rows = query_rows([2.0, 1.9, -0.02], [11, 16], 2000)
        np.testing.assert_allclose(rows, [[2.0, 1.9, -0.02, 0.0025],
                                          [2.0, 1.9, -0.02, 0.005]])

    def test_first_strict_crossing_and_no_crossing_are_explicit(self):
        cycles = np.array([11, 16, 21])
        self.assertEqual(predicted_rul([1.5, 1.44, 1.43], cycles, 11, 1.44, 2000),
                         (10.0, "crossing_found"))
        self.assertEqual(predicted_rul([1.5, 1.46, 1.45], cycles, 11, 1.44, 2000),
                         (2000.0, "horizon_censored"))

    def test_equal_or_censored_rul_uses_observed_capacity_not_row_order(self):
        packs = rank_predicted_rul([2000, 2000, 100, 100], [1.8, 1.9, 1.7, 1.6],
                                   [1, 2, 3, 4], 2)
        np.testing.assert_array_equal(packs, [[1, 0], [2, 3]])


if __name__ == "__main__":
    unittest.main()
