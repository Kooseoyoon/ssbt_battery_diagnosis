import unittest
import numpy as np

from battery_sim.value_metrics import (current_only_assignment, effective_future,
    exact_small_oracle, fixed_random_pool, observed_prefix, pack_values,
    regret_components, regrouping_upper_bound, total_value)


class CapacityValueChecks(unittest.TestCase):
    def test_n10_reveals_only_cycles_1_and_6(self):
        c = np.array([[2., 1.9, 1.8, 1.3]])
        np.testing.assert_array_equal(observed_prefix(c, 10)[0], [2., 1.9])
        c[0, 2:] = 999.
        np.testing.assert_array_equal(observed_prefix(c, 10)[0], [2., 1.9])

    def test_different_diagnostic_offsets_are_aligned_by_relative_k(self):
        c = np.array([[2., 1.9, 1.8, 1.7, 1.3, np.nan],
                      [2., 1.95, 1.85, 1.75, 1.65, 1.3]])
        # Cell 0 starts at column 2, cell 1 at column 3: minima 1.75, 1.65.
        np.testing.assert_allclose(pack_values(c, [10, 15], [[0, 1]]), [5 * (1.75 + 1.65)])

    def test_strict_threshold_and_first_crossing_even_after_recovery(self):
        c = np.array([[1.5, 1.44, 1.43, 1.8], [1.6, 1.5, 1.6, 1.2]])
        self.assertAlmostEqual(total_value(c, 0, [[0, 1]]), 5 * (1.5 + 1.44))

    def test_nan_before_eol_is_not_imputed_as_zero(self):
        with self.assertRaises(ValueError):
            effective_future([[1.8, np.nan, 1.3]], 0)

    def test_all_retained_cells_must_be_used_once(self):
        with self.assertRaises(ValueError):
            pack_values([[1.8, 1.3], [1.9, 1.3]], 0, [[0, 0]])

    def test_random_exclusion_is_fixed_and_not_selected_by_capacity(self):
        kept, dropped = fixed_random_pool(np.arange(1, 87), 8, 42)
        self.assertEqual((len(kept), len(dropped)), (80, 6))
        np.testing.assert_array_equal(kept, fixed_random_pool(np.arange(1, 87), 8, 42)[0])

    def test_current_only_uses_descending_column_1(self):
        c = np.array([[100., 1.9], [0., 1.7], [50., 2.0], [70., 1.8]])
        np.testing.assert_array_equal(current_only_assignment(c, 2), [[2, 0], [3, 1]])

    def test_regret_components_share_zero_diagnosis_denominator(self):
        r = regret_components(100., 80., 60.)
        self.assertAlmostEqual(r['R_diag'], .2)
        self.assertAlmostEqual(r['R_assign'], .2)
        self.assertAlmostEqual(r['R_total'], .4)
        self.assertAlmostEqual(r['decomposition_error'], 0)

    def test_relaxation_dominates_exact_fixed_partition(self):
        c = np.array([[2., 1.6, 0.], [1.8, 1.7, 1.5],
                      [1.9, 1.8, 0.], [1.7, 1.65, 1.45]])
        exact, packs = exact_small_oracle(c, 2)
        self.assertGreaterEqual(regrouping_upper_bound(c, 2) + 1e-9, exact)
        self.assertAlmostEqual(pack_values(np.column_stack([c, np.full(4, 1.3)]), 0, packs).sum(), exact)


if __name__ == '__main__':
    unittest.main()
