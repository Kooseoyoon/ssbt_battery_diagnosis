"""Checks for the mathematical assumptions used by the starter simulation."""

import unittest

import numpy as np

from battery_sim.core import sorted_assignment, utility


def pairings(ids):
    if not ids:
        yield []
        return
    first, *rest = ids
    for j, second in enumerate(rest):
        for tail in pairings(rest[:j] + rest[j + 1:]):
            yield [[first, second], *tail]


class PackScoringChecks(unittest.TestCase):
    def test_sorted_assignment_matches_exhaustive_oracle(self):
        values = np.array([90., 10., 100., 20., 200., 45.])
        exhaustive = max(float(utility(values, a)) for a in pairings(list(range(6))))
        self.assertEqual(float(utility(values, sorted_assignment(values, 2))), exhaustive)

    def test_fixed_assignment_does_not_reoptimize_each_scenario(self):
        scenarios = np.array([[10., 20., 90., 100.], [10., 90., 20., 100.]])
        assignment = np.array([[0, 1], [2, 3]])
        np.testing.assert_array_equal(utility(scenarios, assignment), [100., 30.])
        perfect_information = [utility(row, sorted_assignment(row, 2)) for row in scenarios]
        self.assertLess(float(utility(scenarios, assignment).mean()), float(np.mean(perfect_information)))

    def test_remainder_is_not_silently_discarded(self):
        with self.assertRaises(ValueError):
            sorted_assignment(np.arange(86), 8)

    def test_duplicate_cell_is_rejected(self):
        with self.assertRaises(ValueError):
            utility([10, 20, 90, 100], [[0, 1], [1, 3]])


if __name__ == "__main__":
    unittest.main()
