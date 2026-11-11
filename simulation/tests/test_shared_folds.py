import unittest
import numpy as np
from battery_sim.shared_folds import create_folds, validate_folds


class SharedFoldChecks(unittest.TestCase):
    def test_five_balanced_cell_folds_are_deterministic(self):
        ids = np.arange(1, 87)
        frame = create_folds(ids)
        self.assertTrue(frame.equals(create_folds(ids)))
        self.assertEqual(sorted(frame.fold.value_counts().tolist()), [17, 17, 17, 17, 18])
        self.assertEqual(len(validate_folds(frame, ids)), 86)

    def test_duplicate_physical_cell_is_rejected(self):
        ids = np.arange(1, 87)
        frame = create_folds(ids)
        frame.loc[1, 'cell_id'] = frame.loc[0, 'cell_id']
        with self.assertRaises(ValueError):
            validate_folds(frame, ids)
