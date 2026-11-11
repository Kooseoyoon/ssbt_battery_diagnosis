import unittest
import numpy as np
from battery_sim.experiment_v3 import (capacity_features_v3, align_feature_cube,
                                       rank_assignment)
from battery_sim.value_metrics import observed_prefix, current_only_assignment


class ExperimentV3Checks(unittest.TestCase):
    def test_two_rpt_features_and_future_isolation(self):
        c=np.array([[2.,1.9,1.8,1.2],[1.9,1.7,1.5,1.2]])
        before=capacity_features_v3([1,6],np.stack(observed_prefix(c,10)))
        c[:,2:]=999.
        after=capacity_features_v3([1,6],np.stack(observed_prefix(c,10)))
        np.testing.assert_array_equal(before,after)
        self.assertEqual(before.shape,(2,10))
        np.testing.assert_allclose(before[:,7:9],0,atol=1e-12)
        np.testing.assert_allclose(before[:,4],[-.02,-.04])

    def test_feature_ids_reordered_without_implicit_axis_guess(self):
        x=np.array([np.full((7,3),3.),np.full((7,3),1.),np.full((7,3),2.)])
        out=align_feature_cube(x,[3,1,2],range(10,41,5),[1,2,3])
        np.testing.assert_array_equal(out[:,0,0],[1,2,3])
        with self.assertRaises(ValueError):
            align_feature_cube(x,[1,1,2],range(10,41,5),[1,2,3])
        with self.assertRaises(ValueError):
            align_feature_cube(x,[1,2,3],range(5,36,5),[1,2,3])

    def test_current_only_equals_n10_updated_control(self):
        c=np.random.default_rng(1).uniform(1.5,2.,size=(86,8))
        for m in [4,8,12,16]:
            np.testing.assert_array_equal(rank_assignment(c[:,1],m),current_only_assignment(c,m))
