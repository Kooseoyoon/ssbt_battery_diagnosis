import unittest
import numpy as np
from battery_sim.inventory_protocol import BUDGET_GRID, diagnostic_batch, companion_dff
from battery_sim.value_metrics import (current_only_assignment, unused_cells,
    swap_neighbors, exact_small_oracle, regrouping_upper_bound)

class InventoryChecks(unittest.TestCase):
    def test_current_only_leaves_lowest_unused(self):
        c = np.tile(np.arange(86.)[:, None], (1,2))
        for m,r in [(4,2),(8,6),(12,2),(16,6)]:
            np.testing.assert_array_equal(unused_cells(current_only_assignment(c,m),86),np.arange(r))

    def test_all_budget_points_and_nominal_rounds(self):
        n = np.full(86,10)
        for target in BUDGET_GRID[1:]:
            sizes=[]
            while n.sum() < target:
                chosen,n=diagnostic_batch(n,np.argsort(n,kind='stable'),target)
                sizes.append(len(chosen))
            self.assertEqual(sizes,[9]*9+[5])
            self.assertEqual(n.sum(),target)
        np.testing.assert_array_equal(n,np.full(86,40))

    def test_cap_can_reduce_round_size(self):
        n=np.full(86,40); n[-1]=10
        chosen,new=diagnostic_batch(n,np.arange(86),3440)
        np.testing.assert_array_equal(chosen,[85])
        self.assertEqual(new[-1],15)
        self.assertLessEqual(new.max(),40)

    def test_both_kinds_of_swap(self):
        neighbors=list(swap_neighbors(np.array([[0,1],[2,3]]),5))
        self.assertEqual(len(neighbors),8)
        self.assertTrue(any(4 in x for x in neighbors))
        self.assertTrue(any(np.array_equal(x,[[2,1],[0,3]]) for x in neighbors))

    def test_unused_dff_cases_and_pack_label_invariance(self):
        ref=[[0,1,2],[3,4,5]]
        np.testing.assert_array_equal(companion_dff(ref,[[[0,1,2],[3,4,6]]],8),[0,0,0,.5,.5,1,1,0])
        np.testing.assert_array_equal(companion_dff(ref,[ref[::-1]],8),np.zeros(8))

    def test_partial_companion_change(self):
        d=companion_dff([[0,1,2],[3,4,5]],[[[0,1,3],[2,4,5]]],7)
        np.testing.assert_allclose(d,[.5,.5,1,1,.5,.5,0])

    def test_exact_oracle_selects_unused(self):
        c=np.array([[1.,1.],[2.,2.],[3.,3.],[4.,4.],[5.,5.]])
        value,packs=exact_small_oracle(c,2)
        self.assertEqual(value,60.)
        np.testing.assert_array_equal(unused_cells(packs,5),[0])
        self.assertGreaterEqual(regrouping_upper_bound(c,2),value)
