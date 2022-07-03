import unittest
import numpy as np
from skrf.vi.vna.hp8510c_sweep_plan import SweepSection, SweepPlan

class TestHP8510SweepPlan(unittest.TestCase):
    def test_801pt_swp(self):
        test_hz_0 = np.linspace(100,1000,801)
        test_sp_0 = SweepPlan.from_hz(test_hz_0)
        assert test_sp_0.matches_f_list(test_hz_0)

    def test_1001pt_swp(self):
        test_hz_1 = np.linspace(100,1000,1001)
        test_sp_1 = SweepPlan.from_hz(test_hz_1)
        assert test_sp_1.matches_f_list(test_hz_1)

    def test_multi_swp(self):
        test_hz_2 = np.concatenate(([1,2,3], np.linspace(100,1000,1001)))
        test_sp_2 = SweepPlan.from_hz(test_hz_2)
        assert test_sp_2.matches_f_list(test_hz_2)

    def test_multi_swp_with_single(self):
        test_hz_3 = np.concatenate(([1,2,3], np.linspace(100,1000,1001), [9999]))
        test_sp_3 = SweepPlan.from_hz(test_hz_3)
        assert test_sp_3.matches_f_list(test_hz_3)
