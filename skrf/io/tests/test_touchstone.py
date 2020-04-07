
import unittest
import os
import numpy as npy

import skrf as rf
from skrf.io.touchstone import Touchstone


class TouchstoneTestCase(unittest.TestCase):
    '''
    TouchstoneTestCase tests the IO of Touchstone files
    '''
    def setUp(self):
        '''
        Sets up the test directory
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'

    def test_read_data(self):
        '''
        This test reads data from simple_touchstone.s2p and compares with known
        true values.
        '''
        filename = os.path.join(self.test_dir, 'simple_touchstone.s2p')
        touch = Touchstone(filename)
        f, s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance)
        f_true = npy.array([1.00000000e+09, 1.10000000e+09])
        s_true = npy.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue((z0 == z0_true))


    def test_read_from_fid(self):
        '''
        This tests reading touch stone data from a file object as compared with
        a string path and name of the file.
        '''
        with open(os.path.join(self.test_dir, 'simple_touchstone.s2p')) as fid:
            touch = Touchstone(fid)
        f, s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance)
        f_true = npy.array([1.00000000e+09, 1.10000000e+09])
        s_true = npy.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue((z0 == z0_true))

    def test_get_sparameter_data(self):
        '''
        This tests the get_sparameter_data function.

        '''
        with open(os.path.join(self.test_dir, 'simple_touchstone.s2p')) as fid:
            touch = Touchstone(fid)

        expected_keys = ["frequency", "S11R", "S11I", "S12R", "S12I",
                "S21R", "S21I", "S22R", "S22I", ]

        unexpected_keys = ['S11DB', 'S11M', ]

        # get dict data structure
        sp_ri = touch.get_sparameter_data(format="ri")

        # test data structure
        for ek in expected_keys:
            self.assertTrue(ek in sp_ri)

        for uk in unexpected_keys:
            self.assertFalse(uk in sp_ri)

        # test data contents
        expected_sp_ri = {
            'frequency': npy.array([1.0e+09, 1.1e+09]),
            'S11R': npy.array([1., 9.]),
            'S11I': npy.array([ 2., 10.]),
            'S21R': npy.array([ 3., 11.]),
            'S21I': npy.array([ 4., 12.]),
            'S12R': npy.array([ 5., 13.]),
            'S12I': npy.array([ 6., 14.]),
            'S22R': npy.array([ 7., 15.]),
            'S22I': npy.array([ 8., 16.]),
        }

        for k in sp_ri:
            self.assertTrue(k in expected_sp_ri)

            self.assertTrue( (expected_sp_ri[k] == sp_ri[k]).all(),
                    msg='Field %s does not match. Expected "%s", got "%s"'%(
                        k, str(expected_sp_ri[k]), str(sp_ri[k]))  )


suite = unittest.TestLoader().loadTestsFromTestCase(TouchstoneTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

