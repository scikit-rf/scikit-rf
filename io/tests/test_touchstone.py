
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

        spardict = touch.get_sparameter_data(format="ri")
        self.assertTrue("frequency" in spardict)
        self.assertTrue("S11R" in spardict)
        self.assertTrue("S11I" in spardict)
        self.assertTrue("S12R" in spardict)
        self.assertTrue("S12I" in spardict)
        self.assertTrue("S21R" in spardict)
        self.assertTrue("S21I" in spardict)
        self.assertTrue("S22R" in spardict)
        self.assertTrue("S22I" in spardict)
        self.assertTrue("S11DB" not in spardict)
        self.assertTrue("S11M" not in spardict)


suite = unittest.TestLoader().loadTestsFromTestCase(TouchstoneTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

