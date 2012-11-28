import unittest
import os
import numpy as npy

import skrf as rf


class TouchstoneTestCase(unittest.TestCase):
    '''

    '''
    def setUp(self):
        '''
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        
    def read_data(self):
        filename = os.path.join(self.test_dir, 'simple_touchstone.s2p')
        touch= rf.touchstone.touchstone(filename)
        f,s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance())
        f_true = array([  1.00000000e+09,   1.10000000e+09])
        s_true = npy.array([
                [[  1. +2.j,   5. +6.j],
                [  3. +4.j,   7. +8.j]],
                [[  9.+10.j,  13.+14.j],
                [ 11.+12.j,  15.+16.j]]
            ])
        z0_true = 50+50j
        
        self.assertEqual(f,f_true)
        self.assertEqual(s,s_strue)
        self.assertEqual(z0,z0_strue)
suite = unittest.TestLoader().loadTestsFromTestCase(TouchstoneTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
