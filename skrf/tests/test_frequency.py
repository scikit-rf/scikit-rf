import unittest
import os
import numpy as npy

import skrf as rf


class FrequencyTestCase(unittest.TestCase):
    '''

    '''
    def setUp(self):
        '''
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
    
    def test_create_linear_sweep(self):
        freq = rf.Frequency(1,10,10,'ghz')
        self.assertTrue((freq.f == npy.linspace(1,10,10)*1e9).all())
        self.assertTrue((freq.f_scaled ==npy.linspace(1,10,10)).all())
    
    def test_create_rando_sweep(self):
        f = npy.array([1,5,200])
        freq = rf.Frequency.from_f(f,unit='khz')
        self.assertTrue((freq.f ==f*1e3).all())
        self.assertTrue((freq.f_scaled== f).all())
    
    def test_rando_sweep_from_touchstone(self):
        '''
        this also tests the ability to read a touchstone file.
        '''
        rando_sweep_ntwk = rf.Network(os.path.join(self.test_dir, 'ntwk_arbitrary_frequency.s2p'))
        self.assertTrue((rando_sweep_ntwk.f == \
            npy.array([1,4,10,20])).all())


suite = unittest.TestLoader().loadTestsFromTestCase(FrequencyTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
