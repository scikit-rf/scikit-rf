import unittest
import os
import numpy as npy

import skrf as rf


class FrequencyTestCase(unittest.TestCase):
    '''

    '''
    def setUp(self):
        '''
        this also tests the ability to read touchstone files
        without an error
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))

    
    def test_create_linear_sweep(self):
        freq = rf.Frequency(1,10,10,'ghz')
        self.assertTrue((freq.f == npy.linspace(1,10,10)*1e9).all())
        self.assertTrue((freq.f_scaled ==npy.linspace(1,10,10)).all())
    
    def test_create_rando_sweep(self):
        f = npy.array([1,5,200])
        freq = rf.Frequency.from_f(f,unit='khz')
        self.assertTrue((freq.f ==f*1e3).all())
        self.assertTrue((freq.f_scaled== f).all())
        


suite = unittest.TestLoader().loadTestsFromTestCase(FrequencyTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
