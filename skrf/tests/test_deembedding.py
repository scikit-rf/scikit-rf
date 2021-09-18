import skrf as rf
import unittest
import os
import numpy as np

class DeembeddingTestCase(unittest.TestCase):
    '''
    Testcase for the Deembedding class
    '''

    def setUp(self):
        '''
        Read in all the network data required for tests
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/deembed/'
        self.raw = rf.Network(os.path.join(self.test_dir, 'deemb_ind.s2p'))
        self.open = rf.Network(os.path.join(self.test_dir, 'deemb_open.s2p'))
        self.short = rf.Network(os.path.join(self.test_dir, 'deemb_short.s2p'))
        self.raw_1f = self.raw['2GHz'] 
        self.open_1f = self.open['2GHz'] 
        self.short_1f = self.short['2GHz']

        # create de-embedding objects
        self.dm = rf.OpenShort(self.open, self.short)
        self.dm_os = rf.OpenShort(self.open_1f, self.short_1f) 
        self.dm_o = rf.Open(self.open_1f)
        self.dm_s = rf.Short(self.short_1f)

        # relative tolerance for comparisons
        self.rtol = 1e-3

    def test_freqmismatch(self):
        '''
        Check that error is caught when networks are of different frequencies
        '''
        with self.assertRaises(ValueError):
            rf.OpenShort(self.open, self.short_1f)
        
        with self.assertRaises(ValueError):
            self.dm_os.deembed(self.raw)

    def test_openshort(self):
        '''
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        '''
        dut = self.dm_os.deembed(self.raw_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_open(self):
        '''
        After open de-embedding, the network is a R-L-R network with 2ohm-1nH-2ohm.
        Test that this is true at a spot frequency.
        '''
        dut = self.dm_o.deembed(self.raw_1f)
        res_calc = np.real(1/dut.y[0,0,0])
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(res_calc, 4, rtol=self.rtol))
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_short(self):
        '''
        First do open de-embedding, and next short. It should give pure inductor of 1nH.
        This is similar to OpenShort, but done in 2 steps.
        '''
        raw_minus_open = self.dm_o.deembed(self.raw_1f)
        dut = self.dm_s.deembed(raw_minus_open)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_shortopen(self):
        '''
        TODO: Add this test; need new networks to be created and added to tests/deembed folder
        '''