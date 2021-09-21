import skrf as rf
import unittest
import os
import numpy as np

class DeembeddingTestCase(unittest.TestCase):
    '''
    Testcase for the Deembedding class

    Pseudo-netlists for s-parameter files used in these tests

    For open-short, open and short de-embedding:
    - deemb_ind.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Rline1  (1 2) resistor r=2ohm
        Dut_ind (2 3) inductor l=1nH
        Rline2  (3 4) resistor r=2ohm
        Cpad2   (4 0) capacitor c=25fF
        Cp2p    (1 4) capacitor c=10fF
        P2      (4 0) port
    - deemb_open.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Rline1  (1 2) resistor r=2ohm
        Rline2  (3 4) resistor r=2ohm
        Cpad2   (4 0) capacitor c=25fF
        Cp2p    (1 4) capacitor c=10fF
        P2      (4 0) port
    - deemb_short.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Rline1  (1 0) resistor r=2ohm
        Rline2  (0 4) resistor r=2ohm
        Cpad2   (4 0) capacitor c=25fF
        Cp2p    (1 4) capacitor c=10fF
        P2      (4 0) port

    For short-open de-embedding:
    - deemb_ind2.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Cpad1   (2 0) capacitor c=25fF
        Dut_ind (2 3) inductor l=1nH
        Cpad2   (3 0) capacitor c=25fF
        Cp2p    (2 3) capacitor c=10fF
        Rline2  (3 4) resistor r=2ohm
        P2      (4 0) port
    - deemb_open2.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Cpad1   (2 0) capacitor c=25fF
        Cpad2   (3 0) capacitor c=25fF
        Cp2p    (2 3) capacitor c=10fF
        Rline2  (3 4) resistor r=2ohm
        P2      (4 0) port
    - deemb_short2.s2p
        P1      (1 0) port
        Rline1  (1 0) resistor r=2ohm
        Cpad1   (0 0) capacitor c=25fF
        Cpad2   (0 0) capacitor c=25fF
        Cp2p    (0 0) capacitor c=10fF
        Rline2  (0 4) resistor r=2ohm
        P2      (4 0) port
    '''

    def setUp(self):
        '''
        Read in all the network data required for tests
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/deembed/'
        
        # for open-short, open and short testing
        self.raw = rf.Network(os.path.join(self.test_dir, 'deemb_ind.s2p'))
        self.open = rf.Network(os.path.join(self.test_dir, 'deemb_open.s2p'))
        self.short = rf.Network(os.path.join(self.test_dir, 'deemb_short.s2p'))

        # for spot frequency checking
        self.raw_1f = self.raw['10GHz'] 
        self.open_1f = self.open['10GHz'] 
        self.short_1f = self.short['10GHz']

        # for short-open testing
        self.raw2 = rf.Network(os.path.join(self.test_dir, 'deemb_ind2.s2p'))
        self.open2 = rf.Network(os.path.join(self.test_dir, 'deemb_open2.s2p'))
        self.short2 = rf.Network(os.path.join(self.test_dir, 'deemb_short2.s2p'))
        
        # for spot frequency checking
        self.raw2_1f = self.raw2['10GHz'] 
        self.open2_1f = self.open2['10GHz'] 
        self.short2_1f = self.short2['10GHz']

        # create de-embedding objects
        self.dm = rf.OpenShort(self.open, self.short)
        self.dm_os = rf.OpenShort(self.open_1f, self.short_1f) 
        self.dm_o = rf.Open(self.open_1f)
        self.dm_s = rf.Short(self.short_1f)
        self.dm_so = rf.ShortOpen(self.short2_1f, self.open2_1f)

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
        First do short de-embedding to remove 2ohm series resistors on each side, then remove
        open shunt capacitors which are 25fF to ground on each pad, and 10fF pad-to-pad.
        The resulting network should be a pure inductor of 1nH.
        '''
        dut = self.dm_so.deembed(self.raw2_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))