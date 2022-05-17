# -*- coding: utf-8 -*-
import unittest
import os
import numpy as npy

from skrf.media import CPW
from skrf.network import Network
from skrf.frequency import Frequency
import skrf as rf
from numpy.testing import assert_array_almost_equal, assert_allclose, run_module_suite


class CPWTestCase(unittest.TestCase):
    def setUp(self):
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )        
        fname = os.path.join(self.files_dir, 'cpw.s2p')
        self.qucs_ntwk = rf.Network(fname)

        # create various examples
        self.freq = rf.Frequency(start=1, stop=20, npoints=21, unit='GHz')
        # infinite dielectric substrate, infinitely thin metal
        self.cpw1 = CPW(frequency=self.freq, w=40e-6, s=20e-6, ep_r=3)
        # infinite GaAs substrate, infinitely thin metal
        self.cpw2 = CPW(frequency=self.freq, w=75e-6, s=50e-6, ep_r=12.9)
        # infinite GaAs substrate, finite metal thickness
        # TODO: not used yet
        self.cpw3 = CPW(frequency=self.freq, w=75e-6, s=50e-6, ep_r=12.9, t=1e-6)
        # coplanar on FR-4 printed circuit board without conductor backing
        # with zero thickness strip
        self.cpw4 = CPW(frequency = self.freq, w = 3.0e-3, s = 0.3e-3,
                        t = None, ep_r = 4.5, rho = None,  z0 = 50.)
        self.cpw5 = CPW(frequency = self.freq, w = 3.0e-3, s = 0.3e-3,
                        t = 0., ep_r = 4.5, rho = None, z0 = 50.)

    def test_qucs_network(self):
        """
        Test against the Qucs project results
        TODO : finalize
        """
        # create an equivalent skrf network
        cpw = CPW(frequency=self.freq, w=75e-6, s=50e-6, ep_r=12.9, rho=0.22e-6)
        ntw = self.cpw2.thru(z0=50)**cpw.line(d=1, unit='m')**self.cpw2.thru(z0=50)
        # self.qucs_ntwk.plot_s_db()
        # ntw.plot_s_db()

    def test_Z0(self):
        """
        Test the CPW Characteristic Impedances
        """
        assert_array_almost_equal(self.cpw1.Z0, 85.25, decimal=3)
        
    def test_eps_eff(self):
        """
        Test the effective permittivity of CPW
        """
        assert_array_almost_equal(self.cpw1.ep_re, 2.00, decimal=3)
        assert_array_almost_equal(self.cpw2.ep_re, 6.95, decimal=3)        
        
    def test_Z0_vs_f(self):
        """
        Test the CPW Characteristic Impedance vs frequency. 
        
        Reference data comes from Qucs Documentation (Fig 12.2)
        """        
        w_over_s_qucs, Z0_qucs = npy.loadtxt(
            os.path.join(self.files_dir, 'cpw_qucs_ep_r9dot5.csv'), 
            delimiter=';', unpack=True)
               
        w = 1
        Z0 = []
        for w_o_s in w_over_s_qucs:
            _cpw = CPW(frequency=self.freq[0], w=w, s=w/w_o_s, ep_r=9.5)
            Z0.append(_cpw.Z0[0].real)
            
        # all to a 3% relative difference
        # this is quite a large discrepancy, but I extracted the ref values from the plot
        # one could do better eventually by extracting values from Qucs directly
        rel_diff = (Z0_qucs-npy.array(Z0))/Z0_qucs
        assert_allclose(rel_diff  - 3/100, 0, atol=0.1)
        
    def test_alpha_warning(self):
        """
        Test if alpha_conductor warns when t < 3 * skin_depth
        """
        # cpw line on 1.5mm FR-4 substrate
        freq = Frequency(1, 1, 1, 'MHz')
        with self.assertWarns(RuntimeWarning) as context:
            cpw = CPW(frequency = freq, z0 = 50., w = 3.0e-3, s = 0.3e-3, t = 35e-6,
                       ep_r = 4.5, rho = 1.7e-8)
            line = cpw.line(d = 25e-3, unit = 'm', embed = True, z0 = cpw.Z0)
            
    def test_zero_thickness(self):
        """
        Test if alpha_conductor is nullified when thikness = 0. or None
        """
        assert_array_almost_equal(self.cpw4.alpha_conductor, 0.00, decimal=6)
        assert_array_almost_equal(self.cpw5.alpha_conductor, 0.00, decimal=6)

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()