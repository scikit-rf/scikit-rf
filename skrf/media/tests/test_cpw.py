import unittest
import os
import numpy as npy

from skrf.media import CPW
from skrf.network import Network
from skrf.frequency import Frequency
import skrf as rf
from numpy.testing import assert_array_almost_equal, assert_allclose, run_module_suite
from matplotlib import pyplot as plt
import pytest

rf.stylely()


class CPWTestCase(unittest.TestCase):
    def setUp(self):
        self.data_dir_qucs = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj')
        self.data_dir_ads = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'ads')
            
        fname = os.path.join(self.data_dir_qucs, 'cpw.s2p')
        self.qucs_ntwk = rf.Network(fname)

        # create various examples
        self.freq = rf.Frequency(start=1, stop=20, npoints=21, unit='GHz')
        # infinite quarz substrate, infinitely thin metal
        self.cpw1 = CPW(frequency=self.freq, w=40e-6, s=20e-6, h = 100e-3, ep_r=3.78)
        # infinite GaAs substrate, infinitely thin metal
        self.cpw2 = CPW(frequency=self.freq, w=75e-6, s=50e-6, h = 100e-3, ep_r=12.9)

        # coplanar on FR-4 printed circuit board without conductor backing
        # with zero thickness strip
        self.cpw4 = CPW(frequency = self.freq, w = 3.0e-3, s = 0.3e-3,
                        t = None, ep_r = 4.5, rho = None,  z0 = 50.)
        self.cpw5 = CPW(frequency = self.freq, w = 3.0e-3, s = 0.3e-3,
                        t = 0., ep_r = 4.5, rho = None, z0 = 50.)
        
        # more newtorks to test against Qucs with air or metal backing
        self.ref_qucs = [
            {'has_metal_backside': True, 'w': 1.6e-3, 's': 0.3e-3, 't': 35e-6,
             'h': 1.55e-3, 'color': 'b',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
             'cpw,t=35um,w=1.6mm,s=0.3mm,l=25mm,backside=metal.s2p'))},
            {'has_metal_backside': False, 'w': 3.0e-3, 's': 0.3e-3, 't': 35e-6,
             'h': 1.55e-3, 'color': 'g',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
             'cpw,t=35um,w=3mm,s=0.3mm,l=25mm,backside=air.s2p'))},
            {'has_metal_backside': False, 'w': 3.0e-3, 's': 0.3e-3, 't': 0,
              'h': 100e-3, 'color': 'r',
              'n': rf.Network(os.path.join(self.data_dir_qucs,
              'cpw,t=0,h=100mm,w=3mm,s=0.3mm,l=25mm,backside=air.s2p'))},
            ]
        
        self.ref_ads = [
            {'has_metal_backside': False, 'w': 3.0e-3, 's': 0.3e-3, 't': 0.,
              'h': 1.55e-3, 'color': 'C0',
              'n': rf.Network(os.path.join(self.data_dir_ads,
                            'cpw,t=0um.s2p'))},
            {'has_metal_backside': True, 'w': 1.6e-3, 's': 0.3e-3, 't': 0.,
              'h': 1.55e-3, 'color': 'C1',
              'n': rf.Network(os.path.join(self.data_dir_ads,
                            'cpwg,t=0um.s2p'))},
            ]
        
        # these would fail comparison because ADS use another strip thickness
        # correction. Kept for reference in case of future work.
        self.ref_ads_failling = [
            {'has_metal_backside': False, 'w': 3.0e-3, 's': 0.3e-3, 't': 35e-6,
             'h': 1.55e-3, 'color': 'C2',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'cpw,t=35um.s2p'))},
            {'has_metal_backside': True, 'w': 1.6e-3, 's': 0.3e-3, 't': 35e-6,
             'h': 1.55e-3, 'color': 'C3',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'cpwg,t=35um.s2p'))},
            ]
        
        # default parameter set for tests
        self.verbose = False # output comparison plots if True
        self.l    = 25e-3
        self.ep_r = 4.5
        self.tand = 0.018
        self.rho  = 1.7e-8

    def test_qucs_network(self):
        """
        Test against the Qucs project results
        """
        if self.verbose:
            fig, axs = plt.subplots(2, 2, figsize = (8,6))
            fig.suptitle('qucs/skrf')
            fig2, axs2 = plt.subplots(2, 2, figsize = (8,6))
            fig2.suptitle('qucs/skrf residuals')
            fig3, ax3 = plt.subplots(1, 1, figsize = (8,3))
            
        limit = 2e-3
        
        for ref in self.ref_qucs:
            cpw = CPW(frequency = ref['n'].frequency, z0 = 50.,
                            w = ref['w'], s = ref['s'], t = ref['t'],
                            h = ref['h'],
                            has_metal_backside = ref['has_metal_backside'],
                            ep_r = self.ep_r, rho = self.rho,
                            tand = self.tand,
                            compatibility_mode = 'qucs',
                            diel = 'frequencyinvariant')
            with pytest.warns(FutureWarning, match="`embed` will be deprecated"):
                line = cpw.line(d=self.l, unit='m', embed = True, z0=cpw.Z0)
            line.name = '`Media.CPW` skrf,qucs'
            
            # residuals
            res = line - ref['n']

            # test if within limit
            self.assertTrue(npy.all(npy.abs(res.s) < limit))
            
            if self.verbose:
                ax3.plot(npy.abs(res.s[0,0]))
                ax3.plot(npy.abs(res.s[0,1]))
                ax3.plot(npy.abs(res.s[1,0]))
                ax3.plot(npy.abs(res.s[1,1]))
                res = line / ref['n']
                res.name = 'residuals ' + ref['n'].name
                line.plot_s_db(0, 0, ax = axs[0, 0], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_db(0, 0, ax = axs[0, 0], color = ref['color'])
                res.plot_s_db(0, 0, ax = axs2[0, 0], linestyle = 'dashed',
                              color = ref['color'])
                
                line.plot_s_deg(0, 0, ax = axs[0, 1], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_deg(0, 0, ax = axs[0, 1], color = ref['color'])
                res.plot_s_deg(0, 0, ax = axs2[0, 1], linestyle = 'dashed',
                              color = ref['color'])
                
                line.plot_s_db(1, 0, ax = axs[1, 0], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_db(1, 0, ax = axs[1, 0], color = ref['color'])
                res.plot_s_db(1, 0, ax = axs2[1, 0], linestyle = 'dashed',
                              color = ref['color'])
                
                line.plot_s_deg(1, 0, ax = axs[1, 1], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_deg(1, 0, ax = axs[1, 1], color = ref['color'])
                res.plot_s_deg(1, 0, ax = axs2[1, 1], linestyle = 'dashed',
                              color = ref['color'])
                
        
        if self.verbose:
            axs[1, 0].legend(prop={'size': 6})
            axs[0, 0].get_legend().remove()
            axs[0, 1].get_legend().remove()
            axs[1, 1].get_legend().remove()
            fig.tight_layout()
            
            axs2[1, 0].legend(prop={'size': 6})
            axs2[0, 0].get_legend().remove()
            axs2[0, 1].get_legend().remove()
            axs2[1, 1].get_legend().remove()
            fig2.tight_layout()
    
    def test_ads_network(self):
        """
        Test against the ADS simulator results
        """
        if self.verbose:
            fig, axs = plt.subplots(2, 2, figsize = (8,6))
            fig.suptitle('ads/skrf')
            fig2, axs2 = plt.subplots(2, 2, figsize = (8,6))
            fig2.suptitle('ads/skrf residuals')
            fig3, ax3 = plt.subplots(1, 1, figsize = (8,3))
            
        limit = 1e-3
        
        for ref in self.ref_ads:
            cpw = CPW(frequency = ref['n'].frequency, z0 = 50.,
                            w = ref['w'], s = ref['s'], t = ref['t'],
                            h = ref['h'],
                            has_metal_backside = ref['has_metal_backside'],
                            ep_r = self.ep_r, rho = self.rho,
                            tand = self.tand,
                            compatibility_mode = 'ads',
                            diel = 'djordjevicsvensson')
            with pytest.warns(FutureWarning, match="`embed` will be deprecated"):
                line = cpw.line(d=self.l, unit='m', embed = True, z0=cpw.Z0)
            line.name = '`Media.CPW` skrf,ads'
            
            # residuals
            res = line - ref['n']

            # test if within limit
            self.assertTrue(npy.all(npy.abs(res.s) < limit))
            
            if self.verbose:
                ax3.plot(npy.abs(res.s[0,0]))
                ax3.plot(npy.abs(res.s[0,1]))
                ax3.plot(npy.abs(res.s[1,0]))
                ax3.plot(npy.abs(res.s[1,1]))
                res = line / ref['n']
                res.name = 'residuals ' + ref['n'].name
                line.plot_s_db(0, 0, ax = axs[0, 0], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_db(0, 0, ax = axs[0, 0], color = ref['color'])
                res.plot_s_db(0, 0, ax = axs2[0, 0], linestyle = 'dashed',
                              color = ref['color'])
                
                line.plot_s_deg(0, 0, ax = axs[0, 1], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_deg(0, 0, ax = axs[0, 1], color = ref['color'])
                res.plot_s_deg(0, 0, ax = axs2[0, 1], linestyle = 'dashed',
                              color = ref['color'])
                
                line.plot_s_db(1, 0, ax = axs[1, 0], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_db(1, 0, ax = axs[1, 0], color = ref['color'])
                res.plot_s_db(1, 0, ax = axs2[1, 0], linestyle = 'dashed',
                              color = ref['color'])
                
                line.plot_s_deg(1, 0, ax = axs[1, 1], color = ref['color'],
                               linestyle = 'none', marker = 'x')
                ref['n'].plot_s_deg(1, 0, ax = axs[1, 1], color = ref['color'])
                res.plot_s_deg(1, 0, ax = axs2[1, 1], linestyle = 'dashed',
                              color = ref['color'])
                
        
        if self.verbose:
            axs[1, 0].legend(prop={'size': 6})
            axs[0, 0].get_legend().remove()
            axs[0, 1].get_legend().remove()
            axs[1, 1].get_legend().remove()
            fig.tight_layout()
            
            axs2[1, 0].legend(prop={'size': 6})
            axs2[0, 0].get_legend().remove()
            axs2[0, 1].get_legend().remove()
            axs2[1, 1].get_legend().remove()
            fig2.tight_layout()

    def test_Z0(self):
        """
        Test the CPW Characteristic Impedances
        
        Values from http://wcalc.sourceforge.net/cgi-bin/coplanar.cgi
        """
        # values from http://wcalc.sourceforge.net/cgi-bin/coplanar.cgi
        assert_array_almost_equal(self.cpw1.Z0, 77.93, decimal=2)
        
    def test_ep_reff(self):
        """
        Test the effective permittivity of CPW
        """
        # values from https://www.microwaves101.com/calculators/864-coplanar-waveguide-calculator
        assert_array_almost_equal(self.cpw1.ep_reff, 2.39, decimal=2)
        assert_array_almost_equal(self.cpw2.ep_reff, 6.94, decimal=2)        
        
    def test_Z0_vs_f(self):
        """
        Test the CPW Characteristic Impedance vs frequency. 
        
        Reference data comes from Qucs Documentation (Fig 12.2)
        """        
        w_over_s_qucs, Z0_qucs = npy.loadtxt(
            os.path.join(self.data_dir_qucs, 'cpw_qucs_ep_r9dot5.csv'), 
            delimiter=';', unpack=True)
               
        w = 1
        Z0 = []
        for w_o_s in w_over_s_qucs:
            # simulate infinite thickness by providing h >> w
            _cpw = CPW(frequency=self.freq[0], w=w, s=w/w_o_s, h=1e9, ep_r=9.5)
            Z0.append(_cpw.Z0[0].real)
            
        # all to a 3% relative difference
        # this is quite a large discrepancy, but I extracted the ref values from the plot
        # one could do better eventually by extracting values from Qucs directly
        rel_diff = (Z0_qucs-npy.array(Z0))/Z0_qucs
        self.assertTrue(npy.all(npy.abs(rel_diff) < 0.03))
        
    def test_alpha_warning(self):
        """
        Test if alpha_conductor warns when t < 3 * skin_depth
        """
        # cpw line on 1.5mm FR-4 substrate
        freq = Frequency(1, 1, 1, 'MHz')
        with self.assertWarns(RuntimeWarning) as context:
            cpw = CPW(frequency = freq, z0 = 50., w = 3.0e-3, s = 0.3e-3, t = 35e-6,
                       ep_r = 4.5, rho = 1.7e-8)
            
            with pytest.warns(FutureWarning, match="`embed` will be deprecated"):
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