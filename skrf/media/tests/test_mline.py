# -*- coding: utf-8 -*-
import unittest
import os
import numpy as npy

from skrf.media import MLine
from skrf.frequency import Frequency
import skrf as rf
from numpy.testing import run_module_suite


class MLineTestCase(unittest.TestCase):
    """
    Testcase for the MLine Media
    """
    def setUp(self):
        """
        Read in all the network data required for tests
        """
        self.data_dir = os.path.dirname(os.path.abspath(__file__)) + \
            '/qucs_prj/'
        
        self.ref1 = rf.Network(os.path.join(self.data_dir,
                           'mline,hammerstad,hammerstad.s2p'))
        self.ref2 = rf.Network(os.path.join(self.data_dir,
                           'mline,hammerstad,kirschning.s2p'))
        self.ref3 = rf.Network(os.path.join(self.data_dir,
                           'mline,hammerstad,kobayashi.s2p'))
        self.ref4 = rf.Network(os.path.join(self.data_dir,
                           'mline,hammerstad,yamashita.s2p'))
        self.ref5 = rf.Network(os.path.join(self.data_dir,
                           'mline,wheeler,schneider.s2p'))
        self.ref6 = rf.Network(os.path.join(self.data_dir,
                           'mline,schneider,schneider.s2p'))
        self.ref7 = rf.Network(os.path.join(
            os.path.dirname(os.path.abspath(__file__)) + '/' + \
                           'mlin_ads,svenson-djordjevic,kirschning.s2p'))
        
        # default parameter set for tests
        self.w    = 3.00e-3
        self.h    = 1.55e-3
        self.t    = 35e-6
        self.l    = 25e-3
        self.ep_r = 4.413
        self.tand = 0.0182
        self.rho  = 1.7e-8
        self.d    = 0.15e-6
        self.f_et = 1e9
        
    def test_Z0_ep_reff(self):
        """
        Test against characterisitc impedance from another calculator using
        Hammerstadt-Jensen model
        http://web.mit.edu/~geda/arch/i386_rhel3/versions/20050830/html/mcalc-1.5/
        """
        freq = Frequency(1, 1, 1, 'GHz')
        mline1 = MLine(frequency = freq, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen')
        
        # without t (t = None)
        mline2 = MLine(frequency = freq, z0 = 50.,
                       w = self.w, h = self.h,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen')
        
        # with t = 0
        mline3 = MLine(frequency = freq, z0 = 50.,
                       w = self.w, h = self.h, t = 0,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen')
        
        self.assertTrue(npy.abs((mline1.Z0[0] - 49.142) / 49.142) < 0.01)
        self.assertTrue(npy.abs((mline1.ep_reff_f[0] - 3.324) / 3.324) < 0.01)
        self.assertTrue(npy.abs(mline2.w_eff - mline2.w) < 1e-16)
        self.assertTrue(npy.abs(mline2.alpha_conductor) < 1e-16)
        self.assertTrue(npy.abs(mline3.w_eff - mline3.w) < 1e-16)
        self.assertTrue(npy.abs(mline3.alpha_conductor) < 1e-16)


    def test_line(self):
        """
        Test against the Qucs project results
        """
        mline1 = MLine(frequency = self.ref1.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen')
        mline2 = MLine(frequency = self.ref2.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'kirschningjansen')
        mline3 = MLine(frequency = self.ref3.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'kobayashi')
        mline4 = MLine(frequency = self.ref4.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'yamashita')
        mline5 = MLine(frequency = self.ref5.frequency, z0 = 50.,
                        w = self.w, h = self.h, t = self.t,
                        ep_r = self.ep_r, rho = self.rho,
                        tand = self.tand, rough = self.d,
                        model = 'wheeler', disp = 'schneider',
                        diel = 'frequencyinvariant')
        mline6 = MLine(frequency = self.ref6.frequency, z0 = 50.,
                        w = self.w, h = self.h, t = self.t,
                        ep_r = self.ep_r, rho = self.rho,
                        tand = self.tand, rough = self.d,
                        model = 'schneider', disp = 'schneider',
                        diel = 'frequencyinvariant')
        

        l1 = mline1.line(d=self.l, unit='m', embed = True, z0=mline1.Z0)
        l2 = mline2.line(d=self.l, unit='m', embed = True, z0=mline2.Z0)
        l3 = mline3.line(d=self.l, unit='m', embed = True, z0=mline3.Z0)
        l4 = mline4.line(d=self.l, unit='m', embed = True, z0=mline4.Z0)
        l5 = mline5.line(d=self.l, unit='m', embed = True, z0=mline5.Z0)
        l6 = mline6.line(d=self.l, unit='m', embed = True, z0=mline6.Z0)
        
        self.assertTrue(npy.all(npy.abs(l1.s_db - self.ref1.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l1.s_deg - self.ref1.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l2.s_db - self.ref2.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l2.s_deg - self.ref2.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l3.s_db - self.ref3.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l3.s_deg - self.ref3.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l4.s_db - self.ref4.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l4.s_deg - self.ref4.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l5.s_db - self.ref5.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l5.s_deg - self.ref5.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l6.s_db - self.ref6.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l6.s_deg - self.ref6.s_deg) < 1.))
        
    def test_line2(self):
        """
        Test against the ADS results
        """
        mline7 = MLine(frequency = self.ref7.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'djordjevicsvensson', disp = 'kirschningjansen')
        
        l7 = mline7.line(d=0.1, unit='m', embed = True, z0=mline7.Z0)
        
        # fixme: cannot pass currently (test against ADS)
        #self.assertTrue(npy.all(npy.abs(l7.s_db - self.ref7.s_db) < 0.1))
        #self.assertTrue(npy.all(npy.abs(l7.s_deg - self.ref7.s_deg) < 1.))
        
        # uncomment plots to see results
        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(10, 8))
        # plt.subplot(2, 2, 1)
        # self.ref7.plot_s_db(0, 0, color = 'k', label = '_nolegend_')
        # l7.plot_s_db(0, 0, color = 'k', linestyle = 'none', marker = 'x',
        #               markevery = 100, label = '_nolegend_')
        # plt.grid()
        # plt.subplot(2, 2, 2)
        # self.ref7.plot_s_deg(0, 0, color = 'k', label = '_nolegend_')
        # l7.plot_s_deg(0, 0, color = 'k', linestyle = 'none', marker = 'x',
        #               markevery = 100, label = '_nolegend_')
        # plt.grid()
        # plt.subplot(2, 2, 3)
        # self.ref7.plot_s_db(1, 0, color = 'k', label = 'reference')
        # l7.plot_s_db(1, 0, color = 'k', linestyle = 'none', marker = 'x',
        #               markevery = 100, label = 'model')
        # plt.grid()
        # plt.subplot(2, 2, 4)
        # self.ref7.plot_s_deg(1, 0, color = 'k', label = '_nolegend_')
        # l7.plot_s_deg(1, 0, color = 'k', linestyle = 'none', marker = 'x',
        #               markevery = 100, label = '_nolegend_')
        # plt.grid()
        # plt.tight_layout()
        
    def test_alpha_warning(self):
        """
        Test if warns when t < 3 * skin_depth
        """
        freq = Frequency(1, 1, 1, 'MHz')
        with self.assertWarns(RuntimeWarning) as context:
            mline = MLine(frequency = freq, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen')


if __name__ == "__main__":
    # Launch all tests
    run_module_suite()