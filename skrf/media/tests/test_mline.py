# -*- coding: utf-8 -*-
import unittest
import os
import numpy as npy

from skrf.media import MLine
from skrf.network import Network
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

        l1 = mline1.line(d=self.l, unit='m', embed = True, z0=mline1.Z0)
        l2 = mline2.line(d=self.l, unit='m', embed = True, z0=mline2.Z0)
        l3 = mline3.line(d=self.l, unit='m', embed = True, z0=mline3.Z0)
        l4 = mline4.line(d=self.l, unit='m', embed = True, z0=mline4.Z0)
        
        self.assertTrue(npy.all(npy.abs(l1.s_db - self.ref1.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l1.s_deg - self.ref1.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l2.s_db - self.ref2.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l2.s_deg - self.ref2.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l3.s_db - self.ref3.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l3.s_deg - self.ref3.s_deg) < 1.))
        self.assertTrue(npy.all(npy.abs(l4.s_db - self.ref4.s_db) < 0.1))
        self.assertTrue(npy.all(npy.abs(l4.s_deg - self.ref4.s_deg) < 1.))
        
        # uncomment plots to see results
        # from matplotlib import pyplot as plt
        # plt.figure(figsize=(10, 8))
        # plt.subplot(2, 1, 1)
        # self.ref1.plot_s_db(0, 0, color = 'r', label = '_nolegend_')
        # self.ref2.plot_s_db(0, 0, color = 'g', label = '_nolegend_')
        # self.ref3.plot_s_db(0, 0, color = 'b', label = '_nolegend_')
        # self.ref4.plot_s_db(0, 0, color = 'c', label = '_nolegend_')
        # l1.plot_s_db(0, 0, color = 'r', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # l2.plot_s_db(0, 0, color = 'g', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # l3.plot_s_db(0, 0, color = 'b', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # l4.plot_s_db(0, 0, color = 'c', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # plt.grid()
        # plt.subplot(2, 1, 2)
        # self.ref1.plot_s_db(1, 0, color = 'r')
        # self.ref2.plot_s_db(1, 0, color = 'g')
        # self.ref3.plot_s_db(1, 0, color = 'b')
        # self.ref4.plot_s_db(1, 0, color = 'c')
        # l1.plot_s_db(1, 0, color = 'r', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # l2.plot_s_db(1, 0, color = 'g', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # l3.plot_s_db(1, 0, color = 'b', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
        # l4.plot_s_db(1, 0, color = 'c', linestyle = 'none', marker = 'x',
        #              label = '_nolegend_')
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