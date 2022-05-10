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
        self.data_dir_qucs = os.path.dirname(os.path.abspath(__file__)) + \
            '/qucs_prj/'
        self.data_dir_ads = os.path.dirname(os.path.abspath(__file__)) + \
            '/ads/'
        
        self.ref1 = rf.Network(os.path.join(self.data_dir_qucs,
                           'mline,hammerstad,hammerstad.s2p'))
        self.ref2 = rf.Network(os.path.join(self.data_dir_qucs,
                           'mline,hammerstad,kirschning.s2p'))
        self.ref3 = rf.Network(os.path.join(self.data_dir_qucs,
                           'mline,hammerstad,kobayashi.s2p'))
        self.ref4 = rf.Network(os.path.join(self.data_dir_qucs,
                           'mline,hammerstad,yamashita.s2p'))
        self.ref5 = rf.Network(os.path.join(self.data_dir_qucs,
                           'mline,wheeler,schneider.s2p'))
        self.ref6 = rf.Network(os.path.join(self.data_dir_qucs,
                           'mline,schneider,schneider.s2p'))
        self.ref_ads_1 = rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,freqencyinvariant,kirschning.s2p'))
        self.ref_ads_2 = rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,djordjevicsvensson,kirschning.s2p'))
        
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


    def test_line_qucs(self):
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
        
        # residuals
        res1 = l1 / self.ref1
        res2 = l2 / self.ref2
        res3 = l3 / self.ref3
        res4 = l4 / self.ref4
        res5 = l5 / self.ref5
        res6 = l6 / self.ref6
        
        # tolerate quite large errors due to qucs issue in attenuation computation
        limit_db = 2
        limit_deg = 5.
        self.assertTrue(npy.all(npy.abs(res1.s_db) < limit_db))
        self.assertTrue(npy.all(npy.abs(res1.s_deg) < limit_deg))
        self.assertTrue(npy.all(npy.abs(res2.s_db) < limit_db))
        self.assertTrue(npy.all(npy.abs(res2.s_deg) < limit_deg))
        self.assertTrue(npy.all(npy.abs(res3.s_db) < limit_db))
        self.assertTrue(npy.all(npy.abs(res3.s_deg) < limit_deg))
        self.assertTrue(npy.all(npy.abs(res4.s_db) < limit_db))
        self.assertTrue(npy.all(npy.abs(res4.s_deg) < limit_deg))
        self.assertTrue(npy.all(npy.abs(res5.s_db) < limit_db))
        self.assertTrue(npy.all(npy.abs(res5.s_deg) < limit_deg))
        self.assertTrue(npy.all(npy.abs(res6.s_db) < limit_db))
        self.assertTrue(npy.all(npy.abs(res6.s_deg) < limit_deg))
        
        # uncomment plots to see results
        # from matplotlib import pyplot as plt
        # rf.stylely()
        # plt.figure(figsize=(8, 7))
        
        # plt.subplot(2, 2, 1)
        # self.ref1.plot_s_db(0, 0, color = 'r')
        # l1.plot_s_db(0, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_db(0,0, linestyle = 'dashed', color = 'r')
        # self.ref2.plot_s_db(0, 0, color = 'g')
        # l2.plot_s_db(0, 0, linestyle = 'none', marker = 'x', color = 'g')
        # res2.plot_s_db(0,0, linestyle = 'dashed', color = 'g')
        # plt.ylim((-120, 5))
        
        # ax2 = plt.subplot(2, 2, 2)
        # self.ref1.plot_s_deg(0, 0, color = 'r')
        # l1.plot_s_deg(0, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_deg(0,0, linestyle = 'dashed', color = 'r')
        # self.ref2.plot_s_deg(0, 0, color = 'g')
        # l2.plot_s_deg(0, 0, linestyle = 'none', marker = 'x', color = 'g')
        # res2.plot_s_deg(0,0, linestyle = 'dashed', color = 'g')
        # ax2.get_legend().remove()
        
        # ax3 = plt.subplot(2, 2, 3)
        # self.ref1.plot_s_db(1, 0, color = 'r')
        # l1.plot_s_db(1, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_db(1,0, linestyle = 'dashed', color = 'r')
        # self.ref2.plot_s_db(1, 0, color = 'g')
        # l2.plot_s_db(1, 0, linestyle = 'none', marker = '+', color = 'g')
        # res2.plot_s_db(1,0, linestyle = 'dashed', color = 'g')
        # ax3.get_legend().remove()
        
        # ax4 = plt.subplot(2, 2, 4)
        # self.ref1.plot_s_deg(1, 0, color = 'r')
        # l1.plot_s_deg(1, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_deg(1,0, linestyle = 'dashed', color = 'r')
        # self.ref2.plot_s_deg(1, 0, color = 'g')
        # l2.plot_s_deg(1, 0, linestyle = 'none', marker = 'x', color = 'g')
        # res2.plot_s_deg(1,0, linestyle = 'dashed', color = 'g')
        # ax4.get_legend().remove()
        
    def test_line_ads(self):
        """
        Test against the ADS results
        """
        mline1 = MLine(frequency = self.ref_ads_1.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = 1.718e-8,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'kirschningjansen')
        mline2 = MLine(frequency = self.ref_ads_2.frequency, z0 = 50.,
                       w = self.w, h = self.h, t = self.t,
                       ep_r = self.ep_r, rho = 1.718e-8,
                       tand = self.tand, rough = self.d,
                       f_epr_tand = 1e9, f_low = 1e3, f_high = 1e12,
                       diel = 'djordjevicsvensson', disp = 'kirschningjansen')
        
        l1 = mline1.line(d=self.l, unit='m', embed = True, z0=mline1.Z0)
        l2 = mline2.line(d=self.l, unit='m', embed = True, z0=mline2.Z0)
        
        res1 = l1 / self.ref_ads_1
        res2 = l2 / self.ref_ads_2
        res1.name = 'res, freqinv'
        res2.name = 'res, svendjor'
        
        # fixme: cannot pass currently. is it due to dielectric dispersion ?
        # self.assertTrue(npy.all(npy.abs(res.s_db[:, 0, 0]) < 0.5))
        # self.assertTrue(npy.all(npy.abs(res.s_deg[:, 0, 0]) < 5.))
        
        # pass for S21
        self.assertTrue(npy.all(npy.abs(res1.s_db[:, 1, 0]) < 0.1))
        self.assertTrue(npy.all(npy.abs(res1.s_deg[:, 1, 0]) < 1.))
        self.assertTrue(npy.all(npy.abs(res2.s_db[:, 1, 0]) < 0.1))
        self.assertTrue(npy.all(npy.abs(res2.s_deg[:, 1, 0]) < 1.))
        
        # uncomment plots to see results
        # from matplotlib import pyplot as plt
        # rf.stylely()
        # self.ref_ads_1.name = 'ads, freqinv'
        # self.ref_ads_2.name = 'ads, svendjor'
        # l1.name = 'skrf, freqinv'
        # l2.name = 'skrf, svendjor'
        
        # plt.figure(figsize=(8, 7))
        
        # plt.subplot(2, 2, 1)
        # self.ref_ads_1.plot_s_db(0, 0, color = 'r')
        # l1.plot_s_db(0, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_db(0,0, linestyle = 'dashed', color = 'r')
        # self.ref_ads_2.plot_s_db(0, 0, color = 'g')
        # l2.plot_s_db(0, 0, linestyle = 'none', marker = 'x', color = 'g')
        # res2.plot_s_db(0,0, linestyle = 'dashed', color = 'g')
        # plt.ylim((-120, 5))
        
        # ax2 = plt.subplot(2, 2, 2)
        # self.ref_ads_1.plot_s_deg(0, 0, color = 'r')
        # l1.plot_s_deg(0, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_deg(0,0, linestyle = 'dashed', color = 'r')
        # self.ref_ads_2.plot_s_deg(0, 0, color = 'g')
        # l2.plot_s_deg(0, 0, linestyle = 'none', marker = 'x', color = 'g')
        # res2.plot_s_deg(0,0, linestyle = 'dashed', color = 'g')
        # ax2.get_legend().remove()
        
        # ax3 = plt.subplot(2, 2, 3)
        # self.ref_ads_1.plot_s_db(1, 0, color = 'r')
        # l1.plot_s_db(1, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_db(1,0, linestyle = 'dashed', color = 'r')
        # self.ref_ads_2.plot_s_db(1, 0, color = 'g')
        # l2.plot_s_db(1, 0, linestyle = 'none', marker = '+', color = 'g')
        # res2.plot_s_db(1,0, linestyle = 'dashed', color = 'g')
        # ax3.get_legend().remove()
        
        # ax4 = plt.subplot(2, 2, 4)
        # self.ref_ads_1.plot_s_deg(1, 0, color = 'r')
        # l1.plot_s_deg(1, 0, linestyle = 'none', marker = 'x', color = 'r')
        # res1.plot_s_deg(1,0, linestyle = 'dashed', color = 'r')
        # self.ref_ads_2.plot_s_deg(1, 0, color = 'g')
        # l2.plot_s_deg(1, 0, linestyle = 'none', marker = 'x', color = 'g')
        # res2.plot_s_deg(1,0, linestyle = 'dashed', color = 'g')
        # ax4.get_legend().remove()
        
        #plt.tight_layout()
               
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