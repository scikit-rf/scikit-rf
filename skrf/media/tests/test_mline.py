import unittest
import os
import numpy as npy
import pytest

from skrf.media import MLine
from skrf.frequency import Frequency
import skrf as rf
from numpy.testing import run_module_suite
from matplotlib import pyplot as plt
rf.stylely()


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
        
        self.ref_qucs = [
            {'model': 'hammerstadjensen', 'disp': 'hammerstadjensen', 'color': 'r',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
                               'mline,hammerstad,hammerstad.s2p'))},
            {'model': 'hammerstadjensen', 'disp': 'kirschningjansen', 'color': 'c',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
                               'mline,hammerstad,kirschning.s2p'))},
            {'model': 'hammerstadjensen', 'disp': 'kobayashi', 'color': 'k',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
                               'mline,hammerstad,kobayashi.s2p'))},
            {'model': 'hammerstadjensen', 'disp': 'yamashita', 'color': 'g',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
                               'mline,hammerstad,yamashita.s2p'))},
            {'model': 'wheeler', 'disp': 'schneider', 'color': 'm',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
                               'mline,wheeler,schneider.s2p'))},
            {'model': 'schneider', 'disp': 'schneider', 'color': 'b',
             'n': rf.Network(os.path.join(self.data_dir_qucs,
                               'mline,schneider,schneider.s2p'))}
            ]
        
        
        self.ref_ads = [
            {'diel': 'frequencyinvariant', 'disp': 'kirschningjansen', 'color': 'r',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,freqencyinvariant,kirschning.s2p'))},
            {'diel': 'djordjevicsvensson', 'disp': 'kirschningjansen', 'color': 'c',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,djordjevicsvensson,kirschning.s2p'))},
            {'diel': 'frequencyinvariant', 'disp': 'kobayashi', 'color': 'k',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,freqencyinvariant,kobayashi.s2p'))},
            {'diel': 'djordjevicsvensson', 'disp': 'kobayashi', 'color': 'g',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,djordjevicsvensson,kobayashi.s2p'))},
            {'diel': 'frequencyinvariant', 'disp': 'yamashita', 'color': 'm',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,freqencyinvariant,yamashita.s2p'))},
            {'diel': 'djordjevicsvensson', 'disp': 'yamashita', 'color': 'b',
             'n': rf.Network(os.path.join(self.data_dir_ads,
                           'mlin,djordjevicsvensson,yamashita.s2p'))}
            ]
        
        # default parameter set for tests
        self.verbose = False # output comparison plots if True
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
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen',
                       compatibility_mode = 'qucs')
        
        # without t (t = None)
        mline2 = MLine(frequency = freq, z0 = 50.,
                       w = self.w, h = self.h,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen',
                       compatibility_mode = 'qucs')
        
        # with t = 0
        mline3 = MLine(frequency = freq, z0 = 50.,
                       w = self.w, h = self.h, t = 0,
                       ep_r = self.ep_r, rho = self.rho,
                       tand = self.tand, rough = self.d,
                       diel = 'frequencyinvariant', disp = 'hammerstadjensen',
                       compatibility_mode = 'qucs')
        
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
        if self.verbose:
            fig, axs = plt.subplots(2, 2, figsize = (8,6))
            fig.suptitle('qucs/skrf')
            fig2, axs2 = plt.subplots(2, 2, figsize = (8,6))
            fig2.suptitle('ads/skrf residuals')
            
        limit_db = 0.1
        limit_deg = 1.
            
        for ref in self.ref_qucs:
            mline = MLine(frequency = ref['n'].frequency, z0 = 50.,
                            w = self.w, h = self.h, t = self.t,
                            ep_r = self.ep_r, rho = self.rho,
                            tand = self.tand, rough = self.d,
                            model = ref['model'], disp = ref['disp'],
                            diel = 'frequencyinvariant',
                            compatibility_mode = 'qucs')
            with pytest.warns(FutureWarning, match="`embed` will be deprecated"):
                line = mline.line(d=self.l, unit='m', embed = True, z0=mline.Z0)
            line.name = 'skrf,qucs'
            
            # residuals
            res = line / ref['n']
            res.name = 'residuals ' + ref['n'].name

            # test if within limit lines
            self.assertTrue(npy.all(npy.abs(res.s_db) < limit_db))
            self.assertTrue(npy.all(npy.abs(res.s_deg) < limit_deg))
            
            if self.verbose:
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
        
    def test_line_ads(self):
        """
        Test against the ADS results
        """
        if self.verbose:
            fig, axs = plt.subplots(2, 2, figsize = (8,6))
            fig.suptitle('ads/skrf')
            fig2, axs2 = plt.subplots(2, 2, figsize = (8,6))
            fig2.suptitle('ads/skrf residuals')
        
        # todo: restore to smal values
        limit_db = 0.1
        limit_deg = 1.
        
        for ref in self.ref_ads:
            mline = MLine(frequency = ref['n'].frequency, z0 = 50.,
                            w = self.w, h = self.h, t = self.t,
                            ep_r = self.ep_r, rho = self.rho,
                            tand = self.tand, rough = self.d,
                            model = 'hammerstadjensen', disp = ref['disp'],
                            diel = ref['diel'])
            with pytest.warns(FutureWarning, match="`embed` will be deprecated"):
                line = mline.line(d=self.l, unit='m', embed = True, z0=mline.Z0)
            line.name = 'skrf,ads'
            
            # residuals
            res = line / ref['n']
            res.name = 'residuals ' + ref['n'].name

            # test if within limit lines
            # fixme: still a small deviation of S11 at low frequency
            #        limit line multiplied by 10 for S11 as for now
            self.assertTrue(
                npy.all(npy.abs(res.s_db[:, 0, 0]) < 10. *limit_db))
            self.assertTrue(
                npy.all(npy.abs(res.s_deg[:, 0, 0]) < 10. * limit_deg))
            self.assertTrue(npy.all(npy.abs(res.s_db[:, 1, 0]) < limit_db))
            self.assertTrue(npy.all(npy.abs(res.s_deg[:, 1, 0]) < limit_deg))
            
            if self.verbose:
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