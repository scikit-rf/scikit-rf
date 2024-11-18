import os
import unittest

import numpy as np

import skrf as rf
from skrf.media import DefinedAEpTandZ0

try:
    from matplotlib import pyplot as plt
    rf.stylely()
except ImportError:
    pass



class DefinedAEpTandZ0TestCase(unittest.TestCase):
    """
    Testcase for the DefinedAEpTandZ0 Media
    """
    def setUp(self):
        """
        Read in all the network data required for tests
        """
        self.data_dir_awr = os.path.dirname(os.path.abspath(__file__)) + \
            '/awr/'

        self.ref_awr = [
            {'color': 'k',
             'n': rf.Network(os.path.join(self.data_dir_awr,
                               'tlinp.s2p'))},
            ]

        # default parameter set for tests
        self.verbose = False # output comparison plots if True
        self.Zn   = 41.0635803351402
        self.A    = 0.0670188722315605
        self.l    = 74.2415154883262e-3
        self.ep_r = 3.25354286428265
        self.tand = 0.0133936758939493
        # frequency vector with only 3 points
        self.freq = rf.Frequency(1, 3, 3, 'GHz')


    def test_line_awr(self):
        """
        Test against the AWR results
        """
        if self.verbose:
            fig, axs = plt.subplots(2, 2, figsize = (8,6))
            fig.suptitle('awr/skrf')
            fig2, axs2 = plt.subplots(2, 2, figsize = (8,6))
            fig2.suptitle('awr/skrf residuals')

        limit_db = 1e-5
        limit_deg = 1e-4

        for ref in self.ref_awr:
            m = DefinedAEpTandZ0(
                frequency = ref['n'].frequency,
                z0 = self.Zn,
                ep_r = self.ep_r,
                A = self.A,
                f_A = ref['n'].frequency.f[0],
                tanD = self.tand,
                z0_port = 50
                )
            line = m.line(d=self.l, unit='m')
            line.name = 'skrf,awr'

            # residuals
            res = line / ref['n']
            res.name = 'residuals ' + ref['n'].name

            # test if within limit lines
            self.assertTrue(
                np.all(np.abs(res.s_db[:, 0, 0]) < limit_db))
            self.assertTrue(
                np.all(np.abs(res.s_deg[:, 0, 0]) < limit_deg))
            self.assertTrue(np.all(np.abs(res.s_db[:, 1, 0]) < limit_db))
            self.assertTrue(np.all(np.abs(res.s_deg[:, 1, 0]) < limit_deg))

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
            fig.tight_layout()
            fig2.tight_layout()

    def test_nominal_impedance_dispersion(self):
        """
        Test if passing scalar value to z0 results to this nominal impedance
        being dispersed to a complex frequency-dependent characteristic
        impedance.
        """
        Zn = 75
        m = DefinedAEpTandZ0(
            frequency = self.freq,
            z0 = Zn,
            ep_r = self.ep_r,
            A = self.A,
            f_A = self.freq.f[0],
            tanD = self.tand,
            z0_port = 50
            )
        # z0_characteristic complex and frequency dependant
        self.assertTrue(np.all(np.abs(np.imag(m.z0_characteristic) > 1e-1)))
        self.assertTrue(m.z0_characteristic[0] != m.z0_characteristic[-1])

    def test_raw_z0_array(self):
        """
        Test if passing array-like value to z0 results to this raw value being
        assigned to characteristic impedance.
        """

        # Single-element array are kept as raw z0_characteristic
        z1 = [75]
        m1 = DefinedAEpTandZ0(
            frequency = self.freq,
            z0 = z1,
            ep_r = self.ep_r,
            A = self.A,
            f_A = self.freq.f[0],
            tanD = self.tand,
            z0_port = 50
            )
        self.assertTrue(np.allclose(z1 * np.ones(3), m1.z0_characteristic))
        # Sequence are kept as raw z0_characteristic
        z2 = [45, 50, 55]
        m2 = DefinedAEpTandZ0(
            frequency = self.freq,
            z0 = z2,
            ep_r = self.ep_r,
            A = self.A,
            f_A = self.freq.f[0],
            tanD = self.tand,
            z0_port = 50
            )
        self.assertTrue(np.allclose(z2, m2.z0_characteristic))
        # Numpy arrays are kept as raw z0_characteristic
        z3 = np.array([48, 50, 52])
        m3 = DefinedAEpTandZ0(
            frequency = self.freq,
            z0 = z3,
            ep_r = self.ep_r,
            A = self.A,
            f_A = self.freq.f[0],
            tanD = self.tand,
            z0_port = 50
            )
        self.assertTrue(np.allclose(z3, m3.z0_characteristic))

if __name__ == '__main__':
    unittest.main()
