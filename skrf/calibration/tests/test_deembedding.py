import os
import unittest
import warnings

import numpy as np
import pytest
from scipy.constants import c, pi

import skrf as rf
from skrf.calibration.deembedding import (
    _MIXED_MODE_TV,
    IEEEP370_FD_QM,
    IEEEP370_TD_QM,
    AdmittanceCancel,
    IEEEP370_MM_NZC_2xThru,
    IEEEP370_SE_NZC_2xThru,
    IEEEP370_SE_ZC_2xThru,
    ImpedanceCancel,
    Modal_NZC_2xThru,
    Open,
    OpenShort,
    Short,
    ShortOpen,
    SplitPi,
    SplitTee,
    _abcd_2nport,
    _eig_modal_transform,
    _modal2se,
    _modal_references,
    _modal_xi,
    _se2modal,
)
from skrf.media import MLine
from skrf.network import concat_ports


class DeembeddingTestCase(unittest.TestCase):
    """
    Testcase for the Deembedding class
    Pseudo-netlists for s-parameter files used in these tests
    For open-short de-embedding:
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
    For open de-embedding:
    - deemb_ind7.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Dut_ind (1 2) inductor l=1nH
        Cpad2   (2 0) capacitor c=25fF
        Cp2p    (1 2) capacitor c=10fF
        P2      (2 0) port
    - deemb_open7.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Cpad2   (2 0) capacitor c=25fF
        Cp2p    (1 2) capacitor c=10fF
        P2      (2 0) port
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
    For short de-embedding:
    - deemb_ind8.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Dut_ind (2 3) inductor l=1nH
        Rline2  (3 4) resistor r=2ohm
        P2      (4 0) port
    - deemb_short8.s2p
        P1      (1 0) port
        Rline1  (1 0) resistor r=2ohm
        Rline2  (0 4) resistor r=2ohm
        P2      (4 0) port
    For SplitPi de-embedding:
    - deemb_ind3.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Rline1  (1 2) resistor r=2ohm
        Dut_ind (2 3) inductor l=1nH
        Rline2  (3 4) resistor r=2ohm
        Cpad2   (4 0) capacitor c=25fF
        P2      (4 0) port
    - deemb_thru3.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Rline1  (1 2) resistor r=2ohm
        Rline2  (3 4) resistor r=2ohm
        Cpad2   (4 0) capacitor c=25fF
        P2      (4 0) port
    For SplitTee de-embedding:
    - deemb_ind4.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Cpad1   (2 0) capacitor c=25fF
        Dut_ind (2 3) inductor l=1nH
        Cpad2   (3 0) capacitor c=25fF
        Rline2  (3 4) resistor r=2ohm
        P2      (4 0) port
    - deemb_thru4.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Cpad1   (2 0) capacitor c=25fF
        Cpad2   (3 0) capacitor c=25fF
        Rline2  (3 4) resistor r=2ohm
        P2      (4 0) port
    For AdmittanceCancel de-embedding:
    - deemb_ind5.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Dut_ind (1 2) inductor l=1nH
        Cpad2   (2 0) capacitor c=25fF
        P2      (2 0) port
    - deemb_thru5.s2p
        P1      (1 0) port
        Cpad1   (1 0) capacitor c=25fF
        Cpad2   (1 0) capacitor c=25fF
        P2      (1 0) port
    For ImpedanceCancel de-embedding:
    - deemb_ind6.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Dut_ind (2 3) inductor l=1nH
        Rline2  (3 4) resistor r=2ohm
        P2      (4 0) port
    - deemb_thru6.s2p
        P1      (1 0) port
        Rline1  (1 2) resistor r=2ohm
        Rline2  (2 3) resistor r=2ohm
        P2      (3 0) port
    """

    def setUp(self):
        """
        Read in all the network data required for tests
        """
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/deembed/'

        # for open-short testing
        self.raw = rf.Network(os.path.join(self.test_dir, 'deemb_ind.s2p'))
        self.open = rf.Network(os.path.join(self.test_dir, 'deemb_open.s2p'))
        self.short = rf.Network(os.path.join(self.test_dir, 'deemb_short.s2p'))

        # for spot frequency checking
        self.raw_1f = self.raw['10GHz']
        self.open_1f = self.open['10GHz']
        self.short_1f = self.short['10GHz']

        # for open testing
        self.raw7 = rf.Network(os.path.join(self.test_dir, 'deemb_ind7.s2p'))
        self.open7 = rf.Network(os.path.join(self.test_dir, 'deemb_open7.s2p'))

        # for spot frequency checking
        self.raw7_1f = self.raw7['10GHz']
        self.open7_1f = self.open7['10GHz']

        # for short-open testing
        self.raw2 = rf.Network(os.path.join(self.test_dir, 'deemb_ind2.s2p'))
        self.open2 = rf.Network(os.path.join(self.test_dir, 'deemb_open2.s2p'))
        self.short2 = rf.Network(os.path.join(self.test_dir, 'deemb_short2.s2p'))

        # for spot frequency checking
        self.raw2_1f = self.raw2['10GHz']
        self.open2_1f = self.open2['10GHz']
        self.short2_1f = self.short2['10GHz']

        # for short testing
        self.raw8 = rf.Network(os.path.join(self.test_dir, 'deemb_ind8.s2p'))
        self.short8 = rf.Network(os.path.join(self.test_dir, 'deemb_short8.s2p'))

        # for spot frequency checking
        self.raw8_1f = self.raw8['10GHz']
        self.short8_1f = self.short8['10GHz']

        # for spilit-pi testing
        self.raw3 = rf.Network(os.path.join(self.test_dir, 'deemb_ind3.s2p'))
        self.thru3 = rf.Network(os.path.join(self.test_dir, 'deemb_thru3.s2p'))

        # for spot frequency checking
        self.raw3_1f = self.raw3['10GHz']
        self.thru3_1f = self.thru3['10GHz']

        # for spilit-tee testing
        self.raw4 = rf.Network(os.path.join(self.test_dir, 'deemb_ind4.s2p'))
        self.thru4 = rf.Network(os.path.join(self.test_dir, 'deemb_thru4.s2p'))

        # for spot frequency checking
        self.raw4_1f = self.raw4['10GHz']
        self.thru4_1f = self.thru4['10GHz']

        # for admittance cancel testing
        self.raw5 = rf.Network(os.path.join(self.test_dir, 'deemb_ind5.s2p'))
        self.thru5 = rf.Network(os.path.join(self.test_dir, 'deemb_thru5.s2p'))

        # for spot frequency checking
        self.raw5_1f = self.raw5['10GHz']
        self.thru5_1f = self.thru5['10GHz']

        # for impedance cancel cancel testing
        self.raw6 = rf.Network(os.path.join(self.test_dir, 'deemb_ind6.s2p'))
        self.thru6 = rf.Network(os.path.join(self.test_dir, 'deemb_thru6.s2p'))

        # for spot frequency checking
        self.raw6_1f = self.raw6['10GHz']
        self.thru6_1f = self.thru6['10GHz']

        # create de-embedding objects
        self.dm = OpenShort(self.open, self.short)
        self.dm_os = OpenShort(self.open_1f, self.short_1f)
        self.dm_o = Open(self.open7_1f)
        self.dm_so = ShortOpen(self.short2_1f, self.open2_1f)
        self.dm_s = Short(self.short8_1f)
        self.dm_pi = SplitPi(self.thru3_1f)
        self.dm_tee = SplitTee(self.thru4_1f)
        self.dm_ac = AdmittanceCancel(self.thru5_1f)
        self.dm_ic = ImpedanceCancel(self.thru6_1f)

        # relative tolerance for comparisons
        self.rtol = 1e-6

    def test_freqmismatch(self):
        """
        Check that a warning is caught when networks are of different frequencies
        """
        with self.assertWarns(RuntimeWarning):
            OpenShort(self.open, self.short[0:len(self.short)//2])

        with self.assertWarns(RuntimeWarning):
            self.dm.deembed(self.raw[0:len(self.raw)//2])

    def test_openshort(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_os.deembed(self.raw_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_open(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_o.deembed(self.raw7_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_shortopen(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_so.deembed(self.raw2_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_short(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_s.deembed(self.raw8_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_splitpi(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_pi.deembed(self.raw3_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_splittee(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_tee.deembed(self.raw4_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_admittance_cancel(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_ac.deembed(self.raw5_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_impedance_cancel(self):
        """
        After de-embedding, the network is a pure inductor of 1nH.
        Test that this is true at a spot frequency.
        """
        dut = self.dm_ic.deembed(self.raw6_1f)
        ind_calc = 1e9*np.imag(1/dut.y[0,0,0])/2/np.pi/dut.f
        self.assertTrue(np.isclose(ind_calc, 1, rtol=self.rtol))

    def test_IEEEP370_SE_NZC_2xThru(self):
        """
        Test test_IEEEP370_SE_NZC_2xThru.

        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        Test that this thru has S21 amplitude and phase smaller than a limit.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        dm_nzc = IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru,
                                        name = '2xthru')
        residuals = dm_nzc.deembed(s2xthru)
        # insertion loss magnitude deviate from 1.0 from less than 0.1 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.1, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 1 degree
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 1.0, 'residual IL Phase')

    def test_IEEEP370_SE_NZC_2xThru_with_dc(self):
        """
        Test test_IEEEP370_SE_NZC_2xThru_with_dc.

        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        Test that this thru has S21 amplitude and phase smaller than a limit.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        # interpolate to dc
        s2xthru_dc = s2xthru.extrapolate_to_dc(kind='linear')
        with pytest.warns(RuntimeWarning, match="DC point detected"):
            dm_nzc = IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru_dc,
                                        name = '2xthru')
        residuals = dm_nzc.deembed(s2xthru_dc)
        # insertion loss magnitude deviate from 1.0 from less than 0.1 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.1, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 1 degree
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 1.0, 'residual IL Phase')

    def test_IEEEP370_SE_NZC_2xThru_with_interpolation(self):
        """
        Test test_IEEEP370_SE_NZC_2xThru_with_interpolation.
        While interpolation should be avoided, it keep useful in some cases,
        e.g. when the measurement frequency axis was not well formatted but
        a result is still needed and some deviation acceptable.
        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        Test that this thru has S21 amplitude and phase smaller than a limit.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        # with non-hormonic sweep
        nonuniform_freq = rf.Frequency(s2xthru.f[0], s2xthru.f[-1],
                                       npoints=len(s2xthru)-10, unit='Hz')
        s2xthru_nu = s2xthru.interpolate(nonuniform_freq)
        with pytest.warns(RuntimeWarning, match="Non-uniform frequency vector detected"):
            dm_nzc_nu = IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru_nu,
                                                name = '2xthru')
        residuals = dm_nzc_nu.deembed(s2xthru_nu)
        # insertion loss magnitude deviate from 1.0 from less than 0.1 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.1, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 1 degree
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 1.0, 'residual IL Phase')

    def test_IEEEP370_SE_ZC_2xThru(self):
        """
        Test test_IEEEP370_SE_ZC_2xThru.

        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        Test that this thru has S21 amplitude and phase smaller than a limit.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        fdf = rf.Network(os.path.join(self.test_dir, 'fdf.s2p'))
        dm_zc  = IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru,
                                       dummy_fix_dut_fix = fdf,
                                       bandwidth_limit = 10e9,
                                       pullback1 = 0, pullback2 = 0,
                                       leadin = 0,
                                       NRP_enable = False,
                                       name = 'zc2xthru')
        residuals = dm_zc.deembed(s2xthru)
        # insertion loss magnitude deviate from 1.0 from less than 0.2 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.2, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 45 degree
        # too much tolerance here allowed as for now
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 2.0, 'residual IL Phase')

    def test_IEEEP370_SE_ZC_2xThru_with_dc(self):
        """
        Test test_IEEEP370_SE_ZC_2xThru_with_dc.

        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        Test that this thru has S21 amplitude and phase smaller than a limit.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        fdf = rf.Network(os.path.join(self.test_dir, 'fdf.s2p'))
        s2xthru_dc = s2xthru.extrapolate_to_dc(kind='linear')
        fdf_dc = fdf.extrapolate_to_dc(kind='linear')
        with pytest.warns(RuntimeWarning, match="DC point detected"):
            dm_zc  = IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru_dc,
                                        dummy_fix_dut_fix = fdf_dc,
                                        bandwidth_limit = 10e9,
                                        pullback1 = 0, pullback2 = 0,
                                        leadin = 0,
                                        NRP_enable = False,
                                        name = 'zc2xthru')
        residuals = dm_zc.deembed(s2xthru_dc)
        # insertion loss magnitude deviate from 1.0 from less than 0.2 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.2, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 45 degree
        # too much tolerance here allowed as for now
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 2.0, 'residual IL Phase')

    def test_IEEEP370_SE_ZC_2xThru_with_interpolation(self):
        """
        Test test_IEEEP370_SE_ZC_2xThru_with_interpolation.
        While interpolation should be avoided, it keep useful in some cases,
        e.g. when the measurement frequency axis was not well formatted but
        a result is still needed and some deviation acceptable.
        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        Test that this thru has S21 amplitude and phase smaller than a limit.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        fdf = rf.Network(os.path.join(self.test_dir, 'fdf.s2p'))
        # with non-hormonic sweep
        nonuniform_freq = rf.Frequency(s2xthru.f[0], s2xthru.f[-1],
                                       npoints=len(s2xthru)-10, unit='Hz')
        s2xthru_nu = s2xthru.interpolate(nonuniform_freq)
        fdf_nu = fdf.interpolate(nonuniform_freq)
        with pytest.warns(RuntimeWarning, match="Non-uniform frequency vector detected"):
            dm_zc_nu  = IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru_nu,
                                        dummy_fix_dut_fix = fdf_nu,
                                        bandwidth_limit = 10e9,
                                        pullback1 = 0, pullback2 = 0,
                                        leadin = 0,
                                        NRP_enable = False,
                                        name = 'zc2xthru')
        residuals = dm_zc_nu.deembed(s2xthru_nu)
        # insertion loss magnitude deviate from 1.0 from less than 0.2 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.2, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 45 degree
        # too much tolerance here allowed as for now
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 2.0, 'residual IL Phase')

    def test_IEEEP370_SE_NZC_2xThru_no_180deg_phase_jumps(self):
        """
        Test test_IEEEP370_SE_NZC_2xThru_no_180deg_phase_jumps.

        After de-embedding fixtures model from 2xtru, the network is a perfect
        thru.
        If there is noise on the 2xthru phase, incorrect computation of e01 and
        e10 could lead to 180° jumps into the side models s21 phase.
        Check this not happens.
        """
        s2xthru = rf.Network(os.path.join(self.test_dir, 's2xthru.s2p'))
        # with phase noise
        # add enough phase noise to trigger phase jumps with original
        # implementation, but small enough to keep within 1° limit line
        s2xthru_pn = s2xthru.copy()
        s2xthru_pn.add_noise_polar(0.0002, 0.2)
        dm_nzc_pn = IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru_pn,
                                        name = '2xthru')
        residuals = dm_nzc_pn.deembed(s2xthru_pn)
        # insertion loss magnitude deviate from 1.0 from less than 0.1 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.1, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 1 degree
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 1.0, 'residual IL Phase')

    def test_IEEEP370_check_fd_td_se_quality(self):
        """
        Test test_IEEEP370_check_fd_se_quality.

        Check for passivity, reciprocity, and causality in the frequency
        and in the time domains.
        """
        freq = rf.Frequency(10e-3, 10, 1000, 'GHz')
        W   = 3.00e-3
        H   = 1.55e-3
        T   = 50e-6
        ep_r = 4.459
        tanD = 0.0183
        f_epr_tand = 1e9

        beta  = 2 * pi * freq.f / c # propagation in air
        # 50 ohm air line
        m50 = rf.media.DefinedGammaZ0(frequency = freq, z0_port = 50, gamma = 1j * beta)

        # microstrip segments
        with pytest.warns(RuntimeWarning, match = r"^Conductor"):
            m = MLine(frequency=freq, z0_port=50, w=W, h=H, t=T,
                    ep_r=ep_r, mu_r=1, rho=1.712e-8, tand=tanD, rough=0.15e-6,
                    f_low=1e3, f_high=1e12, f_epr_tand=f_epr_tand,
                    diel='djordjevicsvensson', disp='kirschningjansen')

        thru  = m.line(0.050, 'm', z0 = 52.5)
        thru.name = "thru"
        # perfect data
        fd_qm = IEEEP370_FD_QM()
        td_qm = IEEEP370_TD_QM(1e9, # bps
                                  32,  # samples per UI
                                  0.4, # rise time per UI
                                  1,   # gaussian pulse
                                  2,   # zero extrapolation
                                  verbose = False)
        qm = fd_qm.check_se_quality(thru)
        self.assertTrue(qm['passivity']['value'] > 99.9 and
                        qm['reciprocity']['value'] > 99.9 and
                        qm['causality']['value'] > 99.9,
                        'FD quality perfect thru')
        # passivity violation (reference value from Matlab R2024a)
        thru_non_passive = thru.copy()
        thru_non_passive.s[:, 1, 0] = 1.137 * thru_non_passive.s[:, 1, 0]
        thru_non_passive.s[:, 0, 1] = 1.137 * thru_non_passive.s[:, 0, 1]
        qm = fd_qm.check_se_quality(thru_non_passive)
        self.assertTrue(np.round(qm['passivity']['value'], 4) == 49.7701,
                        'FD quality passivity violation')
        qm = td_qm.check_se_quality(thru_non_passive)
        self.assertTrue(np.round(qm['passivity']['value'], 4) == 71.3500,
                        'TD quality passivity violation')
        # reciprocity violation (reference value from Matlab R2024a)
        thru_non_reciprocal = thru.copy()
        thru_non_reciprocal.s[:, 1, 0] = 0.945 * thru_non_reciprocal.s[:, 0, 1]
        qm = fd_qm.check_se_quality(thru_non_reciprocal)
        self.assertTrue(np.round(qm['reciprocity']['value'], 4) == 49.6120,
                        'FD quality reciprocity violation')
        qm = td_qm.check_se_quality(thru_non_reciprocal)
        self.assertTrue(np.round(qm['reciprocity']['value'], 4) == 28.4500,
                        'TD quality reciprocity violation')
        # causality violation (reference value from Matlab R2024a)
        half_fd  = m50.line(0.0445, 'm', z0 = 52)
        thru_non_causal_fd = half_fd.inv ** thru ** half_fd.inv
        thru_non_causal_fd.name = 'thru'
        qm = fd_qm.check_se_quality(thru_non_causal_fd)
        self.assertTrue(np.round(qm['causality']['value'], 4) == 48.2637,
                        'FD quality causality violation')
        half  = m50.line(0.18, 'm', z0 = 52)
        thru_non_causal = half.inv ** thru ** half.inv
        thru_non_causal.name = 'thru'
        qm = td_qm.check_se_quality(thru_non_causal)
        self.assertTrue(np.round(qm['causality']['value'], 4) == 10.0500,
                        'TD quality causality violation')


# ---------------------------------------------------------------------------
# Synthetic N coupled lines, for Modal_NZC_2xThru
# ---------------------------------------------------------------------------

def _modal_line_s(gamma, zc, length, z_ref):
    """
    2N-port ('second' order) S-parameters of N uncoupled lines, one for each mode.

    gamma: (F, N) propagation constants, zc: (N,) characteristic impedances,
    z_ref: (N,) reference impedances of the modal ports.
    """
    gamma = np.asarray(gamma)
    n_f, n = gamma.shape
    zc = np.asarray(zc, dtype=float)
    z_ref = np.asarray(z_ref, dtype=float)
    refl = (zc - z_ref) / (zc + z_ref)
    e = np.exp(-gamma * length)
    den = 1 - refl**2 * e**2
    s = np.zeros((n_f, 2 * n, 2 * n), dtype=complex)
    for k in range(n):
        s11 = refl[k] * (1 - e[:, k]**2) / den[:, k]
        s21 = e[:, k] * (1 - refl[k]**2) / den[:, k]
        s[:, k, k] = s[:, n + k, n + k] = s11
        s[:, k, n + k] = s[:, n + k, k] = s21
    return s


def _mtl(freq, tv, eps_eff, alpha, zc, length, z0=50):
    """
    Uniform coupled line in terminal S-parameters, built from its modes.

    tv: (N, N) or (F, N, N) modal transform, eps_eff and alpha: (N,) effective permittivity and
    conductor loss (Np/m at 1 GHz, growing as sqrt(f)), zc: (N,) modal characteristic impedance.
    """
    f = freq.f
    n = np.shape(tv)[-1]
    tv = np.broadcast_to(tv, (len(f), n, n))
    gamma = np.stack([alpha[k] * np.sqrt(f / 1e9) + 1j * 2 * pi * f * np.sqrt(eps_eff[k]) / c for k in range(n)],
                     axis=1)
    s_m = _modal_line_s(gamma, zc, length, zc)
    return rf.Network(frequency=freq, s=_modal2se(s_m, _modal_xi(tv, z0, zc)), z0=z0)


_MTL_FREQ = rf.Frequency(10, 10000, 1000, 'MHz')
_MTL_TV0 = {
    2: np.array([[1.0, 0.5], [1.0, -0.5]]),
    3: np.array([[1.0, 0.8, 0.3], [0.6, -0.1, -1.0], [1.0, -0.9, 0.4]]),
    4: np.array([[1.0, 0.8, 0.3, 0.2], [0.6, -0.1, -1.0, 0.5], [1.0, -0.9, 0.4, -0.7], [0.3, 0.6, 0.5, 1.0]]),
}
_MTL_EPS = {2: [3.0, 3.5], 3: [3.2, 3.6, 4.0], 4: [3.0, 3.3, 3.7, 4.1]}
_MTL_ALPHA = {2: [0.5, 0.8], 3: [0.5, 0.7, 0.9], 4: [0.5, 0.6, 0.8, 0.9]}


def _coupled_case(n_lines=3, mismatch=1.0, tv_slope=0.0, launch=None, freq=None):
    """
    Fixture, DUT, 2x-thru and FIX-DUT-FIX of N coupled lines with known modes ('second' port order).

    mismatch scales the modal impedances of the fixture, tv_slope makes the modal transform of the
    fixture and of the DUT frequency dependent, launch adds a mode converting launch for N = 2
    ('skew': a line of one conductor only, 'cap': a shunt capacitor on one conductor).
    """
    freq = _MTL_FREQ if freq is None else freq
    f = freq.f
    tv0 = _MTL_TV0[n_lines]
    slope = tv_slope * np.roll(np.eye(n_lines), 1, axis=0)
    ramp = (f / f[-1])[:, None, None]
    tv = tv0[None] + ramp * slope[None]
    zc = _modal_references(tv, 50.0) * mismatch
    line = _mtl(freq, tv, _MTL_EPS[n_lines], _MTL_ALPHA[n_lines], zc, 0.025)
    if launch is None:
        fix = line
    else:
        m50 = rf.media.DefinedGammaZ0(freq, z0_port=50, z0=50, gamma=1j * 2 * pi * f * np.sqrt(3.2) / c)
        if launch == 'skew':
            leg = m50.line(0.0015, 'm')
        else:
            leg = m50.shunt_capacitor(0.15e-12)
        fix = concat_ports([leg, m50.thru()], port_order='second') ** line
    tv_dut = tv0[None] + 0.3 * np.eye(n_lines)[None] + ramp * slope[None]
    dut = _mtl(freq, tv_dut, [e * 1.05 for e in _MTL_EPS[n_lines]], [0.3] * n_lines,
               _modal_references(tv_dut, 50.0) * 0.9, 0.01)
    s2x = fix ** fix.flipped()
    fdf = fix ** dut ** fix.flipped()
    return {'n': n_lines, 'tv': tv, 'fix': fix, 'dut': dut, 's2x': s2x, 'fdf': fdf}


def _add_noise(ntwk, rng, mag_dev=0.0002, phase_dev_deg=0.2):
    """
    Complex noise of a given standard deviation in magnitude and phase, like Network.add_noise_polar
    but with its own random generator, so that tests are repeatable.
    """
    out = ntwk.copy()
    mag = np.abs(out.s) + rng.normal(0.0, mag_dev, out.s.shape)
    phase = np.angle(out.s) + np.deg2rad(rng.normal(0.0, phase_dev_deg, out.s.shape))
    out.s = mag * np.exp(1j * phase)
    return out


_REAL_PAIR_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..',
                              'doc', 'source', 'examples', 'networktheory', 'ieeep370deembedding')


def _assert_is_thru(ntwk, atol=1e-9):
    """The network is N ideal thru connections (2N ports, 'second' order) at every frequency."""
    n = ntwk.nports // 2
    thru = np.block([[np.zeros((n, n)), np.eye(n)], [np.eye(n), np.zeros((n, n))]])
    np.testing.assert_allclose(ntwk.s, np.broadcast_to(thru, ntwk.s.shape), atol=atol)


def _warning_messages(func, *args, **kwargs):
    """
    Call func with all the warnings recorded, return its result and the list of messages.
    Used where several warnings are expected at once, pytest.warns re-emits those it did not match.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = func(*args, **kwargs)
    return result, [str(w.message) for w in caught]


def _renumber_from_second(ntwk, port_order):
    """Copy of a network in 'second' port order, numbered with another port_order scheme."""
    n = ntwk.nports
    if port_order == 'first':
        perm = list(range(0, n, 2)) + list(range(1, n, 2))
    elif port_order == 'third':
        perm = list(range(0, n // 2)) + list(range(n - 1, n // 2 - 1, -1))
    else:
        return ntwk.copy()
    return ntwk.renumbered(list(range(n)), perm)


class TestModalNZC2xThru:
    """
    Tests of the N-coupled-line modal 2x-thru de-embedding.

    Fixtures are synthetic uniform coupled lines with a known modal transform, so that the
    FIX-DUT-FIX de-embedding can be compared with the known DUT.
    """

    # --- transform helpers ---------------------------------------------------------------------

    def test_modal_transform_equals_se2gmm(self):
        """The modal transform with the differential/common transform is Network.se2gmm."""
        # not symmetric between the lines, to be generic
        fix = _coupled_case(n_lines=2, mismatch=1.15, launch='skew')['fix']
        tv = np.broadcast_to(_MIXED_MODE_TV, (fix.frequency.npoints, 2, 2))
        s_modal = _se2modal(fix.s, _modal_xi(tv, 50.0, [100.0, 25.0]))
        ref = fix.copy()
        ref.se2gmm(p=2)
        # se2gmm order: dd_left, dd_right, cc_left, cc_right. modal order: d_left, c_left, d_right, c_right
        perm = [0, 2, 1, 3]
        np.testing.assert_allclose(s_modal[:, perm][:, :, perm], ref.s, atol=1e-12)

    def test_modal_transform_round_trip(self):
        """modal -> terminal returns the input, also for N = 3 with a non-orthogonal frequency dependent transform."""
        case = _coupled_case(n_lines=3, tv_slope=0.05)
        z_modal = _modal_references(case['tv'], 50.0)
        xi = _modal_xi(case['tv'], 50.0, z_modal)
        s_modal = _se2modal(case['fdf'].s, xi)
        np.testing.assert_allclose(_modal2se(s_modal, xi), case['fdf'].s, atol=1e-12)

    def test_abcd_2nport(self):
        """N = 1 equals s2a, and ABCD matrices of cascaded N = 3 line networks multiply."""
        freq = _MTL_FREQ
        tl = rf.media.DefinedGammaZ0(freq, z0_port=50, z0=60, gamma=1j * 2 * pi * freq.f / c * 2).line(0.03, 'm')
        np.testing.assert_allclose(_abcd_2nport(tl.s, 50.0), rf.network.s2a(tl.s, 50.0), atol=1e-12)
        case = _coupled_case(n_lines=3, mismatch=1.15)
        cascade = _abcd_2nport((case['fix'] ** case['dut']).s, 50.0)
        product = _abcd_2nport(case['fix'].s, 50.0) @ _abcd_2nport(case['dut'].s, 50.0)
        np.testing.assert_allclose(cascade, product, rtol=1e-9, atol=1e-9)

    def test_modal_references(self):
        """The mixed-mode transform gives 2 * z0 for the differential and z0 / 2 for the common mode."""
        tv = np.broadcast_to(_MIXED_MODE_TV, (5, 2, 2))
        np.testing.assert_allclose(_modal_references(tv, 50.0), [100.0, 25.0])

    # --- synthetic builders ---------------------------------------------------------------------

    @pytest.mark.parametrize('launch', [None, 'skew', 'cap'])
    def test_builders_are_reciprocal_and_passive(self, launch):
        case = _coupled_case(n_lines=2, mismatch=1.15, launch=launch)
        for ntwk in (case['fix'], case['dut'], case['s2x'], case['fdf']):
            np.testing.assert_allclose(ntwk.s, np.swapaxes(ntwk.s, 1, 2), atol=1e-12)
            assert np.linalg.svd(ntwk.s, compute_uv=False).max() <= 1 + 1e-12

    def test_builder_matched_line_has_no_reflection(self):
        """Matched means matched in the modal basis, where the modal impedances are the references."""
        case = _coupled_case(n_lines=3)
        n = 3
        z_modal = _modal_references(case['tv'], 50.0)
        s_modal = _se2modal(case['fix'].s, _modal_xi(case['tv'], 50.0, z_modal))
        assert np.abs(s_modal[:, :n, :n]).max() < 1e-12
        assert np.abs(s_modal[:, n:, n:]).max() < 1e-12
        through = s_modal[:, np.arange(n), n + np.arange(n)]
        assert np.abs(through).min() > 0.5
        # the transmission of different modes vanishes, the terminal S-parameters are coupled
        off_diagonal = s_modal[:, :n, n:] - np.einsum('fk,kl->fkl', through, np.eye(n))
        assert np.abs(off_diagonal).max() < 1e-12
        assert np.abs(case['fix'].s[:, 0, 4]).max() > 1e-3

    # --- eigenvector estimation -------------------------------------------------------------------

    def test_eig_transform_tracks_the_modes(self):
        """Per-frequency eigenvectors are continuous and equal to the true modes, slowest mode first."""
        case = _coupled_case(n_lines=3, tv_slope=0.05)
        tv, info = _eig_modal_transform(case['s2x'], 50.0, 'freq')
        true = case['tv'] / np.linalg.norm(case['tv'], axis=1, keepdims=True)
        overlap = np.abs(np.einsum('fik,fil->fkl', true.conj(), tv))
        np.testing.assert_allclose(overlap.max(axis=1), 1.0, atol=1e-6)
        # every estimated mode is one true mode, and always the same one
        match = overlap.argmax(axis=1)
        assert np.all(np.sort(match, axis=1) == np.arange(3))
        assert np.all(match == match[0])
        # effective permittivities are [3.2, 3.6, 4.0], the first column is the slowest mode
        assert match[0, 0] == 2
        assert np.all(np.abs(np.einsum('fik,fik->fk', tv, tv).imag) < 1e-12)
        # one transform for all the frequencies
        tv_const, _ = _eig_modal_transform(case['s2x'], 50.0, 'const')
        assert np.all(tv_const == tv_const[0])
        assert info['eig_gap'].min() > 0

    def test_eig_transform_rejects_bad_options(self):
        s2x = _coupled_case(n_lines=2)['s2x']
        with pytest.raises(ValueError, match='mode must be'):
            _eig_modal_transform(s2x, 50.0, 'fixed')
        with pytest.raises(ValueError, match='eig_scaling'):
            _eig_modal_transform(s2x, 50.0, 'freq', 'volts')

    # --- reduction to the mixed-mode class --------------------------------------------------------

    @pytest.mark.parametrize('modal_transform', ['mixed', 'eig_const', 'eig'])
    def test_reduces_to_mixed_mode_class(self, modal_transform):
        """N = 2 without conversion gives the result of IEEEP370_MM_NZC_2xThru."""
        case = _coupled_case(n_lines=2, mismatch=1.15)
        ref = IEEEP370_MM_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50, modal_transform=modal_transform)
        np.testing.assert_allclose(dm.se_side1.s, ref.se_side1.s, atol=1e-9)
        np.testing.assert_allclose(dm.se_side2.s, ref.se_side2.s, atol=1e-9)
        np.testing.assert_allclose(dm.deembed(case['fdf']).s, ref.deembed(case['fdf'].copy()).s, atol=1e-9)

    @pytest.mark.parametrize('modal_transform', ['mixed', 'eig_const', 'eig'])
    @pytest.mark.parametrize('connectors', ['', '_with_connectors'])
    def test_reduces_to_mixed_mode_class_on_example_data(self, modal_transform, connectors):
        """Same on the differential pair of the IEEE 370 example notebook (from the documentation folder)."""
        path_2xthru = os.path.join(_REAL_PAIR_DIR, f'diff_2xthru{connectors}.s4p')
        path_fdf = os.path.join(_REAL_PAIR_DIR, f'diff_fdf{connectors}.s4p')
        if not (os.path.exists(path_2xthru) and os.path.exists(path_fdf)):
            pytest.skip('example data not found')
        s2xthru = rf.Network(path_2xthru)
        fdf = rf.Network(path_fdf)
        ref = IEEEP370_MM_NZC_2xThru(dummy_2xthru=s2xthru, z0=50)
        dm = Modal_NZC_2xThru(dummy_2xthru=s2xthru, z0=50, modal_transform=modal_transform)
        np.testing.assert_allclose(dm.se_side1.s, ref.se_side1.s, atol=1e-9)
        np.testing.assert_allclose(dm.se_side2.s, ref.se_side2.s, atol=1e-9)
        np.testing.assert_allclose(dm.deembed(fdf).s, ref.deembed(fdf.copy()).s, atol=1e-9)

    @pytest.mark.parametrize('modal_transform', ['eig_const', 'eig'])
    def test_single_line_is_the_single_ended_method(self, modal_transform):
        """N = 1 (a 2-port 2x-thru) is IEEEP370_SE_NZC_2xThru, on the measured-like data of this folder."""
        test_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'deembed')
        s2xthru = rf.Network(os.path.join(test_dir, 's2xthru.s2p'))
        fdf = rf.Network(os.path.join(test_dir, 'fdf.s2p'))
        ref = IEEEP370_SE_NZC_2xThru(dummy_2xthru=s2xthru, z0=50)
        dm = Modal_NZC_2xThru(dummy_2xthru=s2xthru, z0=50, modal_transform=modal_transform)
        np.testing.assert_allclose(dm.se_side1.s, ref.s_side1.s, atol=1e-9)
        np.testing.assert_allclose(dm.se_side2.s, ref.s_side2.s, atol=1e-9)
        np.testing.assert_allclose(dm.deembed(fdf).s, ref.deembed(fdf).s, atol=1e-9)
        np.testing.assert_allclose(dm.z_modal, [50.0])

    def test_use_z_instead_ifft(self):
        """The option is passed to every mode."""
        case = _coupled_case(n_lines=2, mismatch=1.15)
        ref = IEEEP370_MM_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50, use_z_instead_ifft=True)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50, modal_transform='mixed', use_z_instead_ifft=True)
        np.testing.assert_allclose(dm.se_side1.s, ref.se_side1.s, atol=1e-9)
        np.testing.assert_allclose(dm.se_side2.s, ref.se_side2.s, atol=1e-9)

    def test_forced_z0_line_has_the_meaning_of_dd_and_cc(self):
        """With 'mixed', the entries of forced_z0_line are the differential and common split impedances."""
        case = _coupled_case(n_lines=2, mismatch=1.15)
        ref = IEEEP370_MM_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50, forced_z0_line_dd=105.0,
                                     forced_z0_line_cc=24.0)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50, modal_transform='mixed',
                              forced_z0_line=[105.0, 24.0])
        np.testing.assert_allclose(dm.se_side1.s, ref.se_side1.s, atol=1e-9)
        np.testing.assert_allclose(dm.se_side2.s, ref.se_side2.s, atol=1e-9)
        np.testing.assert_allclose(dm.z_x, [105.0, 24.0])

    # --- N coupled lines with known modal transform -----------------------------------------------

    @pytest.mark.parametrize('tv_slope', [0.0, 0.05])
    def test_known_transform_matched_lines(self, tv_slope):
        """N = 3 matched lines, constant or frequency dependent known transform: exact."""
        case = _coupled_case(n_lines=3, tv_slope=tv_slope)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=case['tv'])
        np.testing.assert_allclose(dm.deembed(case['fdf']).s, case['dut'].s, atol=1e-10)
        np.testing.assert_allclose(dm.se_side1.s, case['fix'].s, atol=1e-10)
        np.testing.assert_allclose(dm.se_side2.flipped().s, case['fix'].s, atol=1e-10)

    def test_known_constant_transform_as_matrix(self):
        """A (N, N) array is a constant transform."""
        case = _coupled_case(n_lines=3)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=case['tv'][0])
        np.testing.assert_allclose(dm.deembed(case['fdf']).s, case['dut'].s, atol=1e-10)

    def test_known_transform_mismatched_lines(self):
        """15 % launch mismatch, limit of the single-ended NZC routine (about 2e-3 here)."""
        case = _coupled_case(n_lines=3, mismatch=1.15)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=case['tv'])
        assert np.abs(dm.deembed(case['fdf']).s - case['dut'].s).max() < 5e-3

    def test_known_transform_works_in_a_homogeneous_medium(self):
        """Modes with equal propagation constants cannot be found from the 2x-thru, but a given transform works."""
        freq = _MTL_FREQ
        tv = _MTL_TV0[3]
        zc = _modal_references(np.broadcast_to(tv, (freq.npoints, 3, 3)), 50.0)
        fix = _mtl(freq, tv, [3.5] * 3, [0.6] * 3, zc, 0.025)
        dut = _mtl(freq, tv + 0.3 * np.eye(3), [3.5] * 3, [0.6] * 3, zc * 0.9, 0.01)
        dm = Modal_NZC_2xThru(dummy_2xthru=fix ** fix.flipped(), modal_transform=tv)
        np.testing.assert_allclose(dm.deembed(fix ** dut ** fix.flipped()).s, dut.s, atol=1e-9)

    def test_eigenvalue_gap_warning_in_a_homogeneous_medium(self):
        """The eigenvector options warn when the modes cannot be told apart."""
        freq = _MTL_FREQ
        tv = _MTL_TV0[3]
        zc = _modal_references(np.broadcast_to(tv, (freq.npoints, 3, 3)), 50.0)
        fix = _mtl(freq, tv, [3.5] * 3, [0.6] * 3, zc, 0.025)
        _, messages = _warning_messages(Modal_NZC_2xThru, fix ** fix.flipped(), modal_transform='eig_const')
        assert any(m.startswith('Modal 2x-thru:') and 'nearly equal eigenvalues' in m for m in messages)

    @pytest.mark.parametrize('modal_transform', ['eig_const', 'eig'])
    def test_eigenvector_transform_constant_modal_transform(self, modal_transform):
        """N = 3 lines, the transform comes from the 2x-thru; matched, then 15 % mismatch."""
        case = _coupled_case(n_lines=3)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=modal_transform)
        np.testing.assert_allclose(dm.deembed(case['fdf']).s, case['dut'].s, atol=1e-10)
        case = _coupled_case(n_lines=3, mismatch=1.15)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=modal_transform)
        assert np.abs(dm.deembed(case['fdf']).s - case['dut'].s).max() < 5e-3

    def test_eig_image_scaling_with_frequency_dependent_transform(self):
        """Unit-norm eigenvectors are wrong if the modes change with frequency, image impedance scaling is exact."""
        case = _coupled_case(n_lines=3, tv_slope=0.05)
        unit = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig')
        err_unit = np.abs(unit.deembed(case['fdf']).s - case['dut'].s).max()
        assert 1e-3 < err_unit < 1e-1
        image = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig', eig_scaling='image')
        np.testing.assert_allclose(image.deembed(case['fdf']).s, case['dut'].s, atol=1e-10)

    @pytest.mark.parametrize('modal_transform', ['eig_const', 'eig', 'known'])
    def test_2xthru_self_consistency(self, modal_transform):
        """De-embedding the 2x-thru gives a thru. For 'eig' this is always true, it is no quality check."""
        case = _coupled_case(n_lines=3)
        if modal_transform == 'known':
            modal_transform = case['tv']
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=modal_transform)
        _assert_is_thru(dm.deembed(case['s2x']))

    def test_2xthru_self_consistency_mixed(self):
        case = _coupled_case(n_lines=2, mismatch=1.15)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='mixed')
        _assert_is_thru(dm.deembed(case['s2x']))

    def test_noisy_2xthru_constant_transform(self):
        """Measurement-like noise on the 2x-thru: the constant eigenvector transform is as good as the known one."""
        case = _coupled_case(n_lines=3)
        noisy = _add_noise(case['s2x'], np.random.default_rng(1))
        dm = Modal_NZC_2xThru(dummy_2xthru=noisy, modal_transform='eig_const')
        err_const = np.abs(dm.deembed(case['fdf']).s - case['dut'].s).max()
        known = Modal_NZC_2xThru(dummy_2xthru=noisy, modal_transform=case['tv'])
        err_known = np.abs(known.deembed(case['fdf']).s - case['dut'].s).max()
        assert err_const < 1e-2
        assert err_const < 1.5 * err_known

    # --- mode conversion and warnings -------------------------------------------------------------

    def test_noisy_2xthru_per_frequency_transform_warns(self):
        """Per-frequency eigenvectors are unreliable with noise: they are not smooth, and it is said."""
        case = _coupled_case(n_lines=3)
        noisy = _add_noise(case['s2x'], np.random.default_rng(1))
        dm, messages = _warning_messages(Modal_NZC_2xThru, noisy, modal_transform='eig')
        assert any(m.startswith('Modal 2x-thru:') and 'eigenvectors of the 2x-thru' in m for m in messages)
        assert dm.transform_ripple > Modal_NZC_2xThru.transform_ripple_limit
        err = np.abs(dm.deembed(case['fdf']).s - case['dut'].s).max()
        assert err > 3e-2

    @pytest.mark.parametrize('launch', ['skew', 'cap'])
    def test_mode_converting_launch_warns_constant_transform(self, launch):
        """A launch that converts modes is seen from the conversion left in the modal basis."""
        case = _coupled_case(n_lines=2, mismatch=1.15, launch=launch)
        with pytest.warns(RuntimeWarning, match='Modal 2x-thru: the mode conversion'):
            dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig_const')
        assert dm.mode_conversion_db.max() > dm.conversion_limit_db
        assert dm.transform_ripple > Modal_NZC_2xThru.transform_ripple_limit

    @pytest.mark.parametrize('launch', ['skew', 'cap'])
    def test_mode_converting_launch_warns_per_frequency_transform(self, launch):
        """Per-frequency eigenvectors decouple the 2x-thru by construction, the ripple of the vectors tells."""
        case = _coupled_case(n_lines=2, mismatch=1.15, launch=launch)
        with pytest.warns(RuntimeWarning, match='Modal 2x-thru: the eigenvectors of the 2x-thru'):
            dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig')
        assert dm.mode_conversion_db.max() < -200
        # the 2x-thru thru-test is passed whatever the quality, see the docstring
        _assert_is_thru(dm.deembed(case['s2x']))

    @pytest.mark.parametrize('launch', ['skew', 'cap'])
    def test_mode_converting_launch_warns_mixed(self, launch):
        case = _coupled_case(n_lines=2, mismatch=1.15, launch=launch)
        with pytest.warns(RuntimeWarning, match='Modal 2x-thru: the mode conversion'):
            dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='mixed')
        assert dm.mode_conversion_db.max() > -15

    def test_conversion_limit_is_adjustable(self):
        case = _coupled_case(n_lines=2, mismatch=1.15, launch='skew')
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig_const', conversion_limit_db=-10.0)
        assert dm.mode_conversion_db.max() < -10.0

    def test_no_warning_for_uniform_lines(self):
        """The warnings are errors in this test suite: uniform lines must not produce any."""
        for kwargs in ({'n_lines': 3}, {'n_lines': 3, 'mismatch': 1.15}, {'n_lines': 2, 'mismatch': 1.15}):
            case = _coupled_case(**kwargs)
            for modal_transform in ('eig_const', 'eig'):
                dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=modal_transform)
                dm.deembed(case['fdf'])
                assert dm.mirror_asymmetry < 1e-9
                assert dm.eig_gap.min() > 0

    def test_not_mirror_symmetric_2xthru_warns(self):
        """FIX-2 is not the mirror image of FIX-1: the A and D blocks of the ABCD matrix tell."""
        case = _coupled_case(n_lines=3)
        tv = case['tv']
        other = _mtl(_MTL_FREQ, tv, _MTL_EPS[3], _MTL_ALPHA[3], _modal_references(tv, 50.0) * 1.05, 0.025)
        dm, messages = _warning_messages(Modal_NZC_2xThru, case['fix'] ** other.flipped(), modal_transform='eig_const')
        assert any(m.startswith('Modal 2x-thru:') and 'mirror image' in m for m in messages)
        assert dm.mirror_asymmetry > Modal_NZC_2xThru.mirror_asymmetry_limit

    # --- frequency axis ---------------------------------------------------------------------------

    def test_dc_point(self):
        """A DC point is excluded from the estimation, the warning of the single-ended routine is given once."""
        freq = rf.Frequency(0, 10000, 1001, 'MHz')
        case = _coupled_case(n_lines=3, mismatch=1.15, freq=freq)
        with pytest.warns(RuntimeWarning, match='DC point detected'):
            dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig_const')
        _, messages = _warning_messages(Modal_NZC_2xThru, case['s2x'], modal_transform='eig_const')
        assert sum('DC point detected' in m for m in messages) == 1
        assert np.abs(dm.deembed(case['fdf']).s - case['dut'].s).max() < 5e-3

    def test_non_uniform_frequency_axis(self):
        """A non-harmonic sweep is interpolated by the single-ended routine, which warns (once)."""
        freq = rf.Frequency(10, 10000, 990, 'MHz')
        case = _coupled_case(n_lines=3, freq=freq)
        _, messages = _warning_messages(Modal_NZC_2xThru, case['s2x'], modal_transform='eig_const')
        assert sum('Non-uniform frequency vector detected' in m for m in messages) == 1
        with pytest.warns(RuntimeWarning, match='Non-uniform frequency vector detected'):
            dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='eig_const')
        assert np.abs(dm.deembed(case['fdf']).s - case['dut'].s).max() < 5e-2

    def test_frequency_mismatch(self):
        """FIX-DUT-FIX on another frequency range: overlap, with a warning."""
        case = _coupled_case(n_lines=2, mismatch=1.15)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform='mixed')
        fdf = case['fdf']['10-5000MHz']
        with pytest.warns(RuntimeWarning, match='Network frequencies dont match dummy frequencies'):
            dut = dm.deembed(fdf)
        assert dut.frequency.npoints == fdf.frequency.npoints
        assert np.abs(dut.s - case['dut']['10-5000MHz'].s).max() < 5e-3

    # --- ports, arguments -------------------------------------------------------------------------

    @pytest.mark.parametrize('port_order', ['first', 'second', 'third'])
    @pytest.mark.parametrize('n_lines', [2, 4])
    def test_port_orders(self, port_order, n_lines):
        """The numbering schemes of the mixed-mode class, for any number of lines."""
        case = _coupled_case(n_lines=n_lines)
        s2xthru = _renumber_from_second(case['s2x'], port_order)
        fdf = _renumber_from_second(case['fdf'], port_order)
        fdf_before = fdf.copy()
        dm = Modal_NZC_2xThru(dummy_2xthru=s2xthru, port_order=port_order, modal_transform='eig_const')
        dut = dm.deembed(fdf)
        np.testing.assert_allclose(dut.s, _renumber_from_second(case['dut'], port_order).s, atol=1e-9)
        # the networks of the caller are left alone
        np.testing.assert_array_equal(fdf.s, fdf_before.s)
        np.testing.assert_array_equal(s2xthru.s, _renumber_from_second(case['s2x'], port_order).s)
        # error boxes are in 'second' order
        np.testing.assert_allclose(dm.se_side1.s, case['fix'].s, atol=1e-9)

    def test_input_networks_are_not_modified(self):
        case = _coupled_case(n_lines=2)
        s2xthru = case['s2x'].copy()
        fdf = case['fdf'].copy()
        dm = Modal_NZC_2xThru(dummy_2xthru=s2xthru, modal_transform='mixed', port_order='first')
        dm.deembed(fdf)
        np.testing.assert_array_equal(s2xthru.s, case['s2x'].s)
        np.testing.assert_array_equal(fdf.s, case['fdf'].s)

    def test_other_reference_impedance(self):
        """A 2x-thru with another reference impedance is renormalized, a copy of it."""
        case = _coupled_case(n_lines=2, mismatch=1.15)
        s2xthru = case['s2x'].copy()
        s2xthru.renormalize(75.0)
        dm = Modal_NZC_2xThru(dummy_2xthru=s2xthru, z0=50, modal_transform='mixed')
        ref = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], z0=50, modal_transform='mixed')
        np.testing.assert_allclose(dm.se_side1.s, ref.se_side1.s, atol=1e-9)
        np.testing.assert_allclose(s2xthru.z0, 75.0)

    def test_modal_z0_and_debug_outputs(self):
        case = _coupled_case(n_lines=3)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=case['tv'])
        np.testing.assert_allclose(dm.z_modal, _modal_references(case['tv'], 50.0))
        assert dm.tv.shape == (1000, 3, 3)
        assert len(dm.x_end) == len(dm.z_x) == 3
        assert dm.modal_2xthru.nports == 6
        assert dm.mode_conversion_db.shape == (1000,)
        assert dm.transform_ripple is None
        # the split depends weakly on the modal reference, in the same way as for the mixed-mode class:
        # about 1e-3 for a reference 10 % off the modal impedance of the lines
        z_modal = 1.1 * dm.z_modal
        dm2 = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], modal_transform=case['tv'], modal_z0=z_modal)
        np.testing.assert_allclose(dm2.z_modal, z_modal)
        np.testing.assert_allclose(dm2.modal_2xthru.z0[0], np.concatenate([z_modal, z_modal]))
        err = np.abs(dm2.deembed(case['fdf']).s - case['dut'].s).max()
        assert 1e-6 < err < 5e-3

    def test_str(self):
        case = _coupled_case(n_lines=2)
        dm = Modal_NZC_2xThru(dummy_2xthru=case['s2x'], name='modal', modal_transform='mixed')
        assert 'Modal_NZC_2xThru' in str(dm)
        assert 'modal' in str(dm)

    def test_argument_errors(self):
        case = _coupled_case(n_lines=2)
        s2x = case['s2x']
        three_port = rf.Network(frequency=_MTL_FREQ, s=np.zeros((1000, 3, 3)), z0=50)
        with pytest.raises(ValueError, match='even number of ports'):
            Modal_NZC_2xThru(three_port)
        with pytest.raises(ValueError, match='modal_transform must be one of'):
            Modal_NZC_2xThru(s2x, modal_transform='magic')
        with pytest.raises(ValueError, match='differential pairs only'):
            Modal_NZC_2xThru(_coupled_case(n_lines=3)['s2x'], modal_transform='mixed')
        with pytest.raises(ValueError, match='modal_transform array must have shape'):
            Modal_NZC_2xThru(s2x, modal_transform=np.eye(3))
        with pytest.raises(ValueError, match='frequency points'):
            Modal_NZC_2xThru(s2x, modal_transform=np.ones((7, 2, 2)))
        with pytest.raises(ValueError, match='not invertible'):
            Modal_NZC_2xThru(s2x, modal_transform=np.ones((2, 2)))
        with pytest.raises(ValueError, match='port_order'):
            Modal_NZC_2xThru(s2x, port_order='fourth')
        with pytest.raises(ValueError, match='z0 must be'):
            Modal_NZC_2xThru(s2x, z0=50 + 5j)
        with pytest.raises(ValueError, match='eig_scaling'):
            Modal_NZC_2xThru(s2x, eig_scaling='volts')
        with pytest.raises(ValueError, match="applies to modal_transform='eig'"):
            Modal_NZC_2xThru(s2x, modal_transform='eig_const', eig_scaling='image')
        with pytest.raises(ValueError, match='modal_z0'):
            Modal_NZC_2xThru(s2x, modal_transform='mixed', modal_z0=[50.0])
        with pytest.raises(ValueError, match='forced_z0_line'):
            Modal_NZC_2xThru(s2x, modal_transform='mixed', forced_z0_line=[50.0, 50.0, 50.0])
        dm = Modal_NZC_2xThru(s2x, modal_transform='mixed')
        with pytest.raises(ValueError, match='ports'):
            dm.deembed(three_port)
