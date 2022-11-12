import pytest
import skrf as rf
import unittest
import os
import numpy as np
from numpy.testing import assert_equal, run_module_suite, assert_almost_equal


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
        self.dm = rf.OpenShort(self.open, self.short)
        self.dm_os = rf.OpenShort(self.open_1f, self.short_1f) 
        self.dm_o = rf.Open(self.open7_1f)
        self.dm_so = rf.ShortOpen(self.short2_1f, self.open2_1f)
        self.dm_s = rf.Short(self.short8_1f)
        self.dm_pi = rf.SplitPi(self.thru3_1f)
        self.dm_tee = rf.SplitTee(self.thru4_1f)
        self.dm_ac = rf.AdmittanceCancel(self.thru5_1f)
        self.dm_ic = rf.ImpedanceCancel(self.thru6_1f)
        
        # relative tolerance for comparisons
        self.rtol = 1e-6

    def test_freqmismatch(self):
        """
        Check that error is caught when networks are of different frequencies
        """
        with self.assertRaises(ValueError):
            rf.OpenShort(self.open, self.short_1f)
        
        with self.assertRaises(ValueError):
            self.dm_os.deembed(self.raw)

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
        dm_nzc = rf.IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru, 
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
            dm_nzc = rf.IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru_dc, 
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
        While interpolation should be avoided, it keep usefull in some cases,
        e.g. when the measurement frequency axis was not well formated but
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
            dm_nzc_nu = rf.IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru_nu, 
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
        dm_zc  = rf.IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru, 
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
            dm_zc  = rf.IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru_dc, 
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
        While interpolation should be avoided, it keep usefull in some cases,
        e.g. when the measurement frequency axis was not well formated but
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
            dm_zc_nu  = rf.IEEEP370_SE_ZC_2xThru(dummy_2xthru = s2xthru_nu, 
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
        dm_nzc_pn = rf.IEEEP370_SE_NZC_2xThru(dummy_2xthru = s2xthru_pn, 
                                        name = '2xthru')
        residuals = dm_nzc_pn.deembed(s2xthru_pn)
        # insertion loss magnitude deviate from 1.0 from less than 0.1 dB
        il_mag = 20.*np.log10(np.abs(residuals.s[:, 1, 0] + 1e-12))
        self.assertTrue(np.max(np.abs(il_mag)) <= 0.1, 'residual IL magnitude')
        # insertion loss phase deviate from 0 degree from less than 1 degree
        il_phase = np.angle(residuals.s[:, 1, 0]) * 180/np.pi
        self.assertTrue(np.max(np.abs(il_phase)) <= 1.0, 'residual IL Phase')

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()