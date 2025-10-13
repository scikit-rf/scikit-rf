import copy
import unittest
import warnings

import numpy as np
import pytest

import skrf as rf
from skrf import two_port_reflect
from skrf.calibration import (
    PHN,
    SOLT,
    MultiportCal,
    MultiportSOLT,
    NISTMultilineTRL,
    TUGMultilineTRL,
    TwelveTerm,
    TwoPortOnePath,
    UnknownThru,
    compute_switch_terms,
    determine_line,
    determine_reflect,
    terminate,
    terminate_nport,
)
from skrf.media import Coaxial, DistributedCircuit
from skrf.networkSet import NetworkSet
from skrf.util import suppress_warning_decorator

# number of frequency points to test calibration at .
# i choose 1 for speed, but given that many tests employ *random*
# networks values >100 are better for initial verification
global NPTS
NPTS = 1

# WR10/WG27/R900 75 to 110 GHz, 0.1x0.05 inch (2.54x1.27 mm)
# z0 from 610 to 446 ohm
WG_lossless = rf.RectangularWaveguide(rf.F(75, 100, NPTS, unit='GHz'),
                                      a=100*rf.mil, z0_override=50)
WG = rf.RectangularWaveguide(rf.F(75, 100, NPTS, unit='GHz'), a=100*rf.mil,
                                      rho='gold', z0_override=50)

def _compare_dicts_allclose(first: dict, second: dict) -> None:
    assert first.keys() == second.keys()
    for k in first.keys():
        np.testing.assert_allclose(first[k], second[k], err_msg=f"Values from key '{k}' not equal!")


class DetermineTest(unittest.TestCase):
    def setUp(self):
        self.wg = WG
        wg = self.wg
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')


        #thru
        self.T = wg.thru()
        self.T_m = self.embed(self.T)

        #line
        self.L = wg.line(80,'deg')
        self.L_approx = self.wg.line(90,'deg')
        self.L_m = self.embed(self.L)

        # reflect
        self.r = [ wg.delay_load(p,k,'deg') \
                    for k in [-10,10,88,92] \
                    for p in [-.9,-1]]


        self.R = [rf.two_port_reflect(k,k) for k in self.r]

        short= wg.short()
        open = wg.open()
        self.r_estimate = [short, short, short, short, open, open, open ,open]
        self.R_m = [self.embed(k) for k in self.R]

    def embed(self,x):
        return self.X**x**self.Y

    def test_determine_line(self):
        L_found = determine_line(self.T_m, self.L_m,
                                 line_approx=self.L_approx)
        self.assertEqual(L_found,self.L)

    def test_determine_reflect(self):
        r_found = [determine_reflect(self.T_m,k,self.L_m, l, self.L_approx) \
                   for k,l in zip(self.R_m, self.r_estimate)]

        [ self.assertEqual(k,l) for k,l in zip(self.r, r_found)]

    def test_determine_reflect_matched_thru_and_line_ideal_reflect(self):
        freq = rf.F(.25, .7, 21, unit = 'GHz')
        medium = Coaxial.from_attenuation_VF(freq, att = 3.0, VF = .69)

        thru = medium.line(0, 'm')
        line = medium.line(0.12, 'm')

        short = rf.two_port_reflect(medium.short(), medium.short())
        r = determine_reflect(thru, short, line)
        np.testing.assert_array_almost_equal( r.s, -np.ones_like(r.s))

        reflect = rf.two_port_reflect(medium.open(), medium.open())
        r = determine_reflect(thru, reflect, line, reflect_approx = medium.open())
        np.testing.assert_array_almost_equal(r.s, np.ones_like(r.s))

    def test_determine_reflect_matched_thru_and_line(self):
        freq = rf.F(.25, .7, 40, unit = 'GHz')
        medium = Coaxial.from_attenuation_VF(freq, att = 3.0, VF = .69)

        thru = medium.line(0, 'm')
        line = medium.line(0.12, 'm')
        short = medium.short()

        rng = np.random.default_rng(12)
        short.s[:, 0, 0] += rng.uniform(-.02, 0.02, freq.f.size) + rng.uniform(-.02, 0.02, freq.f.size)*1j

        r = determine_reflect(thru, rf.two_port_reflect(short, short), line)
        np.testing.assert_array_almost_equal( r.s, short.s)


    def test_determine_reflect_regression(self):
        # this test case fails with only regularization on the t-parameters of thru ** line.inv, see gh-870
        freq = rf.F(434615384.6153846e-9, .7, 1, unit = 'GHz')
        medium = Coaxial.from_attenuation_VF(freq, att = 3.0, VF = .69)
        thru= medium.line(0, 'm')
        line = medium.line(0.12, 'm')
        short = rf.Network(frequency=freq, s=[[[(-1.+0.017870117714376983j)] ] ])

        r = determine_reflect(thru, rf.two_port_reflect(short, short), line)
        np.testing.assert_array_almost_equal( r.s, short.s)

class ComputeSwitchTermsTest(unittest.TestCase):
    '''
    test the indirect method of computing the switch terms
    with at least three reciprocal devices
    '''
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')

        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        # Reciprocal devices: asymmetric and both transmissive and reflective devices
        Rs = [25, 50, 100, ]
        stands = [wg.resistor(R)**wg.shunt_resistor(R) for R in Rs]
        stands_meas = [terminate(self.X**k**self.Y, self.gamma_f, self.gamma_r) for k in stands]

        self.gamma_f_indirect, self.gamma_r_indirect = compute_switch_terms(stands_meas)

    def test_gamma_f(self):
        self.assertTrue(all(np.abs(self.gamma_f_indirect.s - self.gamma_f.s) < 1e-9))

    def test_gamma_r(self):
        self.assertTrue(all(np.abs(self.gamma_r_indirect.s - self.gamma_r.s) < 1e-9))


class TCheckTest(unittest.TestCase):
    """
    Test the T-Check metric
    """
    def setUp(self):
        pass

    def test_t_check(self):
        medium = rf.media.DefinedGammaZ0()
        tee = medium.tee()
        ct = rf.calibration.t_check(tee.subnetwork([1, 2]))
        self.assertTrue(np.allclose(ct, 1))


class CalibrationTest:
    """
    This is the generic Calibration test case which all Calibration
    Subclasses should be able to pass. They must implement
    """

    def test_accuracy_of_dut_correction(self):
        a = self.wg.random(n_ports=self.n_ports, name = 'actual')
        m = self.measure(a)
        c = self.cal.apply_cal(m)
        c.name = 'corrected'
        self.assertEqual(c,a)

    def test_error_ntwk(self):
        a = self.cal.error_ntwk

    def test_coefs_ntwks(self):
        a = self.cal.coefs_ntwks

    @suppress_warning_decorator("only gave a single measurement orientation")
    def test_caled_ntwks(self):
        a = self.cal.caled_ntwks

    @suppress_warning_decorator("only gave a single measurement orientation")
    def test_residual_ntwks(self):
        a = self.cal.residual_ntwks

    def test_embed_then_apply_cal(self):
        a = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(self.cal.apply_cal(self.cal.embed(a)),a)

    def test_embed_equal_measure(self):
        a = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(self.cal.embed(a),self.measure(a))

    @suppress_warning_decorator("n_thrus is None")
    def test_from_coefs(self):
        cal_from_coefs = self.cal.from_coefs(self.cal.frequency, self.cal.coefs)
        ntwk = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(cal_from_coefs.apply_cal(self.cal.embed(ntwk)),ntwk)

    @suppress_warning_decorator("n_thrus is None")
    def test_from_coefs_ntwks(self):
        cal_from_coefs = self.cal.from_coefs_ntwks(self.cal.coefs_ntwks)
        ntwk = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(cal_from_coefs.apply_cal(self.cal.embed(ntwk)),ntwk)


class CalibrationInputsTest(unittest.TestCase):
    """Test Calibration class input validation"""
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        self.ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]

        self.measured = [k.copy() for k in self.ideals]

    def init(self):
        self.cal = rf.Calibration(
            ideals = self.ideals,
            measured = self.measured
            )

    def test_meas_z0(self):
        self.measured[0].z0 *= 0.5
        with pytest.raises(ValueError):
            self.init()

    def test_ideals_z0(self):
        self.ideals[0].z0 *= 0.5
        with pytest.raises(ValueError):
            self.init()

    def test_ideals_frequency(self):
        freqs = rf.F(1, 10, NPTS+1, unit="GHz")
        self.ideals[-1] = rf.Network(s=np.zeros([NPTS + 1, 2, 2]), frequency=freqs)
        with pytest.raises(IndexError):
            self.init()

    def test_measured_frequency(self):
        freqs = rf.F(1, 10, NPTS+1, unit="GHz")
        self.measured[-1] = rf.Network(s=np.zeros([NPTS + 1, 2, 2]), frequency=freqs)
        with pytest.raises(ValueError):
            self.init()

    def test_ideals_frequency2(self):
        freqs = self.wg.frequency.copy()
        freqs._f += 1e9
        self.ideals[-1].frequency = freqs
        with pytest.raises(IndexError):
            self.init()

    def test_measured_frequency2(self):
        freqs = self.wg.frequency.copy()
        freqs._f += 1e9
        self.measured[-1].frequency = freqs
        with pytest.raises(ValueError):
            self.init()

class OnePortTest(unittest.TestCase, CalibrationTest):
    """
    One-port calibration test.


    """
    def setUp(self):
        self.n_ports = 1
        self.wg = WG
        wg = self.wg

        self.E = wg.random(n_ports =2, name = 'E')

        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.match( name='load'),
                ]
        measured = [self.measure(k) for k in ideals]

        self.cal = rf.OnePort(
            is_reciprocal = True,
            ideals = ideals,
            measured = measured,
            )

    def test_input_networks_1port(self):
        # test users do not enter 2-port networks by accident
        with self.assertRaises(RuntimeError):
            wg = self.wg
            ideals = [
                    two_port_reflect(wg.short( name='short'), wg.short( name='short')),
                    wg.delay_short( 45.,'deg',name='ew'),
                    wg.delay_short( 90.,'deg',name='qw'),
                    wg.match( name='load'),
                    ]
            measured = [self.measure(k) for k in ideals]
            cal = rf.OnePort(
               is_reciprocal = True,
               ideals = ideals,
               measured = measured,
               )
            cal.run()

    def measure(self, ntwk):
        out = self.E**ntwk
        out.name = ntwk.name
        return out

    def test_accuracy_of_directivity(self):
        self.assertEqual(
            self.E.s11,
            self.cal.coefs_ntwks['directivity'],
            )

    def test_accuracy_of_source_match(self):
        self.assertEqual(
            self.E.s22,
            self.cal.coefs_ntwks['source match'],
            )

    def test_accuracy_of_reflection_tracking(self):
        self.assertEqual(
            self.E.s21*self.E.s12,
            self.cal.coefs_ntwks['reflection tracking'],
            )


class SDDLTest(OnePortTest):
    def setUp(self):
        self.n_ports = 1
        #Exact only with a lossless waveguide
        self.wg = WG_lossless
        wg = self.wg

        self.E = wg.random(n_ports =2, name = 'E')
        #self.E.s[0,:,:] = np.array([[.1j,1],[1j,1j+2]])
        #print self.E.s[0]

        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 33.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        measured = [self.measure(k) for k in actuals]

        self.cal = rf.SDDL(
            is_reciprocal = True,
            ideals = ideals,
            measured = measured,
            )

    def test_init_with_nones(self):
        wg=self.wg
        wg.frequency = rf.F.from_f([100], unit='GHz')

        self.E = wg.random(n_ports =2, name = 'E')

        ideals = [
                wg.short( name='short'),
                None,
                None,
                wg.load(.2+.2j, name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 33.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        measured = [self.measure(k) for k in actuals]

        self.cal = rf.SDDL(
            is_reciprocal = True,
            ideals = ideals,
            measured = measured,
            )
        self.cal.run()

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs_ntwks(self):
        pass


class SDDLWeikle(OnePortTest):
    def setUp(self):
        self.n_ports = 1
        #Exact only with a lossless waveguide
        self.wg = WG_lossless
        wg = self.wg
        self.E = wg.random(n_ports =2, name = 'E')
        #self.E.s[0,:,:] = np.array([[.1j,1],[1j,1j+2]])
        #print self.E.s[0]

        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 80.,'deg',name='qw'),
                wg.load(.2+.2j, name='load'),
                ]
        measured = [self.measure(k) for k in actuals]

        self.cal = rf.SDDLWeikle(
            is_reciprocal = True,
            ideals = ideals,
            measured = measured,
            )

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs_ntwks(self):
        pass


class SDDMTest(OnePortTest):
    """
    This is a specific test of SDDL to verify it works when the load is
    a matched load. This test has been used to show that the SDDLWeikle
    variant fails, with a perfect matched load.
    """
    def setUp(self):

        self.n_ports = 1
        #Exact only with a lossless waveguide
        self.wg = WG_lossless
        wg = self.wg

        self.E = wg.random(n_ports =2, name = 'E')

        ideals = [
                wg.short( name='short'),
                wg.delay_short( 45.,'deg',name='ew'),
                wg.delay_short( 90.,'deg',name='qw'),
                wg.match( name='load'),
                ]
        actuals = [
                wg.short( name='short'),
                wg.delay_short( 10.,'deg',name='ew'),
                wg.delay_short( 80.,'deg',name='qw'),
                wg.match(name='load'),
                ]
        measured = [self.measure(k) for k in actuals]

        self.cal = rf.SDDL(
            is_reciprocal = True,
            ideals = ideals,
            measured = measured,
            )

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs_ntwks(self):
        pass


@pytest.mark.skip()
class PHNTest(OnePortTest):
    """
    """
    def setUp(self):

        self.n_ports = 1
        self.wg = WG
        wg = self.wg

        self.E = wg.random(n_ports =2, name = 'E')
        known1 = wg.random()
        known2 = wg.random()
        #known1 = wg.short()
        #known2 = wg.load(rand() + rand()*1j)

        ideals = [
                wg.delay_short( 45.,'deg',name='ideal ew'),
                wg.delay_short( 90.,'deg',name='ideal qw'),
                known1,
                known2,
                ]
        actuals = [
                wg.delay_short( 33.,'deg',name='true ew'),
                wg.delay_short( 110.,'deg',name='true qw'),
                known1,
                known2,
                ]
        measured = [self.measure(k) for k in actuals]
        self.actuals = actuals

        self.cal = PHN(
            is_reciprocal = True,
            ideals = ideals,
            measured = measured,
            )


    def test_determine_ideals(self):
        self.cal.run()

        self.assertEqual(self.actuals[0], self.cal.ideals[0])
        self.assertEqual(self.actuals[1], self.cal.ideals[1])

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable ')
    def test_from_coefs_ntwks(self):
        pass


class EightTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg =WG
        wg= self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        #Isolation terms
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.EightTerm(
            ideals = ideals,
            measured = measured,
            isolation = measured[2],
            switch_terms = (self.gamma_f, self.gamma_r)
            )

    def terminate(self, ntwk):
        """
        terminate a measured network with the switch terms
        """
        return terminate(ntwk,self.gamma_f, self.gamma_r)

    def measure(self,ntwk):
        out =  self.terminate(self.X**ntwk**self.Y)
        out.name = ntwk.name
        out.s[:,1,0] += self.If.s[:,0,0]
        out.s[:,0,1] += self.Ir.s[:,0,0]
        return out

    def test_renormalize(self):
        dut = self.wg.random(n_ports=2, name='DUT')
        dut_meas = self.measure(dut)
        z_new = 100-30j
        self.cal.renormalize(50, z_new)
        dut_cal = self.cal.apply_cal(dut_meas)
        dut.renormalize(z_new, s_def="traveling")
        dut.z0 = 50
        self.assertEqual(dut, dut_cal)

    def test_renormalize2(self):
        """
        Test that roundtrip renormalization gives back the original calibration
        """
        old_coefs = copy.deepcopy(self.cal.coefs)
        z_new = 10 + 20j
        self.cal.renormalize(50, z_new)
        self.cal.renormalize(z_new, 50)
        for k in self.cal.coefs.keys():
            np.testing.assert_array_almost_equal(old_coefs[k], self.cal.coefs[k], err_msg=k)

    def test_unterminating(self):
        a = self.wg.random(n_ports=self.n_ports)
        #unterminated measurement
        ut =  self.X**a**self.Y
        #terminated measurement
        m = self.measure(a)
        #Remove leakage
        m.s[:,1,0] -= self.If.s[:,0,0]
        m.s[:,0,1] -= self.Ir.s[:,0,0]
        self.assertEqual(self.cal.unterminate(m), ut)

    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.X.s11,
            self.cal.coefs_ntwks['forward directivity'])

    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.X.s22 ,
            self.cal.coefs_ntwks['forward source match'] )

    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.X.s21 * self.X.s12 ,
            self.cal.coefs_ntwks['forward reflection tracking'])

    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Y.s11 ,
            self.cal.coefs_ntwks['reverse source match']   )

    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Y.s22 ,
            self.cal.coefs_ntwks['reverse directivity']  )

    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Y.s21 * self.Y.s12 ,
            self.cal.coefs_ntwks['reverse reflection tracking'])

    def test_k_accuracy(self):
        self.assertEqual(
            self.X.s21/self.Y.s12 ,
            self.cal.coefs_ntwks['k']  )

    def test_forward_isolation_accuracy(self):
        self.assertEqual(
            self.If.s11 ,
            self.cal.coefs_ntwks['forward isolation']  )

    def test_reverse_isolation_accuracy(self):
        self.assertEqual(
            self.Ir.s11 ,
            self.cal.coefs_ntwks['reverse isolation']  )

    def test_verify_12term(self):
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)

    def test_coefs_8term(self):
        self.assertEqual(self.cal.coefs_8term, self.cal.coefs)

    def test_coefs_12term(self):
        _compare_dicts_allclose(self.cal.coefs_12term, rf.convert_8term_2_12term(self.cal.coefs))

    def test_input_not_modified(self):
        dut_before = self.X.copy()
        self.cal.apply_cal(self.X)
        self.assertEqual(dut_before, self.X)

    def test_convert_8term_2_12term_subtests(self):

        for st in [
            (self.cal.coefs['forward switch term'], self.cal.coefs['reverse switch term']),
            (np.zeros(self.cal.frequency.npoints, dtype=complex), np.zeros(self.cal.frequency.npoints, dtype=complex)),
            (self.cal.coefs['forward switch term'], np.zeros(self.cal.frequency.npoints, dtype=complex)),
            (np.zeros(self.cal.frequency.npoints, dtype=complex), self.cal.coefs['reverse switch term']),
        ]:
            with self.subTest(st=st):
                coefs = copy.deepcopy(self.cal.coefs)

                coefs['forward switch term'] = st[0]
                coefs['reverse switch term'] = st[1]

                coefs_12term = rf.convert_8term_2_12term(coefs)
                coefs_8term = rf.convert_12term_2_8term(coefs_12term)
                for k in self.cal.coefs.keys():
                    np.testing.assert_almost_equal(coefs_8term[k], coefs[k])

class TRLTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg


        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        self.X.s[:,0,0] *=1e-1
        self.Y.s[:,0,0] *=1e-1
        self.X.s[:,1,1] *=1e-1
        self.Y.s[:,1,1] *=1e-1

        actuals = [
            wg.thru( name='thru'),
            rf.two_port_reflect(wg.load(-.9-.1j),wg.load(-.9-.1j)),
            wg.attenuator(-3,True, 45,'deg')
            #wg.line(45,'deg',name='line'),
            ]
        self.actuals=actuals
        ideals = [
            wg.thru( name='thru'),
            wg.short(nports=2, name='short'),
            wg.line(90,'deg',name='line'),
            ]

        measured = [self.measure(k) for k in actuals]

        self.cal = rf.TRL(
            ideals = ideals,
            measured = measured,
            isolation = measured[1],
            switch_terms = (self.gamma_f, self.gamma_r)
            )

    def test_found_line(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[2]==self.actuals[2])

    def test_found_reflect(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[1]==self.actuals[1])

class TRLLongThruTest(EightTermTest):
    """
    Test TRL calibration with non-zero length thru.
    In this case calibration is done at thru center.
    """
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        self.X.s[:,0,0] *= 1e-1
        self.Y.s[:,0,0] *= 1e-1
        self.X.s[:,1,1] *= 1e-1
        self.Y.s[:,1,1] *= 1e-1

        # Reflect reference plane is at the thru center
        self.reflect = wg.load(-.9-.1j)
        reflect_shifted = wg.line(50, 'um') ** self.reflect

        actuals = [
            wg.line(100, 'um', name='thru'),
            rf.two_port_reflect(reflect_shifted, reflect_shifted),
            wg.line(1100, 'um', name='thru'),
            ]

        self.actuals = actuals

        ideals = [
            actuals[0],
            wg.short(nports=2, name='short'),
            actuals[2]
            ]

        measured = [self.measure(k) for k in actuals]

        # Calibration is done at the center of the thru
        # Add half thru to error networks so that tests pass
        self.X = self.X ** wg.line(50, 'um')
        self.Y = wg.line(50, 'um') ** self.Y

        self.cal = rf.TRL(
            ideals = ideals,
            measured = measured,
            isolation = measured[1],
            switch_terms = (self.gamma_f, self.gamma_r)
            )

    def test_found_line(self):
        self.cal.run()
        # Solved line is difference between line and thru
        self.assertTrue(self.cal.ideals[2]==self.actuals[2] ** self.actuals[0].inv)

    def test_found_reflect(self):
        self.cal.run()
        # Solved reflect is at the thru center
        self.assertTrue(self.cal.ideals[1]==rf.two_port_reflect(self.reflect, self.reflect))

class TRLWithNoIdealsTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        self.X.s[:,0,0] *=1e-1
        self.Y.s[:,0,0] *=1e-1
        self.X.s[:,1,1] *=1e-1
        self.Y.s[:,1,1] *=1e-1

        r = wg.delay_short(20,'deg')

        self.actuals=[wg.thru( name='thru'),
                      rf.two_port_reflect(r,r),\
                      wg.attenuator(-3,True, 45,'deg')]

        measured = [self.measure(k) for k in self.actuals]

        self.cal = rf.TRL(
            ideals = None,
            measured = measured,
            isolation = measured[1],
            switch_terms = (self.gamma_f, self.gamma_r)
            )

    def test_found_line(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[2]==self.actuals[2])

    def test_found_reflect(self):
        self.cal.run()
        self.assertTrue(self.cal.ideals[1]==self.actuals[1])


class TRLMultiline(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        # make error networks have s21,s12 >> s11,s22 so that TRL
        # can guess at line length
        #self.X.s[:,0,0] *=1e-1
        #self.Y.s[:,0,0] *=1e-1
        #self.X.s[:,1,1] *=1e-1
        #self.Y.s[:,1,1] *=1e-1

        actuals = [
            wg.thru( name='thru'),
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.attenuator(-3,True, 45,'deg'),
            wg.attenuator(-6,True, 90,'deg'),
            wg.attenuator(-8,True, 145,'deg'),
            ]
        self.actuals=actuals

        measured = [self.measure(k) for k in actuals]

        self.cal = rf.TRL(
            ideals = [None, -1,1,None,None,None],
            measured = measured,
            isolation = measured[1],
            switch_terms = (self.gamma_f, self.gamma_r),
            n_reflects=2,
            )

    def test_found_line(self):
        self.cal.run()
        for k in range(2,5):
            self.assertTrue(self.cal.ideals[k]==self.actuals[k])


class NISTMultilineTRLTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name = 'Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')

        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        actuals = [
            wg.thru(),
            rf.two_port_reflect(wg.load(-.98-.1j),wg.load(-.98-.1j)),
            rf.two_port_reflect(wg.load(.99+0.05j),wg.load(.99+0.05j)),
            wg.line(100,'um'),
            wg.line(200,'um'),
            wg.line(900,'um'),
            ]

        self.actuals=actuals

        measured = [self.measure(k) for k in actuals]

        self.cal = NISTMultilineTRL(
            measured = measured,
            isolation = measured[1],
            Grefls = [-1, 1],
            l = [0, 100e-6, 200e-6, 900e-6],
            er_est = 1,
            switch_terms = (self.gamma_f, self.gamma_r),
            gamma_root_choice = 'real'
            )

    def test_gamma(self):
        self.assertTrue(max(np.abs(self.wg.gamma-self.cal.gamma)) < 1e-3)


class NISTMultilineTRLTest2(NISTMultilineTRLTest):
    """ Test characteristic impedance change and reference plane shift.
    """
    def setUp(self):
        global NPTS
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        rng = np.random.default_rng()

        r = rng.uniform(10,100,NPTS)
        l = 1e-9*rng.uniform(100,200,NPTS)
        g = np.zeros(NPTS)
        c = 1e-12*rng.uniform(100,200,NPTS)

        rlgc = DistributedCircuit(frequency=wg.frequency, z0=None, R=r, L=l, G=g, C=c)
        self.rlgc = rlgc

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name = 'Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        actuals = [
            rlgc.line(1000,'um'),
            rlgc.line(500,'um') ** rlgc.short(nports=2) ** rlgc.line(500,'um'),
            rlgc.line(1010,'um'),
            rlgc.line(1100,'um'),
            rlgc.line(1800,'um'),
            ]

        self.actuals=actuals

        measured = [self.measure(k) for k in actuals]
        self.measured = measured

        self.cal = NISTMultilineTRL(
            measured = measured,
            isolation = measured[1],
            Grefls = [-1],
            refl_offset = 500e-6,
            l = [1000e-6, 1010e-6, 1100e-6, 1800e-6],
            switch_terms = (self.gamma_f, self.gamma_r),
            c0=c,
            z0_ref=50,
            gamma_root_choice = 'real'
            )

        self.cal_shift = NISTMultilineTRL(
            measured = measured,
            isolation = measured[1],
            Grefls = [-1],
            refl_offset = 0,
            l = [0, 10e-6, 100e-6, 800e-6],
            switch_terms = (self.gamma_f, self.gamma_r),
            ref_plane = -500e-6,
            c0=c,
            z0_ref=50,
            gamma_root_choice = 'real'
            )

    def test_gamma(self):
        self.assertTrue(max(np.abs(self.rlgc.gamma-self.cal.gamma)) < 1e-3)

    def test_z0(self):
        self.assertTrue(max(np.abs(self.rlgc.z0-self.cal.z0)) < 1e-3)

    def test_shift(self):
        for k in self.cal.coefs.keys():
            self.assertTrue(all(np.abs(self.cal.coefs[k] - self.cal_shift.coefs[k]) < 1e-9))


    def test_numpy_float_arguments(self):
        # see gh-895
        cal = NISTMultilineTRL(
            measured = self.measured[:3],
            Grefls = [-1],
            l = [np.float64(1000e-6), 1010e-6],
            switch_terms = (self.gamma_f, self.gamma_r),
            )
        cal.run()
        cal.apply_cal(self.measured[0])

        cal = NISTMultilineTRL(
            measured = self.measured[:3],
            Grefls = [-1],
            l = [1000e-6, 1010e-6],
            z0_ref = np.float64(50),
            z0_line = np.float64(50),
            switch_terms = (self.gamma_f, self.gamma_r),
            )
        cal.run()
        cal.apply_cal(self.measured[0])

class TUGMultilineTest(EightTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg

        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name = 'Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')

        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        actuals = [
            wg.thru(),
            rf.two_port_reflect(wg.load(-.98-.1j),wg.load(-.98-.1j)),
            rf.two_port_reflect(wg.load(.99+0.05j),wg.load(.99+0.05j)),
            wg.line(100,'um'),
            wg.line(200,'um'),
            wg.line(900,'um'),
            ]

        self.actuals=actuals

        measured = [self.measure(k) for k in actuals]

        self.cal = TUGMultilineTRL(
            line_meas = [measured[0]] + measured[3:],
            line_lengths = [0, 100e-6, 200e-6, 900e-6],
            er_est = 1,
            reflect_meas = measured[1:3],
            reflect_est = [-1, 1],
            isolation = measured[1],
            switch_terms = (self.gamma_f, self.gamma_r)
            )

    def test_gamma(self):
        self.assertTrue(max(np.abs(self.wg.gamma-self.cal.gamma)) < 1e-3)

@pytest.mark.skip()
class TREightTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg= self.wg


        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')


        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]

        measured = [self.measure_std(k) for k in ideals]

        cal1 = rf.TwoPortOnePath(
            ideals = ideals,
            measured = measured
            )
        switch_terms = (cal1.coefs_ntwks['forward switch term'],
                        cal1.coefs_ntwks['reverse switch term'])


        measured = [self.measure(k) for k in ideals]
        self.cal = rf.EightTerm(
            ideals = ideals,
            measured = measured,
            switch_terms = switch_terms,
            )
        raise ValueError()

    def measure_std(self,ntwk):
        r= self.wg.random(2)
        m = ntwk.copy()
        mf = self.X**ntwk**self.Y

        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,1,1] = r.s[:,1,1]
        m.s[:,0,1] = r.s[:,0,1]
        return m
    def measure(self,ntwk):

        m = ntwk.copy()
        mf = self.X**ntwk**self.Y
        mr = self.X**ntwk.flipped()**self.Y

        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,1,1] = mr.s[:,0,0]
        m.s[:,0,1] = mr.s[:,1,0]
        return m


class TwelveTermTest(unittest.TestCase, CalibrationTest):
    """
    This test verifies the accuracy of the SOLT calibration. Generating
    measured networks requires different error networks for forward and
    reverse excitation states, these are described as follows

    forward excitation
        used for S21 and S11
        Mf = Xf ** S ** Yf

    reverse excitation
        used for S12 and S22
        Mr = Xr ** S ** Yr


    """
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
        self.If = wg.random(n_ports =1, name='If')
        self.Ir = wg.random(n_ports =1, name='Ir')

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.random(2,name='rand1'),
            wg.random(2,name='rand2'),
            ]

        measured = [ self.measure(k) for k in ideals]

        self.cal = rf.TwelveTerm(
            ideals = ideals,
            measured = measured,
            n_thrus=2,
            isolation=measured[2]
            )

    def measure(self,ntwk):
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf
        mr = self.Xr**ntwk**self.Yr
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,0,1] = mr.s[:,0,1]
        m.s[:,1,1] = mr.s[:,1,1]

        #Leakage
        m.s[:,1,0] += self.If.s[:,0,0]
        m.s[:,0,1] += self.Ir.s[:,0,0]
        return m

    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.Xf.s11,
            self.cal.coefs_ntwks['forward directivity'])

    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.Xf.s22 ,
            self.cal.coefs_ntwks['forward source match'] )

    def test_forward_load_match_accuracy(self):
        self.assertEqual(
            self.Yf.s11 ,
            self.cal.coefs_ntwks['forward load match'])

    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Xf.s21 * self.Xf.s12 ,
            self.cal.coefs_ntwks['forward reflection tracking'])

    def test_forward_transmission_tracking_accuracy(self):
        self.assertEqual(
            self.Xf.s21*self.Yf.s21 ,
            self.cal.coefs_ntwks['forward transmission tracking'])

    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Yr.s11 ,
            self.cal.coefs_ntwks['reverse source match']   )

    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Yr.s22 ,
            self.cal.coefs_ntwks['reverse directivity']  )

    def test_reverse_load_match_accuracy(self):
        self.assertEqual(
            self.Xr.s22 ,
            self.cal.coefs_ntwks['reverse load match'])

    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Yr.s21 * self.Yr.s12 ,
            self.cal.coefs_ntwks['reverse reflection tracking'])

    def test_reverse_transmission_tracking_accuracy(self):
        self.assertEqual(
            self.Yr.s12*self.Xr.s12 ,
            self.cal.coefs_ntwks['reverse transmission tracking'])

    def test_forward_isolation_accuracy(self):
        self.assertEqual(
            self.If.s11,
            self.cal.coefs_ntwks['forward isolation'])

    def test_reverse_isolation_accuracy(self):
        self.assertEqual(
            self.Ir.s11,
            self.cal.coefs_ntwks['reverse isolation'])

    def test_convert_12term_2_8term(self):
        converted = rf.convert_8term_2_12term(
                    rf.convert_12term_2_8term(self.cal.coefs, redundant_k=True))

        for k in converted:
            self.assertTrue(abs(self.cal.coefs[k] - converted[k])<1e-9)

    def test_convert_12term_2_8term_correction_accuracy(self):
        converted = rf.convert_8term_2_12term(
                    rf.convert_12term_2_8term(self.cal.coefs, redundant_k=True))

        self.cal._coefs = converted
        a = self.wg.random(n_ports=2)
        m = self.measure(a)
        c = self.cal.apply_cal(m)

        self.assertEqual(a,c)

    def test_coefs_8term(self):
        self.assertEqual(self.cal.coefs_8term, rf.convert_12term_2_8term(self.cal.coefs))

    def test_coefs_12term(self):
        self.assertEqual(self.cal.coefs_12term, self.cal.coefs)


class TwelveTermSloppyInitTest(TwelveTermTest):
    """
    Test the TwelveTerm.__init__'s ability to
    1) determine the number of thrus (n_thrus) heuristically
    2) put the standards in correct order if they use sloppy_input

    It must be a entirely separate test because we want to ensure it
    creates an accurate calibration.
    """
    @suppress_warning_decorator("dictionary passed, sloppy_input")
    @suppress_warning_decorator("n_thrus is None")
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
        #No leakage as it can interfere with thru detection, which is done
        #based on S21 and S12
        self.If = wg.match(n_ports =1, name='If')
        self.Ir = wg.match(n_ports =1, name='Ir')

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.attenuator(-20,name='atten'),
            wg.line(45,'deg',name = 'line') ,
            ]


        measured = [ self.measure(k) for k in ideals]

        self.cal= TwelveTerm(
            ideals = NetworkSet(ideals).to_dict(),
            measured = NetworkSet(measured).to_dict(),
            n_thrus=None,
            )

    def measure(self,ntwk):
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf
        mr = self.Xr**ntwk**self.Yr
        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,0,1] = mr.s[:,0,1]
        m.s[:,1,1] = mr.s[:,1,1]
        return m


class SOLTTest(TwelveTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Xr = wg.random(n_ports =2, name = 'Xr')
        self.Yf = wg.random(n_ports =2, name='Yf')
        self.Yr = wg.random(n_ports =2, name='Yr')
        self.If = wg.random(n_ports =1, name='If')
        self.Ir = wg.random(n_ports =1, name='Ir')

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            None,
            ]
        actuals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(),
            ]

        measured = [ self.measure(k) for k in actuals]

        self.cal = SOLT(
            ideals = ideals,
            measured = measured,
            n_thrus=1,
            isolation=measured[2]
            )


class TwoPortOnePathTest(TwelveTermTest):
    @suppress_warning_decorator("divide by zero encountered in log10")
    @suppress_warning_decorator("n_thrus is None")
    def setUp(self):
        self.n_ports = 2
        self.wg =WG
        wg  = self.wg
        self.Xf = wg.random(n_ports =2, name = 'Xf')
        self.Yf = wg.random(n_ports =2, name='Yf')

        #No leakage
        self.If = wg.match(n_ports =1, name='If')
        self.Ir = wg.match(n_ports =1, name='Ir')

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.random(2,name='rand1'),
            wg.random(2,name='rand2'),
            ]

        measured = [ self.measure(k) for k in ideals]
        self.cal = TwoPortOnePath(
            ideals = ideals,
            measured = measured,
            source_port=1,
            #n_thrus=2,
            )

    def measure(self,ntwk):
        r = self.wg.match(2)
        m = ntwk.copy()
        mf = self.Xf**ntwk**self.Yf

        m.s[:,1,0] = mf.s[:,1,0]
        m.s[:,0,0] = mf.s[:,0,0]
        m.s[:,1,1] = r.s[:,1,1]
        m.s[:,0,1] = r.s[:,0,1]
        return m

    def test_accuracy_of_dut_correction(self):
        a = self.wg.random(n_ports=self.n_ports, name = 'actual')
        f = self.measure(a)
        r = self.measure(a.flipped())
        c = self.cal.apply_cal((f,r))
        c.name = 'corrected'
        self.assertEqual(c,a)

    def test_embed_then_apply_cal(self):
        a = self.wg.random(n_ports=self.n_ports)
        f = self.cal.embed(a)
        r = self.cal.embed(a.flipped())
        self.assertEqual(self.cal.apply_cal((f,r)),a)

    @pytest.mark.skip(reason='measurement procedure is different so this test doesnt apply')
    def test_embed_equal_measure(self):
        pass

    @suppress_warning_decorator("n_thrus is None")
    def test_from_coefs(self):
        cal_from_coefs = self.cal.from_coefs(self.cal.frequency, self.cal.coefs)
        ntwk = self.wg.random(n_ports=self.n_ports)

    @suppress_warning_decorator("n_thrus is None")
    def test_from_coefs_ntwks(self):
        cal_from_coefs = self.cal.from_coefs_ntwks(self.cal.coefs_ntwks)

    @pytest.mark.skip()
    def test_reverse_source_match_accuracy(self):
        pass

    @pytest.mark.skip()
    def test_reverse_directivity_accuracy(self):
        pass

    @pytest.mark.skip()
    def test_reverse_load_match_accuracy(self):
        pass

    @pytest.mark.skip()
    def test_reverse_reflection_tracking_accuracy(self):
        pass

    @pytest.mark.skip()
    def test_reverse_transmission_tracking_accuracy(self):
        pass

    @pytest.mark.skip()
    def test_convert_12term_2_8term_correction_accuracy(self):
        pass


class UnknownThruTest(EightTermTest):
    def setUp(self):

        self.n_ports = 2
        self.wg = WG
        wg= self.wg
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        #No leakage
        self.If = wg.match(n_ports=1, name='If')
        self.Ir = wg.match(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')


        actuals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='match'),
            wg.impedance_mismatch(50,45)**wg.line(20,'deg',name='line')**wg.impedance_mismatch(45,50)

            ]

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='match'),
            wg.thru(name='thru'),
            ]

        measured = [self.measure(k) for k in actuals]

        self.cal = rf.UnknownThru(
            ideals = ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r]
            )

class LRMTest(EightTermTest):
    def setUp(self):

        self.n_ports = 2
        self.wg = WG
        wg = self.wg
        self.X = wg.random(n_ports=2, name='X')
        self.Y = wg.random(n_ports=2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports=1, name='gamma_f')
        self.gamma_r = wg.random(n_ports=1, name='gamma_r')

        # Our guess of the standards
        ss_i = wg.short(nports=2, name='short')
        thru = wg.line(d=50, unit='um', name='thru')

        # Actual reflects with parasitics
        s = wg.inductor(5e-12) ** wg.load(-0.95, nports=1, name='short')
        ss = rf.two_port_reflect(s, s)

        m = wg.load(0, nports=1, name='match')
        mm = rf.two_port_reflect(m, m)

        # Store reflect for other tests
        self.s = s

        approx_ideals = [
            thru,
            ss_i,
            mm
            ]

        ideals = [
            thru,
            ss,
            mm
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.LRM(
            ideals = approx_ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r],
            isolation = measured[2]
            )

    def test_solved_r(self):
        self.assertTrue(all(np.abs(self.s.s - self.cal.solved_r.s) < 1e-7))

class LRRMTest(EightTermTest):
    def setUp(self):

        self.n_ports = 2
        self.wg = WG
        wg = self.wg
        self.X = wg.random(n_ports=2, name='X')
        self.Y = wg.random(n_ports=2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports=1, name='gamma_f')
        self.gamma_r = wg.random(n_ports=1, name='gamma_r')

        # Our guess of the standards
        o_i = wg.load(1, nports=1, name='open')
        s_i = wg.short(nports=1, name='short')
        m_i = wg.load(0.1, nports=1, name='load')
        thru = wg.line(d=50, z0=75, unit='um', name='thru')
        thru.renormalize(wg.z0)

        # Make sure calibration works with non-symmetric thru
        thru.s[:,1,1] += 0.02 + 0.05j

        # Actual reflects with parasitics
        s = wg.inductor(5e-12) ** wg.load(-0.95, nports=1, name='short')
        o = wg.shunt_capacitor(5e-15) ** wg.open(nports=1, name='open')

        self.match_l = np.random.default_rng().uniform(1e-12, 20e-12)
        l = wg.inductor(L=self.match_l)
        m = l**m_i

        # Make standards two-ports
        oo = rf.two_port_reflect(o, o)
        ss = rf.two_port_reflect(s, s)
        mm = rf.two_port_reflect(m, o)

        oo_i = rf.two_port_reflect(o_i, o_i)
        ss_i = rf.two_port_reflect(s_i, s_i)
        mm_i = rf.two_port_reflect(m_i, m_i)

        # Store open and short for other tests.
        self.s = s
        self.o = o

        approx_ideals = [
            thru,
            ss_i,
            oo_i,
            mm_i
            ]

        ideals = [
            thru,
            ss,
            oo,
            mm
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.LRRM(
            ideals = approx_ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r],
            isolation = measured[3]
            )

    # Test the solved standards, don't use exact equality because of inductance
    # fitting tolerance.
    def test_solved_inductance(self):
        solved_l = np.mean(self.cal.solved_l)
        self.assertTrue(np.abs(self.match_l - solved_l) < 1e-3*self.match_l)

    def test_solved_r1(self):
        self.assertTrue(all(np.abs(self.s.s - self.cal.solved_r1.s) < 1e-7))

    def test_solved_r2(self):
        self.assertTrue(all(np.abs(self.o.s - self.cal.solved_r2.s) < 1e-7))

class LRRMTestNoFit(LRRMTest):
    def setUp(self):

        self.n_ports = 2
        self.wg = WG
        wg = self.wg
        self.X = wg.random(n_ports=2, name='X')
        self.Y = wg.random(n_ports=2, name='Y')
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports=1, name='gamma_f')
        self.gamma_r = wg.random(n_ports=1, name='gamma_r')

        # Our guess of the standards
        o_i = wg.load(1, nports=1, name='open')
        s_i = wg.short(nports=1, name='short')
        # Doesn't work correctly for non-50 ohm match.
        m_i = wg.load(0, nports=1, name='load')
        thru = wg.line(d=50, unit='um', name='thru')

        # Actual reflects with parasitics
        s = wg.inductor(5e-12) ** wg.load(-0.95, nports=1, name='short')
        o = wg.shunt_capacitor(5e-15) ** wg.open(nports=1, name='open')

        self.match_l = np.random.default_rng().uniform(1e-12, 20e-12)
        l = wg.inductor(L=self.match_l)
        m = l**m_i

        # Make standards two-ports
        oo = rf.two_port_reflect(o, o)
        ss = rf.two_port_reflect(s, s)
        mm = rf.two_port_reflect(m, o)

        oo_i = rf.two_port_reflect(o_i, o_i)
        ss_i = rf.two_port_reflect(s_i, s_i)
        mm_i = rf.two_port_reflect(m_i, m_i)

        # Store open and short for other tests.
        self.s = s
        self.o = o

        approx_ideals = [
            thru,
            ss_i,
            oo_i,
            mm_i
            ]

        ideals = [
            thru,
            ss,
            oo,
            mm
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.LRRM(
            ideals = approx_ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r],
            isolation = measured[3],
            match_fit = 'none'
            )

class MRCTest(EightTermTest):
    def setUp(self):

        self.n_ports = 2
        #Exact only with a lossless waveguide
        self.wg = WG_lossless
        wg= self.wg
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        #No leakage
        self.If = wg.match(n_ports=1, name='If')
        self.Ir = wg.match(n_ports=1, name='Ir')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')


        def delay_shorts(d1,d2):
            ds1 = wg.delay_short(d1,'deg')
            ds2 = wg.delay_short(d2,'deg')
            return rf.two_port_reflect(ds1,ds2)

        actuals = [
            wg.short(nports=2, name='short'),
            delay_shorts(65,130),
            delay_shorts(120,75),
            wg.load(.2+.2j,nports=2, name='match'),
            wg.impedance_mismatch(50,45)**wg.line(20,'deg',name='line')**wg.impedance_mismatch(45,50)

            ]

        ideals = [
            wg.short(nports=2, name='short'),
            delay_shorts(45,90),
            delay_shorts(90,45),
            wg.load(.2+.2j,nports=2, name='match'),
            wg.thru(name='thru'),
            ]

        measured = [self.measure(k) for k in actuals]

        self.cal = rf.MRC(
            ideals = ideals,
            measured = measured,
            switch_terms = [self.gamma_f, self.gamma_r]
            )


class TwelveTermToEightTermTest(unittest.TestCase, CalibrationTest):
    """
    This test verifies the accuracy of the SOLT calibration, when used
    on an error-box (8-term) model.


    """
    @suppress_warning_decorator("n_thrus is None")
    def setUp(self):
        self.n_ports = 2
        wg= rf.wr10
        wg.frequency = rf.F.from_f([100], unit='GHz')
        self.wg = wg
        self.X = wg.random(n_ports =2, name = 'X')
        self.Y = wg.random(n_ports =2, name='Y')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')
        #Isolation terms
        self.If = wg.random(n_ports=1, name='If')
        self.Ir = wg.random(n_ports=1, name='Ir')

        ideals = [
            wg.short(nports=2, name='short'),
            wg.open(nports=2, name='open'),
            wg.match(nports=2, name='load'),
            wg.thru(name='thru'),
            ]

        measured = [ self.measure(k) for k in ideals]
        with warnings.catch_warnings(record=False):
            self.cal = rf.TwelveTerm(
            ideals = ideals,
            measured = measured,
            isolation=measured[2]
            )


        coefs = rf.calibration.convert_12term_2_8term(self.cal.coefs)
        coefs = NetworkSet.from_s_dict(coefs,
                                    frequency=self.cal.frequency).to_dict()
        self.coefs= coefs

    def terminate(self, ntwk):
        """
        terminate a measured network with the switch terms
        """
        return terminate(ntwk,self.gamma_f, self.gamma_r)

    def measure(self,ntwk):
        out =  self.terminate(self.X**ntwk**self.Y)
        out.name = ntwk.name
        out.s[:,1,0] += self.If.s[:,0,0]
        out.s[:,0,1] += self.Ir.s[:,0,0]
        return out

    def test_forward_isolation(self):
        self.assertEqual(self.coefs['forward isolation'], self.If.s11)

    def test_reverse_isolation(self):
        self.assertEqual(self.coefs['reverse isolation'], self.Ir.s11)

    def test_forward_switch_term(self):
        self.assertEqual(self.coefs['forward switch term'], self.gamma_f)

    def test_reverse_switch_term(self):
        self.assertEqual(self.coefs['reverse switch term'], self.gamma_r)

    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.X.s11,
            self.coefs['forward directivity'])

    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.X.s22 ,
            self.coefs['forward source match'] )

    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.X.s21 * self.X.s12 ,
            self.coefs['forward reflection tracking'])

    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Y.s11 ,
            self.coefs['reverse source match']   )

    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Y.s22 ,
            self.coefs['reverse directivity']  )

    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Y.s21 * self.Y.s12 ,
            self.coefs['reverse reflection tracking'])

    def test_k_accuracy(self):
        self.assertEqual(
            self.X.s21/self.Y.s12 ,
            self.coefs['k']  )

    def test_verify_12term(self):
        self.assertTrue(self.cal.verify_12term_ntwk.s_mag.max() < 1e-3)


class SixteenTermTest(unittest.TestCase, CalibrationTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        #Port 0: VNA port 0
        #Port 1: DUT port 0
        #Port 2: DUT port 1
        #Port 3: VNA port 1
        self.Z = wg.random(n_ports = 4, name = 'Z')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        o = wg.open(nports=1, name='open')
        s = wg.short(nports=1, name='short')
        m = wg.match(nports=1, name='load')
        om = rf.two_port_reflect(o, m)
        mo = rf.two_port_reflect(m, o)
        oo = rf.two_port_reflect(o, o)
        ss = rf.two_port_reflect(s, s)
        thru = wg.thru(name='thru')

        ideals = [
            thru,
            om,
            mo,
            oo,
            ss
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.SixteenTerm(
            measured = measured,
            ideals = ideals,
            switch_terms=(self.gamma_f, self.gamma_r)
            )

    def terminate(self, ntwk):
        """
        terminate a measured network with the switch terms
        """
        return terminate(ntwk,self.gamma_f, self.gamma_r)

    def measure(self,ntwk):
        out = self.terminate(rf.connect(self.Z, 1, ntwk, 0, num=2))
        out.name = ntwk.name
        return out

    def test_renormalize(self):
        dut = self.wg.random(n_ports=2, name='DUT')
        dut_meas = self.measure(dut)
        z_new = 100-30j
        self.cal.renormalize(50, z_new)
        dut_cal = self.cal.apply_cal(dut_meas)
        dut.renormalize(z_new, s_def="traveling")
        dut.z0 = 50
        self.assertEqual(dut, dut_cal)

    def test_renormalize2(self):
        """
        Test that roundtrip renormalization gives back the original calibration
        """
        old_coefs = copy.deepcopy(self.cal.coefs)
        z_new = 10 + 20j
        self.cal.renormalize(50, z_new)
        self.cal.renormalize(z_new, 50)
        for k in self.cal.coefs.keys():
            np.testing.assert_array_almost_equal(old_coefs[k], self.cal.coefs[k], err_msg=k)

    def test_forward_directivity_accuracy(self):
        self.assertEqual(
            self.Z.s11,
            self.cal.coefs_ntwks['forward directivity'])

    def test_forward_source_match_accuracy(self):
        self.assertEqual(
            self.Z.s22 ,
            self.cal.coefs_ntwks['forward source match'] )

    def test_forward_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Z.s21 * self.Z.s12 ,
            self.cal.coefs_ntwks['forward reflection tracking'])

    def test_reverse_source_match_accuracy(self):
        self.assertEqual(
            self.Z.s33 ,
            self.cal.coefs_ntwks['reverse source match']   )

    def test_reverse_directivity_accuracy(self):
        self.assertEqual(
            self.Z.s44 ,
            self.cal.coefs_ntwks['reverse directivity']  )

    def test_reverse_reflection_tracking_accuracy(self):
        self.assertEqual(
            self.Z.s34 * self.Z.s43 ,
            self.cal.coefs_ntwks['reverse reflection tracking'])

    def test_forward_isolation(self):
        self.assertEqual(
            self.Z.s41 ,
            self.cal.coefs_ntwks['forward isolation'])

    def test_reverse_isolation(self):
        self.assertEqual(
            self.Z.s14 ,
            self.cal.coefs_ntwks['reverse isolation'])

    def test_reverse_port_isolation(self):
        self.assertEqual(
            self.Z.s23 ,
            self.cal.coefs_ntwks['reverse port isolation'])

    def test_forward_port_isolation(self):
        self.assertEqual(
            self.Z.s32 ,
            self.cal.coefs_ntwks['forward port isolation'])

    def test_k(self):
        self.assertEqual(
            self.Z.s21/self.Z.s34 ,
            self.cal.coefs_ntwks['k'])

    def test_forward_port1_isolation(self):
        self.assertEqual(
            self.Z.s31/self.Z.s34 ,
            self.cal.coefs_ntwks['forward port 1 isolation'])

    def test_reverse_port1_isolation(self):
        self.assertEqual(
            self.Z.s13*self.Z.s34 ,
            self.cal.coefs_ntwks['reverse port 1 isolation'])

    def test_forward_port2_isolation(self):
        self.assertEqual(
            self.Z.s42*self.Z.s34 ,
            self.cal.coefs_ntwks['forward port 2 isolation'])

    def test_reverse_port2_isolation(self):
        self.assertEqual(
            self.Z.s24/self.Z.s34 ,
            self.cal.coefs_ntwks['reverse port 2 isolation'])

class SixteenTermCoefficientsTest(unittest.TestCase):
    """Test that 16-term non-isolation coefficients are defined the same way as
    8-term coefficients."""
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        #Port 0: VNA port 0
        #Port 1: DUT port 0
        #Port 2: DUT port 1
        #Port 3: VNA port 1
        self.Z = wg.random(n_ports = 4, name = 'Z')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        o = wg.open(nports=1, name='open')
        s = wg.short(nports=1, name='short')
        m = wg.match(nports=1, name='load')
        om = rf.two_port_reflect(o, m)
        mo = rf.two_port_reflect(m, o)
        oo = rf.two_port_reflect(o, o)
        ss = rf.two_port_reflect(s, s)
        thru = wg.thru(name='thru')

        ideals = [
            thru,
            om,
            mo,
            oo,
            ss
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal16 = rf.SixteenTerm(
            measured = measured,
            ideals = ideals,
            switch_terms=(self.gamma_f, self.gamma_r)
            )

        r = wg.load(.95+.1j,nports=1)
        m = wg.match(nports=1)
        mm = rf.two_port_reflect(m, m)
        rm = rf.two_port_reflect(r, m)
        mr = rf.two_port_reflect(m, r)
        rr = rf.two_port_reflect(r, r)

        ideals = [
            thru,
            mm,
            rr,
            rm,
            mr
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal_lmr16 = rf.LMR16(
            measured = measured,
            ideals = [thru],
            ideal_is_reflect = False,
             #Automatic sign detection doesn't work if the
             #error terms aren't symmetric enough
            sign = 1,
            switch_terms=(self.gamma_f, self.gamma_r)
            )

        #Same error network, but without leakage terms

        #Primary leakage
        self.Z.s[:,3,0] = 0 # forward isolation
        self.Z.s[:,0,3] = 0 # reverse isolation
        self.Z.s[:,2,1] = 0 # forward port isolation
        self.Z.s[:,1,2] = 0 # reverse port isolation

        #Cross leakage
        self.Z.s[:,3,1] = 0 # forward port 2 isolation
        self.Z.s[:,1,3] = 0 # reverse port 2 isolation
        self.Z.s[:,2,0] = 0 # forward port 1 isolation
        self.Z.s[:,0,2] = 0 # reverse port 1 isolation

        measured = [self.measure(k) for k in ideals]

        self.cal8 = rf.EightTerm(
            measured = measured,
            ideals = ideals,
            switch_terms=(self.gamma_f, self.gamma_r),
            )

    def terminate(self, ntwk):
        """
        terminate a measured network with the switch terms
        """
        return terminate(ntwk,self.gamma_f, self.gamma_r)

    def measure(self,ntwk):
        out = self.terminate(rf.connect(self.Z, 1, ntwk, 0, num=2))
        out.name = ntwk.name
        return out

    def test_coefficients(self):
        for k in self.cal8.coefs.keys():
            if k in self.cal16.coefs.keys():
                if 'isolation' in k:
                    continue
                self.assertTrue(all(np.abs(self.cal8.coefs[k] - self.cal16.coefs[k]) < 1e-10))
                self.assertTrue(all(np.abs(self.cal8.coefs[k] - self.cal_lmr16.coefs[k]) < 1e-10))


class LMR16Test(SixteenTermTest):
    def setUp(self):
        self.n_ports = 2
        self.wg = WG
        wg = self.wg

        #Port 0: VNA port 0
        #Port 1: DUT port 0
        #Port 2: DUT port 1
        #Port 3: VNA port 1
        self.Z = wg.random(n_ports = 4, name = 'Z')
        self.gamma_f = wg.random(n_ports =1, name='gamma_f')
        self.gamma_r = wg.random(n_ports =1, name='gamma_r')

        r = wg.short(nports=1, name='short')
        m = wg.match(nports=1, name='load')
        mm = rf.two_port_reflect(m, m)
        rm = rf.two_port_reflect(r, m)
        mr = rf.two_port_reflect(m, r)
        rr = rf.two_port_reflect(r, r)

        thru_length = np.random.default_rng().uniform(0,10)
        thru = wg.line(thru_length,'deg',name='line')

        self.thru = thru

        ideals = [
            thru,
            mm,
            rr,
            rm,
            mr
            ]

        measured = [self.measure(k) for k in ideals]

        self.cal = rf.LMR16(
            measured = measured,
            ideals = [r],
            ideal_is_reflect = True,
             #Automatic sign detection doesn't work if the
             #error terms aren't symmetric enough
            sign = -1,
            switch_terms=(self.gamma_f, self.gamma_r)
            )

    def test_solved_through(self):
        self.assertEqual(
            self.thru,
            self.cal.solved_through)

class MultiportCalTest(unittest.TestCase):
    """Multi-port NISTMultilineTRL calibration test"""

    def test_cal(self):
        self.wg = WG
        wg = self.wg

        self.n_ports = 3
        nports = self.n_ports

        self.make_error_networks(wg, nports)

        cal_dict = {}

        actuals = [
            wg.thru(),
            rf.two_port_reflect(wg.load(-.98-.1j),wg.load(-.98-.1j)),
            wg.line(100,'um'),
            wg.line(900,'um'),
            ]

        dut = self.wg.random(n_ports=self.n_ports, name='dut')
        dut_meas = self.measure(dut)

        l = [0, 100e-6, 900e-6]

        for p in [(0, 1), (0, 2)]:

            measured = [self.measure(rf.twoport_to_nport(k, p[0], p[1], nports)) for k in actuals]
            cal_dict[p] = {}
            cal_dict[p]['method'] = NISTMultilineTRL
            cal_dict[p]['measured'] = measured
            # Two-port switch terms are in reverse order from multi-port switch terms.
            cal_dict[p]['switch_terms'] = [self.gammas[i] for i in p][::-1]
            cal_dict[p]['l'] = l
            cal_dict[p]['Grefls'] = [-1]
            cal_dict[p]['er_est'] = 1
            cal_dict[p]['gamma_root_choice'] = 'real'

        isolation = self.measure(wg.match(nports=nports))

        self.cal = MultiportCal(cal_dict, isolation=isolation)

        # Test coefs
        nports = self.n_ports
        for e, c in enumerate(self.cal.coefs):
            assert np.allclose(c['directivity'], self.Z.s[:,e,e])
            assert np.allclose(c['source match'], self.Z.s[:,nports+e,nports+e])
            assert np.allclose(c['reflection tracking'], self.Z.s[:,e,nports+e] * self.Z.s[:,nports+e,e])
            assert np.allclose(c['switch term'], self.gammas[e].s)
            assert np.allclose(c['k']/self.cal.coefs[0]['k'], self.Z.s[:,nports+0,0]/self.Z.s[:,nports+e,e])

        # Test DUT correction
        dut_cal = self.cal.apply_cal(dut_meas)
        self.assertEqual(dut_cal, dut)

        # Test cal inverts embed
        a = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(self.cal.apply_cal(self.cal.embed(a)),a)

        # Test embed equals measure
        a = self.wg.random(n_ports=self.n_ports)
        self.assertEqual(self.cal.embed(a),self.measure(a))

        # Test gamma solved by TRL
        for p in [(0, 1), (0, 2)]:
            self.assertTrue(max(np.abs(self.wg.gamma-self.cal.cals[p].gamma)) < 1e-3)

    def make_error_networks(self, wg, nports):
        self.Z = wg.random(n_ports = 2*nports, name = 'Z')

        # Isolation terms are between all ports.
        # No error in the same port.
        self.isolation = wg.random(n_ports=nports, name='I')
        for i in range(nports):
            self.isolation.s[:, i, i] = 0

        def port_type(n):
            return 'VNA' if n < nports else 'DUT'
        def port_number(n):
            return n if n < nports else n - nports
        # Remove leakage terms
        for i in range(2*nports):
            for j in range(i+1, 2*nports):
                # No connection between different VNA/DUT ports.
                # No connection between VNA and DUT ports with different number.
                if port_type(i) == port_type(j) or port_number(i) != port_number(j):
                    self.Z.s[:,i,j] = 0
                    self.Z.s[:,j,i] = 0

        self.gammas = []
        for i in range(nports):
            self.gammas.append(wg.random(n_ports=1, name=f'gamma_{i}'))

    def terminate(self, ntwk):
        """
        Terminate a measured network with the switch terms
        """
        return terminate_nport(ntwk, self.gammas)

    def measure(self, ntwk):
        out = self.terminate(rf.connect(self.Z, self.n_ports, ntwk, 0, num=self.n_ports))
        out = out + self.isolation
        out.name = ntwk.name
        return out

class MultiportSOLTTest(MultiportCalTest):

    def test_cal(self):
        nport_list = [3, 4]
        method_list = [SOLT, UnknownThru]

        self.wg = WG
        wg = self.wg

        for nport in nport_list:
            self.n_ports = nport
            nports = self.n_ports

            self.make_error_networks(wg, nports)

            o = wg.open(nports=nports, name='open')
            s = wg.short(nports=nports, name='short')
            m = wg.match(nports=nports, name='load')

            # thru = wg.thru(name='thru')
            thru = wg.impedance_mismatch(50, 45) ** wg.line(20, 'deg') ** wg.impedance_mismatch(45, 50)

            ideals = []
            # nports-1 thrus from port 0 to all other ports.
            for i in range(1, nports):
                thru_i = rf.twoport_to_nport(thru, 0, i, nports)
                ideals.append(thru_i)

            ideals.extend([o,s,m])
            measured = [self.measure(k) for k in ideals]

            dut = self.wg.random(n_ports=self.n_ports, name='dut')
            dut_meas = self.measure(dut)

            for method in method_list:

                if method == SOLT:
                    self.cal = MultiportSOLT(method, measured, ideals, isolation=measured[-1])
                else:
                    self.cal = MultiportSOLT(method, measured, ideals, isolation=measured[-1], switch_terms=self.gammas)

                # Test coefs
                nports = self.n_ports
                for e, c in enumerate(self.cal.coefs):
                    assert np.allclose(c['directivity'], self.Z.s[:,e,e])
                    assert np.allclose(c['source match'], self.Z.s[:,nports+e,nports+e])
                    assert np.allclose(c['reflection tracking'], self.Z.s[:,e,nports+e] * self.Z.s[:,nports+e,e])
                    assert np.allclose(c['switch term'], self.gammas[e].s)
                    assert np.allclose(c['k']/self.cal.coefs[0]['k'], self.Z.s[:,nports+0,0]/self.Z.s[:,nports+e,e])

                # Test DUT correction
                dut_cal = self.cal.apply_cal(dut_meas)
                self.assertEqual(dut_cal, dut)

                # Test cal inverts embed
                a = self.wg.random(n_ports=self.n_ports)
                self.assertEqual(self.cal.apply_cal(self.cal.embed(a)),a)

                # Test embed equals measure
                a = self.wg.random(n_ports=self.n_ports)
                self.assertEqual(self.cal.embed(a),self.measure(a))

if __name__ == "__main__":
    unittest.main()
