import unittest

import numpy as np
import pytest

import skrf
import skrf.calibration as skrf_cal
from skrf.media import Coaxial, DefinedGammaZ0, RectangularWaveguide
from skrf.util import suppress_warning_decorator

# TODO: Fix the random seed of the test, so each test generates reproducible
# networks.
# Issue: https://github.com/scikit-rf/scikit-rf/issues/1314

# Number of frequency points to test calibration at.
# Use 100 for speed, but given that most tests employ *random*
# networks values 1000 are better for initial verification
NPTS = 10

# System reference impedance of the calculated S-parameters,
# also known as a medium or network's "port impedance"
Z0_REF = [1, 50, 75, 93, 600]

# Different transmission lines and waveguides to test calibration with.
# This list should cover diverse media with different characteristic
# impedance and reference impedance (but DO NOT repeat each medium for
# every Z0_REF), Z0_REF renormalization is tested separately.
#
# Don't forget to change AbstractIncompleteCalTest.valid_frequency's
# default value (1 MHz to 110 GHz) if you add new tests beyond this
# range.
MEDIUM = [
    # WR10/WG27/R900 75 to 110 GHz, 0.1x0.05 inch (2.54x1.27 mm)
    # lossless (perfect conductor), z0_characteristic vary from
    # 610 to 446 ohm, z0_port to be renormalized.
    RectangularWaveguide(
        skrf.F(75, 100, NPTS, unit='GHz'),
        a=100 * skrf.mil
    ),

    # WR10/WG27/R900 75 to 110 GHz, 0.1x0.05 inch (2.54x1.27 mm)
    # z0 from 610 to 446 ohm, lossy (gold), z0_characteristic vary from
    # 610 to 446 ohm, z0_port to be renormalized.
    RectangularWaveguide(
        skrf.F(75, 100, NPTS, unit='GHz'),
        a=100 * skrf.mil, rho='gold'
    ),

    # 3.5 mm coaxial air line, Dout = 3.5 mm, Din = 1.52 mm, lossy (copper),
    # upper frequency 26.5 GHz, z0_characteristic close to 50 ohm,
    # z0_port to be renormalized.
    Coaxial(
        skrf.F(0.1, 26.5, NPTS, unit='GHz'),
        Dout=3.5e-3, Din=1.52e-3, epsilon_r=1,
        sigma=1 / skrf.data.materials["copper"]["resistivity(ohm*m)"]
    ),

    # RG-59 Cable TV coax, Dout = 3.7 mm, Din = 0.58 mm, lossy (copper),
    # upper frequency 1 GHz, solid PE dielectric, z0_characteristic close
    # to 75 ohm, z0_port to be renormalized.
    Coaxial(
        skrf.F(0.1, 1, NPTS, unit='GHz'),
        Dout=3.7e-3, Din=0.58e-3, epsilon_r=2.3,
        sigma=1 / skrf.data.materials["copper"]["resistivity(ohm*m)"]
    )
]


def lossless_from_lossy(medium):
    """
    Create a lossless medium instance from a lossy medium instance,
    it's necessary for calibration algorithms that assume lossless
    DUTs.
    """
    if isinstance(medium, RectangularWaveguide):
        return medium.mode(rho=None)
    elif isinstance(medium, Coaxial):
        return medium.mode(sigma=1e99)
    else:
        raise NotImplementedError


def compare_dicts_allclose(first: dict, second: dict) -> None:
    """
    Compare whether two dictionaries' keys all have nearly the same
    numerical values.
    """
    assert first.keys() == second.keys()
    for k in first.keys():
        np.testing.assert_allclose(
            first[k], second[k],
            err_msg=f"Values from key '{k}' not equal!"
        )


class UncalibratedNetworkAnalyzer:
    """
    Simulate a standard 2-port, 2-port Vector Network Analyzer (VNA) with
    random error boxes. The Device-Under-Test (DUT) is connected after these
    error boxes. The errors are so large that they make all measurements
    results meaningless. But they're linear, so a suitable calibration
    algorithm should be able to remove them exactly - as a severe stress test.

    Parameters
    ----------
    medium : skrf.media.Media
        Transmission line or waveguide defined in MEDIUM. Its frequency range
        and reference impedance is inherited by the VNA's error boxes.

    nports : int
        Number of ports, only 1 and 2 are implemented.

    switch_err : bool
        In a practical VNA, measurement errors are not fully modeled by two
        2-port error boxes at the VNA's Port 1 and Port 2. Because the signal
        generator in multiplexed by an RF switch, the port matching changes
        slightly depending on whether Port 1 or Port 2 is excited as the switch
        changes its state. This effect can be modeled by two imperfect
        termination impedances cascaded with the two error box.

        As a result, the EightTerm calibration is inexactly, unless the switch
        terms are measured through other means and provided to the algorithm
        as external inputs. The TwelveTerm calibration is exact because the
        forward and backward paths are characterized separately.

    leakage_err : bool
        Crosstalk between Port 1 and Port 2 in the TwelveTerm error model.
        Note that this is only an approximate model for small crosstalks
        that contains 2 terms only. If crosstalk is significant (in the
        case of the SixteenTerm error model), the VNA error boxes must be
        modeled as 4-port networks (which is not implemented here).
    """
    def __init__(self, medium, nports, switch_err=True, leakage_err=True):
        self.medium = medium
        self.nports = nports
        self.switch_err = switch_err
        self.leakage_err = leakage_err

        if nports not in [1, 2]:
            raise NotImplementedError(
                "nports == %d not implemented" % nports
            )
        self._create_error_boxes()

    def _create_error_boxes(self):
        """
        Create error boxes of the VNA. These error boxes include
        a 2-port error box (err_x, err_y) at each VNA port, and
        optionally 1-port switch terms (gamma_f, gamma_f). In addition,
        two simplified 1-port crosstalk terms (iso_f, iso_r) are also
        included.
        """
        self.err = {}

        if self.nports == 1:
            self.err["x"] = self.medium.random(n_ports=2, name='Err_x')
        elif self.nports == 2:
            self.err["x"] = self.medium.random(n_ports=2, name='Err_x')
            self.err["y"] = self.medium.random(n_ports=2, name='Err_y')

            if self.switch_err:
                self.err["gamma_f"] = self.medium.random(n_ports=1, name='Gamma_f')
                self.err["gamma_r"] = self.medium.random(n_ports=1, name='Gamma_r')
            else:
                self.err["gamma_f"] = self.medium.match(n_ports=1, name='Gamma_f')
                self.err["gamma_r"] = self.medium.match(n_ports=1, name='Gamma_r')

            if self.leakage_err:
                self.err["iso_f"] = self.medium.random(n_ports=1, name='Iso_f')
                self.err["iso_r"] = self.medium.random(n_ports=1, name='Iso_r')
            else:
                self.err["iso_f"] = self.medium.match(n_ports=1, name='Iso_f')
                self.err["iso_r"] = self.medium.match(n_ports=1, name='Iso_r')

    def measure(self, dut):
        """
        Return raw measurement results of a ideal DUT with errors.
        """
        dut = dut.copy()

        if self.nports == 1:
            cascade = self.err["x"] ** dut
        elif self.nports == 2:
            # error terms
            cascade = self.err["x"] ** dut ** self.err["y"]

            # switch terms
            cascade = skrf.terminate(
                cascade, self.err["gamma_f"], self.err["gamma_r"]
            )

            # leakage terms
            cascade.s[:,1,0] += self.err["iso_f"].s[:,0,0]
            cascade.s[:,0,1] += self.err["iso_r"].s[:,0,0]

        cascade.name = 'VNA raw'
        return cascade


class AbstractIncompleteCalTest:
    """
    Base class of all calibration tests.

    Most algorithms (e.g. SOLTTest, SDDLTest) should inherit AbstractCalTest
    to check the correctness of the DUT calibration and solved error terms.

    If the tested algorithm is only a sub-step and doesn't provide error
    terms or corrections in itself (e.g. ComputeSwitchTermsTest), use
    AbstractIncompleteCalTest.
    """
    def setUp(self):
        """
        Create a list of dictionaries, each item represents a "test group".
        Each test group is a combination of all MEDIUM normalized to each
        Z0_REF. A group-specific VNA, calibration standards, and DUTs are
        also provided as both definitions and as measurement with errors.
        """
        self.testgroup_list = []

        for medium in MEDIUM:
            over_frequency_range = (
                medium.frequency.f[0] < self.valid_frequency.f[0] or
                medium.frequency.f[-1] > self.valid_frequency.f[-1]
            )

            if over_frequency_range:
                # skip any medium which a frequency range beyond what's
                # supported by the calibration algorithm
                continue

            for z0_ref in Z0_REF:
                if z0_ref is None:
                    group_medium = medium.copy()
                else:
                    group_medium = medium.mode(z0_port=z0_ref)

                vna = self.setup_vna_for_testgroup(group_medium)
                std_defs, std_meas = self.setup_std_for_testgroup(
                    group_medium, vna
                )
                dut_def = group_medium.random(
                    n_ports=self.nports, name='dut_def'
                )
                dut_meas = vna.measure(dut_def)
                dut_meas.name = 'dut_meas'

                self.testgroup_list.append({
                    "vna": vna,
                    "z0_ref": z0_ref,
                    "std_defs": std_defs,
                    "std_meas": std_meas,
                    "cal": self.setup_cal(vna, std_defs, std_meas),
                    "dut_def": dut_def,
                    "dut_meas": dut_meas
                })

    @property
    def nports(self):
        """
        How many ports are involved in the calibration test?

        Must be implemented by the subclass.
        """
        raise NotImplementedError

    @property
    def valid_frequency(self):
        """
        What is the valid frequency range of this calibration?
        By default, use 1 MHz to 110 GHz.

        Some calibration algorithms are bandwidth-limited. If the
        medium's frequency span exceeds this value, it's currently
        skipped for simplicity (in the future, we can implement
        segmented tests to test all mediums within the window of
        validity).
        """
        return skrf.Frequency(1, 110000, 2, unit='MHz')

    def setup_vna_for_testgroup(self, medium):
        """
        Return an UncalibratedNetworkAnalyzer() instance according to the
        medium and z0_ref within a single test group.
        """
        return UncalibratedNetworkAnalyzer(medium, nports=self.nports)

    def setup_std_for_testgroup(self, medium, vna):
        """
        Return two lists that contain calibration standards definitions
        and imperfect measurements by vna, respectively.

        Must be implemented by the subclass.
        """
        raise NotImplementedError

    def setup_cal(self, vna, std_defs, std_meas):
        """
        Return a skrf.calibration.Calibration instance, initialized using
        calibration standard definitions (std_defs) and imperfect measurements
        by the VNA (std_meas).

        Must be implemented by the subclass.
        """
        raise NotImplementedError


class AbstractCalTest(AbstractIncompleteCalTest):
    """
    Base class of most calibration tests.

    Most algorithms (e.g. SOLTTest, SDDLTest) should inherit AbstractCalTest
    to check the correctness of the DUT calibration and solved error terms.

    If the tested algorithm is only a sub-step and doesn't provide error
    terms or corrections in itself (e.g. ComputeSwitchTermsTest), use
    AbstractIncompleteCalTest.
    """
    def test_accuracy_of_dut_correction(self):
        """
        Calibrate the random DUT network created for each test group,
        and compare it with its original definition. The calibration
        is successful if both networks match.

        All subclasses inherit this feature automatically.
        """
        for testgroup in self.testgroup_list:
            dut_corrected = testgroup["cal"].apply_cal(
                testgroup["dut_meas"]
            )
            dut_corrected.name = 'dut_corrected'
            self.assertEqual(testgroup["dut_def"], dut_corrected)

    def test_embed_equal_measure(self):
        """
        The skrf.calibration.Calibration class has an embed() method, which
        simulates a measurement with error according to the solved error boxes
        by the calibration algorithm. The result should be equal to measuring
        the same ideal network directly by the uncalibrated VNA.
        """
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            vna = testgroup["vna"]
            dut_def = testgroup["dut_def"]
            dut_meas = testgroup["dut_meas"]
            self.assertEqual(cal.embed(dut_def), dut_meas)
            self.assertEqual(cal.embed(dut_def), vna.measure(dut_def))

    def test_embed_then_apply_cal(self):
        """
        The skrf.calibration.Calibration class has an embed() method, which
        simulates a measurement with error according to the solved error boxes
        by the calibration algorithm. If we inject error via embed() to a
        network's definition, and use the same calibration algorithm to
        calibrate the network with errors, we should obtain the original
        network.
        """
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            vna = testgroup["vna"]
            dut_def = testgroup["dut_def"]
            dut_meas = testgroup["dut_meas"]
            self.assertEqual(cal.apply_cal(cal.embed(dut_def)), dut_def)

    @suppress_warning_decorator("n_thrus is None")
    def test_from_coefs(self):
        """
        If the calibration instance is recreated from its error coefficients,
        the calibration should remain correct.
        """
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            dut_def = testgroup["dut_def"]
            dut_meas = testgroup["dut_meas"]
            #print(cal.coefs)
            cal_from_coeffs = cal.from_coefs(cal.frequency, cal.coefs)
            self.assertEqual(
                cal_from_coeffs.apply_cal(cal.embed(dut_def)), dut_def
            )
            self.assertEqual(cal_from_coeffs.apply_cal(dut_meas), dut_def)

    @suppress_warning_decorator("n_thrus is None")
    def test_from_coefs_ntwks(self):
        """
        If the calibration instance is recreated from its error networks,
        the calibration should remain correct.
        """
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            dut_def = testgroup["dut_def"]
            dut_meas = testgroup["dut_meas"]
            cal_from_coeffs = cal.from_coefs_ntwks(cal.coefs_ntwks)
            self.assertEqual(
                cal_from_coeffs.apply_cal(cal.embed(dut_def)), dut_def
            )
            self.assertEqual(cal_from_coeffs.apply_cal(dut_meas), dut_def)

    # Test if the calibration instance can return a meaningful @property
    # without returning None or throwing exceptions, including error_ntwk,
    # coefs_ntwks, caled_ntwks, residual_ntwks
    def test_error_ntwk_property(self):
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            self.assertIsNotNone(cal.error_ntwk)

    def test_coefs_ntwks_property(self):
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            self.assertIsNotNone(cal.coefs_ntwks)

    @suppress_warning_decorator("only gave a single measurement orientation")
    def test_caled_ntwks_property(self):
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            self.assertIsNotNone(cal.caled_ntwks)

    @suppress_warning_decorator("only gave a single measurement orientation")
    def test_residual_ntwks_property(self):
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            self.assertIsNotNone(cal.residual_ntwks)


class ComputeSwitchTermsTest(AbstractIncompleteCalTest, unittest.TestCase):
    """
    Test skrf.calibration.compute_switch_terms(), the indirect method
    of computing the switch terms with at least three reciprocal devices
    """
    @property
    def nports(self):
        return 2

    def setup_vna_for_testgroup(self, medium, z0_ref=None):
        # this algorithm assumes no crosstalks
        return UncalibratedNetworkAnalyzer(
            medium, nports=self.nports, leakage_err=False
        )

    def setup_std_for_testgroup(self, medium, vna):
        # reciprocal devices: asymmetric and both transmissive and reflective
        # devices
        r_list = [25, 50, 100]
        std_defs = [
            medium.resistor(r) ** medium.shunt_resistor(r) for r in r_list
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        # Not applicable, this is not a full VNA calibration test
        # but only a partial calibration that determines the switch
        # terms.
        return None

    def test_accuracy_of_solved_switch_terms(self):
        for testgroup in self.testgroup_list:
            continue  # disable
            vna = testgroup["vna"]
            std_meas = testgroup["std_meas"]

            # solve switch error terms.
            gamma_f, gamma_r = skrf_cal.compute_switch_terms(std_meas)

            # compare them with the actual VNA switch error terms
            print(gamma_f, vna.err["gamma_f"])
            print(gamma_r, vna.err["gamma_r"])
            self.assertTrue(
                gamma_f == vna.err["gamma_f"]
                #all(np.abs(gamma_f.s - vna.err["gamma_f"].s) < 1e-9)
            )
            self.assertTrue(
                gamma_r == vna.err["gamma_r"]
                #all(np.abs(gamma_r.s - vna.err["gamma_r"].s) < 1e-9)
            )


class AbstractOnePortTest(AbstractCalTest):
    """
    Base class of all one-port calibration tests.
    """
    @property
    def nports(self):
        return 1

    def test_error_coeffs_accuracy(self):
        test_list = [
            {
                "name": "directivity",
                "ref_func": lambda vna: vna.err["x"].s11
            },
            {
                "name": "source match",
                "ref_func": lambda vna: vna.err["x"].s22
            },
            {
                "name": "reflection tracking",
                "ref_func": lambda vna: vna.err["x"].s21 * vna.err["x"].s12
            }
        ]

        for test in test_list:
            with self.subTest(i=test["name"]):
                for testgroup in self.testgroup_list:
                    coeff_name = test["name"]
                    vna = testgroup["vna"]
                    cal = testgroup["cal"]

                    self.assertEqual(
                        test["ref_func"](vna),
                        cal.coefs_ntwks[coeff_name]
                    )

    def test_input_networks_1port(self):
        """
        Test that users do not enter 2-port networks by accident
        """
        medium = DefinedGammaZ0()
        vna = self.setup_vna_for_testgroup(medium)
        std_defs = [
            skrf.two_port_reflect(
                medium.short(name='short'), medium.short(name='short')
            ),
            medium.delay_short(45, 'deg', name='ew'),
            medium.delay_short(45, 'deg', name='qw'),
            medium.match(name='load')
        ]
        with self.assertRaises(RuntimeError):
            cal = self.setup_cal(vna, std_defs, std_defs)


class OnePortTest(AbstractOnePortTest, unittest.TestCase):
    """
    Test skrf.calibration.OnePort.
    """
    def setup_std_for_testgroup(self, medium, vna):
        std_defs = [
            medium.short(name='short'),
            medium.delay_short(45, unit='deg', name='ew'),
            medium.delay_short(90, unit='deg', name='qw'),
            medium.match(name='load'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.OnePort(
            is_reciprocal=True,
            ideals=std_defs,
            measured=std_meas
        )
        cal.run()
        return cal

    def test_input_networks_inconsistent_frequency_impedance(self):
        """
        Test that calibration inputs with different impedances per
        frequency generates a warning. This is not supported in scikit-rf
        as it's untested. Nevertheless, the current design already tracks
        reference impedance per frequency, so in theory it can be implemented
        in the future if there's a need.

        We run the test in OnePortTest instead of AbstractOnePortTest because
        different one-port calibration algorithm requires different inputs,
        but this check is found within the calibration base class, so we only
        need to test it once.
        """
        medium = Coaxial(
            skrf.F(0.1, 26.5, 10, unit='GHz')
        )
        vna = self.setup_vna_for_testgroup(medium)
        std_defs = [
            medium.open(),
            medium.short(),
            medium.match()
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]

        for std in std_defs + std_meas:
            std.renormalize(np.array(
                [50, 75, 50, 75, 50, 75, 50, 75, 50, 75]
            ))

        with self.assertWarns(UserWarning):
            cal = self.setup_cal(vna, std_defs, std_defs)


class SDDLTest(AbstractOnePortTest, unittest.TestCase):
    """
    Test skrf.calibration.SDDL.
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna):
        # SDDL assumes the offset shorts are lossless
        lossless_medium = lossless_from_lossy(medium)

        std_defs_incomplete_knowledge = [
            medium.short(name='short'),
            lossless_medium.delay_short(45, unit='deg', name='ew'),
            lossless_medium.delay_short(90, unit='deg', name='qw'),
            medium.load(.2+.2j, name='load'),
        ]
        std_defs = [
            medium.short(name='short'),
            lossless_medium.delay_short(10, unit='deg', name='ew'),
            lossless_medium.delay_short(33, unit='deg', name='qw'),
            medium.load(.2+.2j, name='load'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.SDDL(
            is_reciprocal=True,
            ideals=std_defs,
            measured=std_meas
        )
        cal.run()
        return cal

    @pytest.mark.skip(reason='not applicable')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable')
    def test_from_coefs_ntwks(self):
        pass


class SDDLNoneTest(SDDLTest):
    """
    Test skrf.calibration.SDDL with None as the definition of
    two delayed short termination.
    """
    def setup_std_for_testgroup(self, medium, vna):
        std_defs_incomplete_knowledge, std_meas = (
            super().setup_std_for_testgroup(medium, vna)
        )
        std_defs_incomplete_knowledge[1] = None
        std_defs_incomplete_knowledge[2] = None
        return std_defs_incomplete_knowledge, std_meas


class SDDLWeikleTest(AbstractOnePortTest, unittest.TestCase):
    """
    Test skrf.calibration.SDDLWeikle.
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna):
        # SDDL assumes the offset shorts are lossless
        lossless_medium = lossless_from_lossy(medium)

        std_defs_incomplete_knowledge = [
            medium.short(name='short'),
            lossless_medium.delay_short(45., 'deg', name='ew'),
            lossless_medium.delay_short(90., 'deg', name='qw'),
            medium.load(.2+.2j, name='load'),
        ]
        std_defs = [
            medium.short(name='short'),
            lossless_medium.delay_short(10, 'deg', name='ew'),
            lossless_medium.delay_short(80, 'deg', name='qw'),
            medium.load(.2+.2j, name='load'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.SDDLWeikle(
            is_reciprocal=True,
            ideals=std_defs,
            measured=std_meas
        )
        cal.run()
        return cal

    @pytest.mark.skip(reason='not applicable')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable')
    def test_from_coefs_ntwks(self):
        pass


class SDDMTest(AbstractOnePortTest, unittest.TestCase):
    """
    Test skrf.calibration.SDDM.

    This is a specific test of SDDL to verify it works when the load is
    a matched load. This test has been used to show that the SDDLWeikle
    variant fails, with a perfect matched load.
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna):
        # SDDL assumes the offset shorts are lossless
        lossless_medium = lossless_from_lossy(medium)

        std_defs_incomplete_knowledge = [
            medium.short(name='short'),
            lossless_medium.delay_short(45, 'deg', name='ew'),
            lossless_medium.delay_short(90, 'deg', name='qw'),
            medium.match(name='load'),
        ]
        std_defs = [
            medium.short(name='short'),
            lossless_medium.delay_short(10, 'deg', name='ew'),
            lossless_medium.delay_short(80, 'deg', name='qw'),
            medium.match(name='load'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.SDDL(
            is_reciprocal=True,
            ideals=std_defs,
            measured=std_meas
        )
        cal.run()
        return cal

    @pytest.mark.skip(reason='not applicable')
    def test_from_coefs(self):
        pass

    @pytest.mark.skip(reason='not applicable')
    def test_from_coefs_ntwks(self):
        pass


#@pytest.mark.skip()
#class PHNTest(AbstractOnePortTest, unittest.TestCase):
#    """
#    Test skrf.calibration.PHN.
#
#    This test is currently disabled because of a square root ambiguity,
#    which means calibration often fails with arbitrary impedance standards
#    in wideband measurements. This is an inherent limitation of the algorithm.
#
#    TODO: Find a suitable standard definition or frequency range to test
#    that the algorithm indeed works within suitable bands and standards.
#    """
#    def setup_std_for_testgroup(self, medium, vna):
#        known1 = medium.random()
#        known2 = medium.random()
#
#        std_defs_incomplete_knowledge = [
#            medium.delay_short(45, 'deg', name='ideal ew'),
#            medium.delay_short(90, 'deg', name='ideal qw'),
#            known1,
#            known2,
#        ]
#        std_defs = [
#            medium.delay_short(33, 'deg', name='true ew'),
#            medium.delay_short(110, 'deg', name='true qw'),
#            known1,
#            known2,
#        ]
#        std_meas = [
#            vna.measure(dut) for dut in std_defs
#        ]
#        return std_defs_incomplete_knowledge, std_meas
#
#    def setup_cal(self, vna, std_defs, std_meas):
#        cal = skrf_cal.PHN(
#            is_reciprocal=True,
#            ideals=std_defs,
#            measured=std_meas
#        )
#        cal.run()
#        return cal
#
#    def test_accuracy_of_solved_half_known_network(self):
#        # TODO: implement this function
#        pass
#
#    @pytest.mark.skip(reason='not applicable')
#    def test_from_coefs(self):
#        pass
#
#    @pytest.mark.skip(reason='not applicable')
#    def test_from_coefs_ntwks(self):
#        pass


class AbstractTwoPortTest(AbstractCalTest):
    """
    Base class of all two-port calibration tests.
    """
    @property
    def nports(self):
        return 2

    def test_error_coeffs_accuracy(self):
        test_list = [
            {
                "name": "forward directivity",
                "ref_func": lambda vna: vna.err["x"].s11
            },
            {
                "name": "forward source match",
                "ref_func": lambda vna: vna.err["x"].s22
            },
            {
                "name": "forward reflection tracking",
                "ref_func": lambda vna: vna.err["x"].s21 * vna.err["x"].s12
            },
            {
                "name": "reverse source match",
                "ref_func": lambda vna: vna.err["y"].s11
            },
            {
                "name": "reverse directivity",
                "ref_func": lambda vna: vna.err["y"].s22
            },
            {
                "name": "reverse reflection tracking",
                "ref_func": lambda vna: vna.err["y"].s21 * vna.err["y"].s12,
            },
            {
                "name": "k",
                "ref_func": lambda vna: vna.err["x"].s21 / vna.err["y"].s12
            },
            {
                "name": "forward isolation",
                "ref_func": lambda vna: vna.err["iso_f"].s11
            },
            {
                "name": "reverse isolation",
                "ref_func": lambda vna: vna.err["iso_r"].s11
            }
        ]
        for test in test_list:
            with self.subTest(i=test["name"]):
                for testgroup in self.testgroup_list:
                    coeff_name = test["name"]
                    vna = testgroup["vna"]
                    cal = testgroup["cal"]

                    self.assertEqual(
                        test["ref_func"](vna),
                        cal.coefs_ntwks[coeff_name]
                    )

    def test_verify_12term(self):
        for testgroup in self.testgroup_list:
            self.assertTrue(
                testgroup["cal"].verify_12term_ntwk.s_mag.max() < 1e-3,
            )

    def test_renormalization(self):
        # TODO: Implement this
        pass

    def test_renormalization_loopback(self):
        # TODO: Implement this
        pass

    def test_untermination(self):
        # TODO: Implement this
        pass

    def test_input_not_modified(self):
        # TODO: Implement this
        pass

    def test_convert_8term_2_12term_subtests(self):
        # TODO: Implement this
        pass


class EightTermTest(AbstractTwoPortTest, unittest.TestCase):
    """
    Test skrf.calibration.EightTermTest
    """
    def setup_std_for_testgroup(self, medium, vna):
        std_defs = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='load'),
            medium.thru(name='thru'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.EightTerm(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[2],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def test_coeffs_8term(self):
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            self.assertEqual(cal.coefs_8term, cal.coefs)

    def test_coeffs_12term(self):
        for testgroup in self.testgroup_list:
            cal = testgroup["cal"]
            compare_dicts_allclose(
                cal.coefs_12term,
                skrf.convert_8term_2_12term(cal.coefs)
            )

    def test_input_networks_inconsistent_port_impedance(self):
        """
        Test that calibration inputs with different impedances per
        port generates a warning. This is not supported in scikit-rf,
        as all calibration algorithms assume a constant system impedance.
        This feature will be extremely difficult, if ever possible to
        support, a major redesign would be necessary.

        We run the test in EightTermTest instead of AbstractTwoPortTest
        because different two-port calibration algorithm requires different
        inputs, but this check is found within the calibration base class,
        so we only need to test it once.
        """
        frequency = skrf.F(0.1, 1, 6, unit='GHz')
        medium = DefinedGammaZ0(frequency)

        vna = self.setup_vna_for_testgroup(medium)
        std_defs = [
            medium.open(nports=2),
            medium.short(nports=2),
            medium.match(nports=2),
            medium.thru()
        ]
        for std in std_defs:
            std.renormalize(np.array([50, 75]))

        with self.assertWarns(UserWarning):
            cal = self.setup_cal(vna, std_defs, std_defs)


class UnknownThruTest(EightTermTest):
    """
    Test skrf.calibration.UnknownThru
    """
    def setup_std_for_testgroup(self, medium, vna):
        std_defs_incomplete_knowledge = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='match'),
            medium.thru(name='thru'),
        ]
        std_defs = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='match'),
            (
                medium.impedance_mismatch(50, 45) **
                medium.line(20, 'deg', name='line') **
                medium.impedance_mismatch(45, 50)
            )
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.UnknownThru(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[2],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal


class MRCTest(EightTermTest):
    """
    Test skrf.calibration.MRCTest
    """
    @property
    def valid_frequency(self):
        # MRC uses SDDL + UnknownThru, the former is bandwidth-limited.
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna):
        # MRC assumes the offset shorts are lossless
        lossless_medium = lossless_from_lossy(medium)

        std_defs_incomplete_knowledge = [
            medium.short(nports=2, name='short'),
            self._two_ports_delay_shorts(lossless_medium, 45, 90),
            self._two_ports_delay_shorts(lossless_medium, 90, 45),
            medium.load(.2+.2j, nports=2, name='match'),
            medium.thru(name='thru'),
        ]
        std_defs = [
            medium.short(nports=2, name='short'),
            self._two_ports_delay_shorts(lossless_medium, 65, 130),
            self._two_ports_delay_shorts(lossless_medium, 120, 75),
            medium.load(.2+.2j, nports=2, name='match'),
            (
                lossless_medium.impedance_mismatch(50, 45) **
                lossless_medium.line(20, 'deg', name='line') **
                lossless_medium.impedance_mismatch(45, 50)
            )
        ]

        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs_incomplete_knowledge, std_meas

    def _two_ports_delay_shorts(self, medium, deg1, deg2):
        ds1 = medium.delay_short(deg1, 'deg')
        ds2 = medium.delay_short(deg2, 'deg')
        return skrf.two_port_reflect(ds1, ds2)

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.MRC(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[3],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    @pytest.mark.skip(reason='not applicable')
    def test_input_networks_inconsistent_port_impedance(self):
        pass


class TwelveTermTest(AbstractTwoPortTest, unittest.TestCase):
    """
    Test skrf.calibration.TwelveTermTest
    """
    def setup_std_for_testgroup(self, medium, vna):
        std_defs = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='load'),
            medium.random(2, name='rand1'),
            medium.random(2, name='rand2'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.TwelveTerm(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[2],
            n_thrus=2
        )
        cal.run()
        return cal

    def test_coeffs_8term(self):
        # TODO
        pass

    def test_coeffs_12term(self):
        # TODO
        pass


class SOLTTest(TwelveTermTest):
    """
    Test skrf.calibration.SOLT
    """
    def setup_std_for_testgroup(self, medium, vna):
        std_defs_incomplete_knowledge = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='load'),
            None,  # test the auto detection of Thru network
        ]
        std_defs = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='load'),
            medium.thru()
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.SOLT(
            ideals=std_defs,
            measured=std_meas,
            n_thrus=1,
            isolation=std_meas[2],
        )
        cal.run()
        return cal


if __name__ == "__main__":
    unittest.main()
