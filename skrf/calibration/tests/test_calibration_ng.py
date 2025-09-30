import unittest

import numpy as np
import pytest

import skrf
import skrf.calibration as skrf_cal
from skrf.media import (
    Coaxial, DefinedGammaZ0, DistributedCircuit, RectangularWaveguide
)
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
Z0_REF = [10, 50, 75, 93, 600]
#Z0_REF = [50]
#Z0_REF = [75]
#Z0_REF = [None]
# TODO: LRMTest fails if Z0_REF is less or equal to 2 ohms.
# Numerical stability issues? I see the schoolbook quadratic
# root formula, perhaps that's the issue?
#
# TODO: LRRMTest fails if Z0_REF is 3 ohms (other values are
# untested), same problem?

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
    #RectangularWaveguide(
    #    skrf.F(75, 100, NPTS, unit='GHz'),
    #    a=100 * skrf.mil,
    #),

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
    #Coaxial(
    #    skrf.F(0.1, 26.5, NPTS, unit='GHz'),
    #    Dout=3.5e-3, Din=1.52e-3, epsilon_r=1,
    #    sigma=1 / skrf.data.materials["copper"]["resistivity(ohm*m)"]
    #),

    ## RG-59 Cable TV coax, Dout = 3.7 mm, Din = 0.58 mm, lossy (copper),
    ## upper frequency 1 GHz, solid PE dielectric, z0_characteristic close
    ## to 75 ohm, z0_port to be renormalized.
    #Coaxial(
    #    skrf.F(0.1, 1, NPTS, unit='GHz'),
    #    Dout=3.7e-3, Din=0.58e-3, epsilon_r=2.3,
    #    sigma=1 / skrf.data.materials["copper"]["resistivity(ohm*m)"]
    #)
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

    transmissive_err : bool
        Modify the random error boxes with much higher transmissive
        coefficients (S21, S12) in comparison to reflection coefficients
        (S11, S22). This is needed for TRL calibration algorithms to
        guess the lengths of line standards.
    """
    def __init__(
        self, medium, nports, switch_err=True, leakage_err=True,
        transmissive_err=False
    ):
        self.medium = medium
        self.nports = nports
        self.switch_err = switch_err
        self.leakage_err = leakage_err
        self.transmissive_err = transmissive_err

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

            if self.transmissive_err:
                self.err["x"].s[:,0,0] *= 1e-1
                self.err["y"].s[:,0,0] *= 1e-1
                self.err["x"].s[:,1,1] *= 1e-1
                self.err["y"].s[:,1,1] *= 1e-1

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

                testgroup = {}

                vna = self.setup_vna_for_testgroup(group_medium)
                std_fulldefs, std_defs, std_meas = self.setup_std_for_testgroup(
                    group_medium, vna, testgroup
                )
                dut_def = group_medium.random(
                    n_ports=self.nports, name='dut_def'
                )
                cal = self.setup_cal(vna, std_defs, std_meas)

                vna_portext = self.setup_vna_port_extension(group_medium, vna)
                dut_meas = vna_portext.measure(dut_def)
                dut_meas.name = 'dut_meas'

                testgroup["vna"] = vna_portext
                testgroup["medium"] = group_medium
                testgroup["z0_ref"] = z0_ref
                testgroup["std_fulldefs"] = std_fulldefs
                testgroup["std_defs"] = std_defs
                testgroup["std_meas"] = std_meas
                testgroup["cal"] = cal
                testgroup["dut_def"] = dut_def
                testgroup["dut_meas"] = dut_meas

                self.testgroup_list.append(testgroup)

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

    def setup_std_for_testgroup(self, medium, vna, testgroup):
        """
        Return three lists that contain the perfect calibration standards
        definitions, partial calibration standards definitions due to
        incomplete knowledge, and imperfect measurements by vna, respectively.

        For most calibration tests, exact definitions are used, so the first
        and second list elements are the same.

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

    def setup_vna_port_extension(self, medium, vna):
        """
        In TRL calibration, its the reference plane is located at the
        center of the Thru standard, not at the VNA port, so all tests
        fail due to reference plane mismatched. Before the TRL calibration's
        result is checked, we need to extend the port of the virtual VNA
        by inserting transmission line segments.

        It's not needed for non-TRL calibration classes or TRL calibration
        that places the reference plane at the ports (e.g. multi-line TRL).
        """
        return vna


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

    def setup_vna_for_testgroup(self, medium):
        # this algorithm assumes no crosstalks
        return UncalibratedNetworkAnalyzer(
            medium, nports=self.nports, leakage_err=False
        )

    def setup_std_for_testgroup(self, medium, vna, testgroup):
        # reciprocal devices: asymmetric and both transmissive and reflective
        # devices
        r_list = [25, 50, 100]
        std_defs = [
            medium.resistor(r) ** medium.shunt_resistor(r) for r in r_list
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        # Not applicable, this is not a full VNA calibration test
        # but only a partial calibration that determines the switch
        # terms.
        return None

    def test_accuracy_of_solved_switch_terms(self):
        for testgroup in self.testgroup_list:
            vna = testgroup["vna"]
            std_meas = testgroup["std_meas"]

            # solve switch error terms
            gamma_f, gamma_r = skrf_cal.compute_switch_terms(std_meas)

            # compare them with the actual VNA switch error terms
            self.assertTrue(gamma_f == vna.err["gamma_f"])
            self.assertTrue(gamma_r == vna.err["gamma_r"])


class InconsistentFreqImpedanceTest(AbstractIncompleteCalTest, unittest.TestCase):
    """
    Test that calibration inputs with different impedances per
    frequency generates a warning. This is not supported in scikit-rf
    as it's untested. Nevertheless, the current design already tracks
    reference impedance per frequency, so in theory it can be implemented
    in the future if there's a need.

    We run the test in OnePort instead of AbstractOnePortTest because
    different one-port calibration algorithm requires different inputs,
    but this check is found within the calibration base class, so we only
    need to test it once.
    """
    def setUp(self):
        pass

    @property
    def nports(self):
        return 1

    def test_input_networks_inconsistent_frequency_impedance(self):
        medium = Coaxial(
            skrf.F(0.1, 26.5, 10, unit='GHz')
        )
        vna = self.setup_vna_for_testgroup(medium)
        std_defs = [
            medium.open(),
            medium.short(),
            medium.match()
        ]

        for std in std_defs:
            std.renormalize(np.array(
                [50, 75, 50, 75, 50, 75, 50, 75, 50, 75]
            ))

        # no need to run calibration, we only check the warning message here
        with self.assertWarns(UserWarning):
            cal = skrf_cal.OnePort(
                is_reciprocal=True,
                ideals=std_defs,
                measured=std_defs  # no need to actually measure them
            )


class InconsistentPortImpedanceTest(AbstractIncompleteCalTest, unittest.TestCase):
    """
    Test that calibration inputs with different impedances per
    port generates a warning. This is not supported in scikit-rf,
    as all calibration algorithms assume a constant system impedance.
    This feature will be extremely difficult, if ever possible to
    support, a major redesign would be necessary.

    We run the test in EightTerm only because different two-port calibration
    algorithm requires different inputs, but this check is found within the
    calibration base class, so we only need to test it once.
    """
    def setUp(self):
        pass

    @property
    def nports(self):
        return 2

    def test_input_networks_inconsistent_port_impedance(self):
        medium = Coaxial(
            skrf.F(0.1, 26.5, 10, unit='GHz')
        )
        vna = self.setup_vna_for_testgroup(medium)
        std_defs = [
            medium.open(nports=2),
            medium.short(nports=2),
            medium.match(nports=2),
            medium.thru()
        ]

        for std in std_defs:
            std.renormalize(np.array([50, 75]))

        # no need to run calibration, we only check the warning message here
        with self.assertWarns(UserWarning):
            cal = skrf_cal.EightTerm(
                ideals=std_defs,
                measured=std_defs,  # no need to actually measure them
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
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        std_defs = [
            medium.short(name='short'),
            medium.delay_short(45, unit='deg', name='ew'),
            medium.delay_short(90, unit='deg', name='qw'),
            medium.match(name='load'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.OnePort(
            is_reciprocal=True,
            ideals=std_defs,
            measured=std_meas
        )
        cal.run()
        return cal


class SDDLTest(AbstractOnePortTest, unittest.TestCase):
    """
    Test skrf.calibration.SDDL.
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs_incomplete_knowledge, std_meas

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
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        std_defs, std_defs_incomplete_knowledge, std_meas = (
            super().setup_std_for_testgroup(medium, vna, testgroup)
        )
        std_defs_incomplete_knowledge[1] = None
        std_defs_incomplete_knowledge[2] = None
        return std_defs, std_defs_incomplete_knowledge, std_meas


class SDDLWeikleTest(AbstractOnePortTest, unittest.TestCase):
    """
    Test skrf.calibration.SDDLWeikle.
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs_incomplete_knowledge, std_meas

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

    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs_incomplete_knowledge, std_meas

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
#    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        std_defs = [
            medium.short(nports=2, name='short'),
            medium.open(nports=2, name='open'),
            medium.match(nports=2, name='load'),
            medium.thru(name='thru'),
        ]
        std_meas = [
            vna.measure(dut) for dut in std_defs
        ]
        return std_defs, std_defs, std_meas

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


class UnknownThruTest(EightTermTest):
    """
    Test skrf.calibration.UnknownThru
    """
    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs_incomplete_knowledge, std_meas

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
    Test skrf.calibration.MRC
    """
    @property
    def valid_frequency(self):
        # MRC uses SDDL + UnknownThru, the former is bandwidth-limited.
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs_incomplete_knowledge, std_meas

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


class TRLTest(EightTermTest):
    """
    Test skrf.calibration.TRL
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_vna_for_testgroup(self, medium):
        # The error networks must have higher transmission than reflection
        # in order for TRL() to automatically detremine which standard is
        # which
        return UncalibratedNetworkAnalyzer(
            medium, nports=self.nports, transmissive_err=True
        )

    def setup_std_for_testgroup(self, medium, vna, testgroup):
        # This is the dark side of z0_port in medium - when it's set,
        # it creates ideal calibration standards with respect to the
        # characteristic impedance of the medium, and renormalizing the
        # result again to a new reference impedance. But in TRL calibration,
        # if we're calibrating to the system impedance, we must create
        # perfect standards with respect to the system impedance without
        # renormalization.
        matched_medium = medium.mode(z0_override=medium.z0_port, z0_port=None)

        std_defs_incomplete_knowledge = [
            matched_medium.thru(name='thru'),
            matched_medium.short(nports=2, name='short'),
            matched_medium.line(90, 'deg', name='line'),
        ]
        std_defs = [
            matched_medium.thru(name='thru'),
            skrf.two_port_reflect(
                matched_medium.load(-.9-.1j),
                matched_medium.load(-.9-.1j)
            ),
            matched_medium.attenuator(-3, True, 45, 'deg')
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.TRL(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[1],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def test_found_reflect(self):
        """
        Can TRL() determine which measurement is the Reflect standards?"
        """
        for testgroup in self.testgroup_list:
            self.assertEqual(
                testgroup["cal"].ideals[1],
                testgroup["std_fulldefs"][1]
            )

    def test_found_line(self):
        """
        Can TRL() determine which measurement is the Line standards?"
        """
        for testgroup in self.testgroup_list:
            self.assertEqual(
                testgroup["cal"].ideals[2],
                testgroup["std_fulldefs"][2]
            )


class TRLUnknownDefinitionTest(TRLTest):
    """
    Test skrf.calibration.TRL
    """
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        matched_medium = medium.mode(z0_override=medium.z0_port, z0_port=None)

        # Don't define the calibration standard at all, not even
        # approximate knowledge. TRL should be able to solve it.
        std_defs_incomplete_knowledge = []
        std_defs = [
            matched_medium.thru(name='thru'),
            skrf.two_port_reflect(
                matched_medium.delay_short(20, 'deg'),
                matched_medium.delay_short(20, 'deg')
            ),
            matched_medium.attenuator(-3, True, 45, 'deg')
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.TRL(
            ideals=None,
            measured=std_meas,
            isolation=std_meas[1],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal


class TRLLongThruTest(EightTermTest):
    """
    Test skrf.calibration.TRL
    """
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        matched_medium = medium.mode(z0_override=medium.z0_port, z0_port=None)

        # reflect reference plane is at the Thru center
        reflect = matched_medium.load(-.9-.1j)
        reflect_shifted = matched_medium.line(50, 'um') ** reflect

        std_defs_incomplete_knowledge = [
            matched_medium.line(100, 'um', name='thru'),
            matched_medium.short(nports=2, name='short'),
            matched_medium.line(1100, 'um', name='thru')
        ]
        std_defs = [
            matched_medium.line(100, 'um', name='thru'),
            skrf.two_port_reflect(
                reflect_shifted,
                reflect_shifted
            ),
            matched_medium.line(1100, 'um', name='thru')
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.TRL(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[1],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def setup_vna_port_extension(self, medium, vna):
        matched_medium = medium.mode(z0_override=medium.z0_port, z0_port=None)

        # Calibration is done at the center of the thru, add half thru
        # to error networks so that tests pass.
        vna.err["x"] = vna.err["x"] ** matched_medium.line(50, 'um')
        vna.err["y"] = matched_medium.line(50, 'um') ** vna.err["y"]
        return vna

    def test_found_reflect(self):
        # solved Reflect is at the Thru center
        for testgroup in self.testgroup_list:
            medium = testgroup["medium"]
            matched_medium = (
                medium.mode(
                    z0_override=medium.z0_port, z0_port=None
                )
            )
            reflect = matched_medium.load(-.9-.1j)

            self.assertEqual(
                testgroup["cal"].ideals[1],
                skrf.two_port_reflect(reflect, reflect)
            )

    def test_found_line(self):
        # solved Line is difference between Line and Thru
        for testgroup in self.testgroup_list:
            self.assertEqual(
                testgroup["cal"].ideals[2],
                testgroup["std_defs"][2] ** testgroup["std_defs"][0].inv
            )


class TRLMultiline(EightTermTest):
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        matched_medium = medium.mode(z0_override=medium.z0_port, z0_port=None)

        std_defs = [
            matched_medium.thru(name='thru'),
            matched_medium.short(nports=2, name='short'),
            matched_medium.open(nports=2, name='open'),
            matched_medium.attenuator(-3, True, 45, 'deg'),
            matched_medium.attenuator(-6, True, 90, 'deg'),
            matched_medium.attenuator(-8, True, 145, 'deg'),
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.TRL(
            ideals=[None, -1, 1, None, None, None],
            measured=std_meas,
            n_reflects=2,
            isolation=std_meas[1],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def test_found_line(self):
        for testgroup in self.testgroup_list:
            for k in range(2, 5):
                self.assertTrue(
                    testgroup["cal"].ideals[k],
                    testgroup["std_fulldefs"][k]
                )


class NISTMultilineTRLTest(EightTermTest):
    """
    Test skrf.calibration.NISTMultilineTRL
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna, testgroup):
        matched_medium = medium.mode(z0_override=medium.z0_port, z0_port=None)

        std_defs = [
            matched_medium.thru(),
            skrf.two_port_reflect(
                matched_medium.load(-.98-.1j), matched_medium.load(-.98-.1j)
            ),
            skrf.two_port_reflect(
                matched_medium.load(.99+0.05j), matched_medium.load(.99+0.05j)
            ),
            matched_medium.line(100, 'um'),
            matched_medium.line(200, 'um'),
            matched_medium.line(900, 'um')
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.NISTMultilineTRL(
            measured=std_meas,
            isolation=std_meas[1],
            Grefls=[-1, 1],
            l=[0, 100e-6, 200e-6, 900e-6],
            er_est=1,
            gamma_root_choice='real',
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def test_solved_line_constant(self):
        for testgroup in self.testgroup_list:
            solved_gamma = testgroup["medium"].gamma
            actual_gamma = testgroup["cal"].gamma
            self.assertTrue(max(solved_gamma - actual_gamma) < 1e-3)


class NISTMultilineTRLTestRefPlaneShift(EightTermTest):
    """
    Test skrf.calibration.NISTMultilineTRL
    """
    @property
    def valid_frequency(self):
        return skrf.Frequency(75, 100, 2, unit='GHz')

    def setup_std_for_testgroup(self, medium, vna, testgroup):
        rng = np.random.default_rng()

        # This is a special test using one medium only. Ignore the
        # global medium passed to us.
        self.c = 1e-12 * rng.uniform(100, 200, NPTS)
        rlgc = DistributedCircuit(
            frequency=medium.frequency,
            R=rng.uniform(10, 100, NPTS),
            L=1e-9 * rng.uniform(100, 200, NPTS),
            G=np.zeros(NPTS),
            C=self.c
        )
        testgroup["rlgc_medium"] = rlgc

        std_defs = [
            rlgc.line(1000, 'um'),
            (
                rlgc.line(500, 'um') **
                rlgc.short(nports=2) **
                rlgc.line(500, 'um')
            ),
            rlgc.line(1010, 'um'),
            rlgc.line(1100, 'um'),
            rlgc.line(1800, 'um'),
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        # Danger here! In theory we're not allowed to pass or access
        # information via class-wide member variables using self.c,
        # because a single instance is used for many different test
        # groups. But our framework doesn't provide any way to insert
        # auxiliary information into a test group. This hack only works
        # here because setup_cal() is executed immediately after
        # setup_std_for_testgroup() for every test group, so self.c
        # happens to be the unchanged at this moment.
        cal = skrf_cal.NISTMultilineTRL(
            measured=std_meas,
            isolation=std_meas[1],
            Grefls=[-1],
            refl_offset=500e-6,
            l=[1000e-6, 1010e-6, 1100e-6, 1800e-6],
            c0=self.c,
            z0_ref=std_meas[0].z0[0,0],
            gamma_root_choice='real',
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def test_solved_gamma(self):
        for testgroup in self.testgroup_list:
            medium_gamma = testgroup["rlgc_medium"].gamma
            solved_gamma = testgroup["cal"].gamma
            self.assertTrue(max(np.abs(medium_gamma - solved_gamma) < 1e-3))

    def test_solved_z0(self):
        for testgroup in self.testgroup_list:
            medium_z0 = testgroup["rlgc_medium"].z0
            solved_z0 = testgroup["cal"].z0
            self.assertTrue(max(np.abs(medium_z0 - solved_z0) < 1e-3))

    def test_ref_plane_shift(self):
        for testgroup in self.testgroup_list:
            vna = testgroup["vna"]
            cal = testgroup["cal"]

            # Run a different calibration with different line length
            # and calibration plane offsets. The result should be equivalent.
            cal_shift = skrf_cal.NISTMultilineTRL(
                measured=testgroup["std_meas"],
                isolation=testgroup["std_meas"][1],
                Grefls=[-1],
                refl_offset=0,
                ref_plane=-500e-6,
                l=[0, 10e-6, 100e-6, 800e-6],
                c0=cal.c0,
                z0_ref=cal.z0_ref,
                gamma_root_choice='real',
                switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
            )
            cal_shift.run()

            # compare solved error coefficients
            for k in cal.coefs.keys():
                self.assertTrue(
                    all(np.abs(cal.coefs[k] - cal_shift.coefs[k]) < 1e-9)
                )

    def test_numpy_float_arguments(self):
        # see gh-895
        for testgroup in self.testgroup_list:
            vna = testgroup["vna"]

            cal = skrf_cal.NISTMultilineTRL(
                measured=testgroup["std_meas"][:3],
                Grefls=[-1],
                l=[np.float64(1000e-6), 1010e-6],
                switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
            )
            cal.run()
            cal.apply_cal(testgroup["std_meas"][0])

            cal = skrf_cal.NISTMultilineTRL(
                measured=testgroup["std_meas"][:3],
                Grefls=[-1],
                l=[1000e-6, 1010e-6],
                z0_ref=np.float64(50),
                z0_line=np.float64(50),
                switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
            )
            cal.run()
            cal.apply_cal(testgroup["std_meas"][0])


class TUGMultilineTRLTest(NISTMultilineTRLTest):
    """
    Test skrf.calibration.TUGMultilineTRL
    """
    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.TUGMultilineTRL(
            line_meas=[std_meas[0]] + std_meas[3:],
            line_lengths=[0, 100e-6, 200e-6, 900e-6],
            er_est=1,
            reflect_meas=std_meas[1:3],
            reflect_est=[-1, 1],
            isolation=std_meas[1],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal


class LRMTest(EightTermTest):
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        std_defs_incomplete_knowledge = [
            medium.line(d=50, unit='um', name='thru'),
            medium.short(nports=2, name='short'),
            skrf.two_port_reflect(
                medium.load(0, nports=1, name='match'),
                medium.load(0, nports=1, name='match')
            )
        ]

        imperfect_reflect = (
            medium.inductor(5e-12) **
            medium.load(-0.95, nports=1, name='short')
        )
        std_defs = [
            medium.line(d=50, unit='um', name='thru'),
            skrf.two_port_reflect(imperfect_reflect, imperfect_reflect),
            skrf.two_port_reflect(
                medium.load(0, nports=1, name='match'),
                medium.load(0, nports=1, name='match')
            )
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.LRM(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[2],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"])
        )
        cal.run()
        return cal

    def test_solved_reflect(self):
        for testgroup in self.testgroup_list:
            self.assertEqual(
                testgroup["std_fulldefs"][1].s11,
                testgroup["std_fulldefs"][1].s22
            )
            self.assertEqual(
                testgroup["std_fulldefs"][1].s11,
                testgroup["cal"].solved_r
            )


class LRRMTest(EightTermTest):
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        imperfect_short = (
            medium.inductor(5e-12) **
            medium.load(-0.95, nports=1, name='short')
        )
        imperfect_open = (
            medium.shunt_capacitor(5e-15) **
            medium.open(nports=1, name='open')
        )
        parasitic_l = np.random.default_rng().uniform(1e-12, 20e-12)
        imperfect_load = (
            medium.inductor(L=parasitic_l) **
            medium.load(0.1, nports=1, name='load')
        )
        imperfect_thru = medium.line(d=50, z0=75, unit='um', name='thru')

        testgroup["parasitic_l"] = parasitic_l

        # make sure calibration works with non-symmetric thru
        imperfect_thru.s[:,1,1] += 0.02 + 0.05j

        std_defs_incomplete_knowledge = [
            imperfect_thru,
            medium.short(nports=2, name='short'),
            medium.load(1, nports=2, name='open'),
            medium.load(0.1, nports=2, name='load')
        ]

        std_defs = [
            imperfect_thru,
            skrf.two_port_reflect(imperfect_short, imperfect_short),
            skrf.two_port_reflect(imperfect_open, imperfect_open),
            skrf.two_port_reflect(imperfect_load, imperfect_load)
        ]
        std_meas = [vna.measure(std) for std in std_defs]
        return std_defs, std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.LRRM(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[3],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"]),
            z0=std_meas[0].z0[0,0]
        )
        cal.run()
        return cal

    # Test the solved standards, don't use exact equality because of inductance
    # fitting tolerance.
    def test_solved_inductance(self):
        for testgroup in self.testgroup_list:
            defined_l = testgroup["parasitic_l"]
            solved_l = np.mean(testgroup["cal"].solved_l)

            self.assertTrue(np.abs(defined_l - solved_l) < 1e-3 * defined_l)

    def test_solved_r1(self):
        for testgroup in self.testgroup_list:
            defined_r1 = testgroup["std_fulldefs"][1]
            solved_r1 = testgroup["cal"].solved_r1

            self.assertEqual(defined_r1.s11, defined_r1.s22)
            self.assertEqual(defined_r1.s11, solved_r1)

    def test_solved_r2(self):
        for testgroup in self.testgroup_list:
            defined_r2 = testgroup["std_fulldefs"][2]
            solved_r2 = testgroup["cal"].solved_r2

            self.assertEqual(defined_r2.s11, defined_r2.s22)
            self.assertEqual(defined_r2.s11, solved_r2)


class LRRMNoFittingTest(EightTermTest):
    def setup_std_for_testgroup(self, medium, vna, testgroup):
        imperfect_short = (
            medium.inductor(5e-12) **
            medium.load(-0.95, nports=1, name='short')
        )
        imperfect_open = (
            medium.shunt_capacitor(5e-15) **
            medium.open(nports=1, name='open')
        )
        parasitic_l = np.random.default_rng().uniform(1e-12, 20e-12)
        imperfect_load = (
            medium.inductor(L=parasitic_l) **
            medium.load(0, nports=1, name='load')
        )
        imperfect_thru = medium.line(d=50, unit='um', name='thru')

        testgroup["parasitic_l"] = parasitic_l

        std_defs_incomplete_knowledge = [
            imperfect_thru,
            medium.short(nports=2, name='short'),
            medium.load(1, nports=2, name='open'),
            # Doesn't work correctly for non-50 ohm match.
            medium.load(0, nports=2, name='load')
        ]

        std_defs = [
            imperfect_thru,
            skrf.two_port_reflect(imperfect_short, imperfect_short),
            skrf.two_port_reflect(imperfect_open, imperfect_open),
            skrf.two_port_reflect(imperfect_load, imperfect_load)
        ]
        std_meas = [vna.measure(std) for std in std_defs_incomplete_knowledge]
        return std_defs, std_defs_incomplete_knowledge, std_meas

    def setup_cal(self, vna, std_defs, std_meas):
        cal = skrf_cal.LRRM(
            ideals=std_defs,
            measured=std_meas,
            isolation=std_meas[3],
            switch_terms=(vna.err["gamma_f"], vna.err["gamma_r"]),
            z0=std_meas[0].z0[0,0],
            match_fit='none',
        )
        cal.run()
        return cal


class TwelveTermTest(AbstractTwoPortTest, unittest.TestCase):
    """
    Test skrf.calibration.TwelveTermTest
    """
    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs, std_meas

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
    def setup_std_for_testgroup(self, medium, vna, testgroup):
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
        return std_defs, std_defs_incomplete_knowledge, std_meas

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
