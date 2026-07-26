import unittest

import numpy as np
from numpy import array, imag, linspace, pi, real
from numpy.testing import assert_allclose, assert_almost_equal, assert_equal
from scipy.constants import epsilon_0, mu_0

import skrf.tlineFunctions as tlFuncs


class TestBasicTransmissionLine(unittest.TestCase):
    """
    Test reflection coefficient-related functions
    """

    def setUp(self):
        # define example test case
        self.d = 1.5  # line length [m]
        self.z0 = 100  # line characteristic impedance [Ohm]
        self.zin = 40 - 280j  # line input impedance [Ohm]

    def test_input_reflection_coefficient(self):
        """
        Test the input reflection coefficient value of the line.
        """
        Gamma_in = tlFuncs.zl_2_Gamma0(self.z0, self.zin)

        expected_Gamma_in = (self.zin - self.z0)/(self.zin + self.z0)

        assert_equal(Gamma_in, expected_Gamma_in)

    def test_propagation_constant_from_reflection_coefficient(self):
        """
        Test the propagation constant value deduced from reflection coef
        """
        Gamma_in = tlFuncs.zl_2_Gamma0(self.z0, self.zin)
        Gamma_l = -1  # short

        gamma = tlFuncs.reflection_coefficient_2_propagation_constant(Gamma_in,
                                                                 Gamma_l,
                                                                 self.d)
        expected_gamma = 0.02971 + 1.272j

        assert_almost_equal(real(gamma), real(expected_gamma), decimal=4)
        assert_almost_equal(imag(gamma), imag(expected_gamma), decimal=4)


class ElectricalLengthTests(unittest.TestCase):
    """
    Test the functions related to electrical length conversions.
    """
    def setUp(self):
        self.d = 1.5  # m
        self.gamma0 = 0.2 + 5j
        self.gammas = 0.2 + 1j*linspace(1, 10, num=50)
        self.f0 = 50e6
        self.fs = linspace(1, 50, num=50)*1e6
        self.theta0 = self.gamma0 * self.d
        self.thetas = self.gammas * self.d


    def gamma_from_f(self, f0):
        """
        Dummy gamma(f)

        Parameters
        ----------
        f0 : number of array-like
            frequency in Hz

        Returns
        -------
        gamma : number of array-like
            propagation constant

        """
        return (0.2 + 5j)*f0

    def test_electrical_length_from_length(self):
        """
        Test the conversions from physical distance to electrical lengths.
        """
        # test for gamma passed as scalar
        theta_scalar = tlFuncs.electrical_length(self.gamma0, self.f0, self.d)
        theta_scalar_expected = self.gamma0 * self.d
        assert_almost_equal(theta_scalar, theta_scalar_expected)

        # test for gamma passed as array like
        theta_array = tlFuncs.electrical_length(self.gammas, self.f0, self.d)
        theta_array_expected = self.gammas * self.d
        assert_almost_equal(theta_array, theta_array_expected)

        # test for gamma passed as function
        theta_function = tlFuncs.electrical_length(self.gamma_from_f, self.fs, self.d)
        theta_function_expected = self.gamma_from_f(self.fs) * self.d
        assert_almost_equal(theta_function, theta_function_expected)


    def test_length_from_electrical_distance(self):
        """
        Test the conversions from electrical length to physical distances.
        """
        # test for gamma passed as scalar
        d_scalar = tlFuncs.electrical_length_2_distance(self.theta0, self.gamma0, self.f0, deg=False)
        d_scalar_expected = real(self.theta0 / self.gamma0)
        assert_almost_equal(d_scalar, d_scalar_expected)

        # test for gamma passed as array-like
        d_array = tlFuncs.electrical_length_2_distance(self.theta0, self.gammas, self.f0, deg=False)
        d_array_expected = real(self.theta0 / self.gammas)
        assert_almost_equal(d_array, d_array_expected)

        # test for gamma passed as function
        d_function = tlFuncs.electrical_length_2_distance(self.theta0, self.gamma_from_f, self.fs, deg=False)
        d_function_expected = real(self.theta0 / self.gamma_from_f(self.fs))
        assert_almost_equal(d_function, d_function_expected)
        # with theta passed as array
        d_function = tlFuncs.electrical_length_2_distance(self.thetas, self.gamma_from_f, self.fs, deg=False)
        d_function_expected = real(self.thetas / self.gamma_from_f(self.fs))
        assert_almost_equal(d_function, d_function_expected)

        # with theta passed in degrees
        d_scalar = tlFuncs.electrical_length_2_distance(self.theta0*180/pi, self.gamma0, self.f0, deg=True)
        d_scalar_expected = real(self.theta0 / self.gamma0)
        assert_almost_equal(d_scalar, d_scalar_expected)





class TestVoltageCurrentPropagation(unittest.TestCase):
    def setUp(self):
        pass

    def test_d_zero(self):
        """
        Propagate voltage and current on a d=0 transmission line.
        Voltage and current are of course equal.
        """
        gamma = array([1j])
        d = 0
        z0 = 50
        v1 = 3
        i1 = 2
        theta = gamma * d

        v2, i2 = tlFuncs.voltage_current_propagation(v1, i1, z0, theta)

        assert_almost_equal(v2, v1)
        assert_almost_equal(i2, i1)

    def test_d_wavelength(self):
        """
        Propagate voltage and current on a d=lambda lossless transmission line.
        Voltage and current are equal.
        """
        z0 = 50
        rng = np.random.default_rng()
        v1 = rng.random()
        i1 = rng.random()
        theta = 1j*2*pi

        v2, i2 = tlFuncs.voltage_current_propagation(v1, i1, z0, theta)
        assert_almost_equal(v2, v1)
        assert_almost_equal(i2, i1)

    def test_d_half_wavelength(self):
        """
        Propagate voltage and current on a d=lambda/2 lossless transmission line.
        Voltage and current are inversed.
        """
        z0 = 50
        rng = np.random.default_rng()
        v1 = rng.random()
        i1 = rng.random()
        theta = 1j*pi

        v2, i2 = tlFuncs.voltage_current_propagation(v1, i1, z0, theta)

        assert_almost_equal(v2, -v1)
        assert_almost_equal(i2, -i1)


class SurfaceImpedanceTests(unittest.TestCase):
    """
    Test the rough surface impedance (transmission line taper method).

    rtol=1e-5 is the tightest safe tolerance: surface_impedance refines only
    until it converges to rtol=1e-6 (its default), and the actual deviations
    are at the 1e-7 level.
    """

    def setUp(self):
        self.f = np.array([1e9, 10e9, 100e9])
        self.sigma_cu = 58e6

    def test_smooth_limit(self):
        """Zero roughness recovers the smooth surface impedance (1+j)*Rs."""
        Zs = tlFuncs.surface_impedance(self.f, {'sigma': self.sigma_cu}, rms_roughness=0)
        Rs = tlFuncs.surface_resistivity(self.f, rho=1/self.sigma_cu, mu_r=1)
        assert_allclose(Zs, (1 + 1j)*Rs, rtol=1e-5)

    def test_smooth_coated_stack(self):
        """Zero roughness with a coating recovers the closed-form input impedance
        of a nickel layer on copper (single impedance transformation)."""
        thickness = 0.5e-6
        sigma_ni, mu_r_ni = 14.5e6, 5
        mats = [{}, {'sigma': sigma_ni, 'mu_r': mu_r_ni}, {'sigma': self.sigma_cu}]
        Zs = tlFuncs.surface_impedance(self.f, mats, rms_roughness=0, boundary_loc=[0, thickness])

        omega = 2*pi*self.f
        ep_ni = epsilon_0 - 1j*sigma_ni/omega  # complex permittivity of the conductors
        ep_cu = epsilon_0 - 1j*self.sigma_cu/omega
        eta_ni = np.sqrt(mu_r_ni*mu_0/ep_ni)
        eta_cu = np.sqrt(mu_0/ep_cu)
        gamma_ni = 1j*omega*np.sqrt(mu_r_ni*mu_0*ep_ni)
        tanh = np.tanh(gamma_ni*thickness)
        expected = eta_ni*(eta_cu + eta_ni*tanh)/(eta_ni + eta_cu*tanh)
        assert_allclose(Zs, expected, rtol=1e-5)

    def test_rough_copper(self):
        """Copper with 1 um rms roughness, against reference values computed with
        the original implementation (https://github.com/ZiadHatab/rough-surface-impedance)."""
        Zs = tlFuncs.surface_impedance(self.f, {'sigma': self.sigma_cu}, rms_roughness=1e-6)
        expected = np.array([0.01075887 + 0.04411716j,
                             0.06303392 + 0.33560499j,
                             0.46406747 + 2.60106873j])
        assert_allclose(Zs, expected, rtol=1e-5)

    def test_copper_on_substrate(self):
        """Bottom side of a microstrip trace: copper against the substrate, rough
        at the copper-substrate boundary (reference values, see test_rough_copper)."""
        Zs = tlFuncs.surface_impedance(self.f, [{'ep_r': 3.2}, {'sigma': self.sigma_cu}],
                                       rms_roughness=0.5e-6)
        expected = np.array([0.00902692 + 0.02705257j,
                             0.04167908 + 0.19687927j,
                             0.27426501 + 1.51542669j])
        assert_allclose(Zs, expected, rtol=1e-5)

    def test_enig_stack(self):
        """Top side of a microstrip trace with ENIG finish: a nearly flat stack of
        gold, nickel and copper (reference values, see test_rough_copper)."""
        mats = [{}, {'sigma': 41.1e6}, {'sigma': 14.5e6, 'mu_r': 20}, {'sigma': self.sigma_cu}]
        Zs = tlFuncs.surface_impedance(self.f, mats, rms_roughness=5e-9,
                                       boundary_loc=[0, 0.05e-6, 4.05e-6])
        expected = np.array([0.07131567 + 0.0552591j,
                             0.18954272 + 0.10120029j,
                             0.34927317 + 0.12440614j])
        assert_allclose(Zs, expected, rtol=1e-5)

    def test_invalid_inputs(self):
        """Erroneous inputs raise ValueError."""
        with self.assertRaises(ValueError):  # non-positive frequency
            tlFuncs.surface_impedance(0, {'sigma': self.sigma_cu}, 1e-6)
        with self.assertRaises(ValueError):  # unknown material key (typo)
            tlFuncs.surface_impedance(1e9, {'sigma': self.sigma_cu, 'epr': 4}, 1e-6)
        with self.assertRaises(ValueError):  # boundaries not in increasing depth order
            tlFuncs.surface_impedance(1e9, [{}, {'sigma': 14.5e6}, {'sigma': self.sigma_cu}], 1e-6,
                                      boundary_loc=[1e-6, 0])
