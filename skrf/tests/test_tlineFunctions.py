import skrf as rf
import unittest
from numpy import real, imag, linspace, pi, tanh, array, conj
from numpy.random import rand
from numpy.testing import assert_equal, run_module_suite, assert_almost_equal


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
        Gamma_in = rf.zl_2_Gamma0(self.z0, self.zin)
        
        expected_Gamma_in = (self.zin - self.z0)/(self.zin + self.z0)
        
        assert_equal(Gamma_in, expected_Gamma_in)

    def test_propagation_constant_from_reflection_coefficient(self):
        """
        Test the propagation constant value deduced from reflection coef
        """
        Gamma_in = rf.zl_2_Gamma0(self.z0, self.zin)
        Gamma_l = -1  # short
        
        gamma = rf.reflection_coefficient_2_propagation_constant(Gamma_in, 
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
        theta_scalar = rf.electrical_length(self.gamma0, self.f0, self.d)
        theta_scalar_expected = self.gamma0 * self.d
        assert_almost_equal(theta_scalar, theta_scalar_expected)

        # test for gamma passed as array like 
        theta_array = rf.electrical_length(self.gammas, self.f0, self.d)
        theta_array_expected = self.gammas * self.d
        assert_almost_equal(theta_array, theta_array_expected)

        # test for gamma passed as function
        theta_function = rf.electrical_length(self.gamma_from_f, self.fs, self.d)
        theta_function_expected = self.gamma_from_f(self.fs) * self.d
        assert_almost_equal(theta_function, theta_function_expected)


    def test_length_from_electrical_distance(self):
        """
        Test the conversions from electrical length to physical distances. 
        """
        # test for gamma passed as scalar 
        d_scalar = rf.electrical_length_2_distance(self.theta0, self.gamma0, self.f0, deg=False)
        d_scalar_expected = real(self.theta0 / self.gamma0)
        assert_almost_equal(d_scalar, d_scalar_expected)

        # test for gamma passed as array-like
        d_array = rf.electrical_length_2_distance(self.theta0, self.gammas, self.f0, deg=False)
        d_array_expected = real(self.theta0 / self.gammas)
        assert_almost_equal(d_array, d_array_expected)

        # test for gamma passed as function 
        d_function = rf.electrical_length_2_distance(self.theta0, self.gamma_from_f, self.fs, deg=False)
        d_function_expected = real(self.theta0 / self.gamma_from_f(self.fs))
        assert_almost_equal(d_function, d_function_expected)
        # with theta passed as array
        d_function = rf.electrical_length_2_distance(self.thetas, self.gamma_from_f, self.fs, deg=False)
        d_function_expected = real(self.thetas / self.gamma_from_f(self.fs))
        assert_almost_equal(d_function, d_function_expected)
        
        # with theta passed in degrees
        d_scalar = rf.electrical_length_2_distance(self.theta0*180/pi, self.gamma0, self.f0, deg=True)
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
        
        v2, i2 = rf.voltage_current_propagation(v1, i1, z0, theta)
        
        assert_almost_equal(v2, v1)
        assert_almost_equal(i2, i1)
        
    def test_d_wavelength(self):
        """
        Propagate voltage and current on a d=lambda lossless transmission line.
        Voltage and current are equal.
        """
        gamma = array([1j])
        z0 = 50
        v1 = rand()
        i1 = rand()
        theta = 1j*2*pi
        
        v2, i2 = rf.voltage_current_propagation(v1, i1, z0, theta)
        assert_almost_equal(v2, v1)
        assert_almost_equal(i2, i1)

    def test_d_half_wavelength(self):
        """
        Propagate voltage and current on a d=lambda/2 lossless transmission line. 
        Voltage and current are inversed.
        """
        gamma = array([1j])
        z0 = 50
        v1 = rand()
        i1 = rand()
        theta = 1j*pi
        
        v2, i2 = rf.voltage_current_propagation(v1, i1, z0, theta)
        
        assert_almost_equal(v2, -v1)
        assert_almost_equal(i2, -i1)


if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
    
