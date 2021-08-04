from skrf.mathFunctions import LOG_OF_NEG
import skrf as rf
import unittest
import numpy as np
from numpy import e, log, pi, isnan, inf
from numpy.testing import assert_equal, run_module_suite, assert_almost_equal


class TestUnitConversions(unittest.TestCase):
    """
    Test unit-conversion functions
    """

    def setUp(self):
        pass        

    def test_complex_2_magnitude(self):
        """
        Test complex to magnitude conversion with:
            5 = 3 + 4j
        """
        assert_almost_equal(rf.complex_2_magnitude(3+4j), 5.0)
        

    def test_complex_2_db10(self):
        """
        Test complex to db10 conversion with:
            10 [dB] = 10 * log10(6+8j)
        """
        assert_almost_equal(rf.complex_2_db10(6+8j), 10.0)


    def test_complex_2_degree(self):
        """
        Test complex to degree conversion with:
            90 = angle(0 + 1j)
        """
        assert_almost_equal(rf.complex_2_degree(0+1j), 90.0)


    def test_complex_2_quadrature(self):
        """
        Test complex to quadrature conversion with:
            2, pi  = abs(2j), angle(2j) * abs(2j) 
        """
        assert_almost_equal(rf.complex_2_quadrature(0+2j), (2, pi))


    def test_complex_components(self):
        """
        Test complex components:
        """
        assert_almost_equal(rf.complex_components(0+2j), (0, 2, 90, 2, pi))


    def test_complex_2_reim(self):
        """
        Test complex to (real, imag) conversion:
        """
        assert_almost_equal(rf.complex_2_reim(1+2j), (1,2))


    def test_magnitude_2_db(self):
        """
        Test magnitude to db conversion
        """
        assert_almost_equal(rf.magnitude_2_db(10, True), 20)
        assert_almost_equal(rf.magnitude_2_db(10, False), 20)

        assert_almost_equal(rf.magnitude_2_db(0), -inf)
        assert_almost_equal(rf.magnitude_2_db(-1, True), LOG_OF_NEG)
        assert_almost_equal(rf.magnitude_2_db([10, -1], True), [20, LOG_OF_NEG])
        self.assertTrue(isnan(rf.magnitude_2_db(-1, False)))

        assert_equal(rf.mag_2_db, rf.magnitude_2_db) # Just an alias

        
    def test_mag_2_db10(self):
        """
        Test magnitude to db10 conversion
        """
        assert_almost_equal(rf.mag_2_db10(10, True), 10)
        assert_almost_equal(rf.mag_2_db10(10, False), 10)

        assert_almost_equal(rf.magnitude_2_db(0), -inf)
        assert_almost_equal(rf.mag_2_db10(-1, True), LOG_OF_NEG)
        assert_almost_equal(rf.mag_2_db10([10, -1], True), [10, LOG_OF_NEG])
        self.assertTrue(isnan(rf.mag_2_db10(-1, False)))


    def test_db10_2_mag(self):
        """
        Test db10 to mag conversion
        """
        assert_almost_equal(rf.db10_2_mag(3+4j), 10**((3+4j)/10))


    def test_magdeg_2_reim(self):
        """
        Test (mag,deg) to (re+j*im)
        """
        assert_almost_equal(rf.magdeg_2_reim(1, 90), (0+1j))


    def test_dbdeg_2_reim(self):
        """
        Test (db, deg) to (re+j*im)
        """
        assert_almost_equal(rf.dbdeg_2_reim(20,90), (0+10j))


    def test_np_2_db(self):
        """
        Test Np to dB conversion with:
            1 [Np] = 20/ln(10) [dB]
        """
        assert_almost_equal(rf.np_2_db(1), 20/log(10))


    def test_db_2_np(self):
        """
        Test dB to Np conversion with: 
            1 [dB] = ln(10)/20 [Np]
        """
        assert_almost_equal(rf.db_2_np(1), log(10)/20)


    def test_radian_2_degree(self):
        assert_almost_equal(rf.radian_2_degree(pi), 180)


    def test_feet_2_meter(self):
        """
        Test feet to meter length conversion
        """
        assert_almost_equal(rf.feet_2_meter(0.01), 0.003048)
        assert_almost_equal(rf.feet_2_meter(1), 0.3048)
        

    def test_meter_2_feet(self):
        """
        Test meter to feet length conversion
        """
        assert_almost_equal(rf.meter_2_feet(0.01), 0.0328084)
        assert_almost_equal(rf.meter_2_feet(1), 3.28084)


    def test_db_per_100feet_2_db_per_100meter(self):
        """
        Test attenuation unit conversion dB/100feet to dB/100m
        """
        assert_almost_equal(rf.db_per_100feet_2_db_per_100meter(), rf.meter_2_feet(), decimal=2)
        assert_almost_equal(rf.db_per_100feet_2_db_per_100meter(2.5), 8.2, decimal=2)
        assert_almost_equal(rf.db_per_100feet_2_db_per_100meter(0.28), 0.92, decimal=2)

    def test_inf_to_num(self):
        """
        Test inf_to_num function
        """
        # scalar
        assert_equal(rf.inf_to_num(np.inf), rf.INF)
        assert_equal(rf.inf_to_num(-np.inf), -rf.INF)
        
        # array
        x = np.array([0, np.inf, 0, -np.inf])
        y = np.array([0, rf.INF, 0, -rf.INF])
        assert_equal(rf.inf_to_num(x), y)

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
    
