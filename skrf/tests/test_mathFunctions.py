import skrf as rf
import unittest
from numpy import log, pi
from numpy.testing import assert_equal, run_module_suite, assert_almost_equal


class TestUnitConversions(unittest.TestCase):
    """
    Test unit-conversion functions
    """

    def setUp(self):
        pass        

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
        
    def test_mag_2_db(self):
        """
        Test magnitude to dB conversion with:
            2 -> +3 dB
        """
        assert_almost_equal(rf.mag_2_db10(2), 10*log(2)/log(10))


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

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
    
