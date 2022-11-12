import unittest
import os

import pytest
import skrf as rf
from skrf.media import Coaxial
import numpy as npy
from numpy.testing import assert_almost_equal, assert_array_almost_equal, run_module_suite

class MediaTestCase(unittest.TestCase):
    """

    """
    def setUp(self):
        """

        """
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )



    def test_line(self):
        """
        """
        fname = os.path.join(self.files_dir,\
                'coaxial.s2p')
        qucs_ntwk = rf.Network(fname)

        a_media = Coaxial(
            frequency = qucs_ntwk.frequency,
            Dint=1e-3, Dout=3e-3, epsilon_r=2.29, \
            tan_delta=4e-4, sigma=1./1.68e-8 \
            )
        skrf_ntwk = a_media.thru(z0=50)**a_media.line(200e-3,'m')\
                    **a_media.thru(z0=50)
        # Equal assertion fails if tan_delta or resistivity are non-zero
        #self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertTrue(
            max(abs(skrf_ntwk.s_mag[:,1,0] - qucs_ntwk.s_mag[:,1,0])) < 1e-3
            )


    def test_init_from_attenuation_VF_units(self):
        """
        Test the attenuation unit conversions in the Coaxial classmethod
        `from_attenuation_VF_units`.
        """
        # create a dummy Coaxial media for various attenuation and test the
        # resulting alpha values (real part of gamma)
        frequency = rf.Frequency(npy.random.rand(), unit='GHz', npoints=1)
        _att = npy.random.rand()
        # dB/m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/m')
        assert_almost_equal(coax.gamma.real,  rf.db_2_np(_att))
        # dB/100m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/100m')
        assert_almost_equal(coax.gamma.real,  rf.db_2_np(_att)/100)
        # dB/feet
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/feet')
        assert_almost_equal(coax.gamma.real,  rf.db_2_np(_att)*rf.meter_2_feet())
        # dB/100m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='dB/100feet')
        assert_almost_equal(coax.gamma.real,  rf.db_2_np(_att)/100*rf.meter_2_feet())
        # Neper/m
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='Np/m')
        assert_almost_equal(coax.gamma.real,  _att)
        # Neper/feet
        coax = Coaxial.from_attenuation_VF(frequency=frequency, VF=1, att=_att, unit='Np/feet')
        assert_almost_equal(coax.gamma.real,  _att*rf.meter_2_feet())


    def test_init_from_attenuation_VF_array_att(self):
        """
        Test passing array as attenuation in the Coaxial classmethod
        `from_attenuation_VF_units`.
        """
        # create a Coaxial media for frequency-dependent attenuation and 
        # test the resulting alpha values (real part of gamma)
        frequency = rf.Frequency(start=1, stop=2, unit='GHz', npoints=101)
        # k0k1k2 attenuation model
        # values taken for HUBER+SUHNER DATA SHEET Coaxial Cable S_10172_B-1
        # attenuation in dB/m for frequency in GHz
        att = 0 + 0.0826*npy.sqrt(frequency.f_scaled) + 0.0129*frequency.f_scaled

        coax = Coaxial.from_attenuation_VF(frequency=frequency, att=att, unit='dB/m')
        # check alpha in gamma
        assert_array_almost_equal(rf.db_2_np(att), coax.gamma.real)

        # if the attenuation array length does not match the frequency,
        # should raise a ValueError
        frequency2 = rf.Frequency(start=1, stop=2, unit='GHz', npoints=10)
        with self.assertRaises(ValueError):
            coax = Coaxial.from_attenuation_VF(frequency=frequency2, att=att)

    def test_R(self):
        freq = rf.Frequency(0, 100, 2)

        rho = 1e-7
        dint = 0.44e-3
        coax = Coaxial(freq, z0=50, Dint=dint, Dout=1.0e-3, sigma=1/rho)

        dc_res = rho / (npy.pi * (dint/2)**2)

        # Old R calculation valid only when skin depth is much smaller
        # then inner conductor radius
        with pytest.warns(RuntimeWarning, match="divide by zero"):
            R_simple = coax.Rs/(2*npy.pi)*(1/coax.a + 1/coax.b)

            self.assertTrue(abs(1 - coax.R[0]/dc_res) < 1e-2)
            self.assertTrue(abs(1 - coax.R[1]/R_simple[1]) < 1e-2)

if __name__ == "__main__":
    # Launch all tests
    run_module_suite()
