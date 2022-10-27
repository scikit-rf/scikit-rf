import unittest
import os
import numpy as npy
from pathlib import Path
from zipfile import ZipFile

import skrf as rf
from skrf.io.touchstone import Touchstone


class TouchstoneTestCase(unittest.TestCase):
    """
    TouchstoneTestCase tests the IO of Touchstone files
    """
    def setUp(self):
        """
        Sets up the test directory
        """
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'

    def test_read_data(self):
        """
        This test reads data from simple_touchstone.s2p and compares with known
        true values.
        """
        filename = os.path.join(self.test_dir, 'simple_touchstone.s2p')
        touch = Touchstone(filename)
        f, s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance)
        f_true = npy.array([1.00000000e+09, 1.10000000e+09])
        s_true = npy.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue(z0 == z0_true)

    def test_read_with_special_encoding(self):
        """
        Read Touchstone files with various file encoding
        """
        filename_utf8_sig = os.path.join(self.test_dir, 'test_encoding_UTF-8-SIG.s2p')
        filename_latin1 = os.path.join(self.test_dir, 'test_encoding_ISO-8859-1.s2p')
        filename_unknown = os.path.join(self.test_dir, 'test_encoding_unknown.s2p')
        
        # most common situation: try and error guessing the encoding
        Touchstone(filename_utf8_sig)
        Touchstone(filename_latin1)
        Touchstone(filename_unknown)
                
        # specify the encoding  
        Touchstone(filename_latin1, encoding='ISO-8859-1')
        Touchstone(filename_utf8_sig, encoding='utf_8_sig')
        
    def test_read_from_fid(self):
        """
        This tests reading touch stone data from a file object as compared with
        a string path and name of the file.
        """
        with open(os.path.join(self.test_dir, 'simple_touchstone.s2p')) as fid:
            touch = Touchstone(fid)
        f, s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance)
        f_true = npy.array([1.00000000e+09, 1.10000000e+09])
        s_true = npy.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue(z0 == z0_true)

    def test_get_sparameter_data(self):
        """
        This tests the get_sparameter_data function.

        """
        with open(os.path.join(self.test_dir, 'simple_touchstone.s2p')) as fid:
            touch = Touchstone(fid)

        expected_keys = ["frequency", "S11R", "S11I", "S12R", "S12I",
                "S21R", "S21I", "S22R", "S22I", ]

        unexpected_keys = ['S11DB', 'S11M', ]

        # get dict data structure
        sp_ri = touch.get_sparameter_data(format="ri")
        # Get dict data in db to check ri -> db/angle conversion
        sp_db = touch.get_sparameter_data(format="db")

        # test data structure
        for ek in expected_keys:
            self.assertTrue(ek in sp_ri)

        for uk in unexpected_keys:
            self.assertFalse(uk in sp_ri)

        # test data contents
        expected_sp_ri = {
            'frequency': npy.array([1.0e+09, 1.1e+09]),
            'S11R': npy.array([1., 9.]),
            'S11I': npy.array([ 2., 10.]),
            'S21R': npy.array([ 3., 11.]),
            'S21I': npy.array([ 4., 12.]),
            'S12R': npy.array([ 5., 13.]),
            'S12I': npy.array([ 6., 14.]),
            'S22R': npy.array([ 7., 15.]),
            'S22I': npy.array([ 8., 16.]),
        }

        S11 = npy.array([1., 9.]) + 1j*npy.array([ 2., 10.])
        S21 = npy.array([ 3., 11.]) + 1j*npy.array([ 4., 12.])
        S12 = npy.array([ 5., 13.]) + 1j*npy.array([ 6., 14.])
        S22 = npy.array([ 7., 15.]) + 1j*npy.array([ 8., 16.])
        expected_sp_db = {
            'frequency': npy.array([1.0e+09, 1.1e+09]),
            'S11DB': 20*npy.log10(npy.abs(S11)),
            'S11A': npy.angle(S11, deg=True),
            'S21DB': 20*npy.log10(npy.abs(S21)),
            'S21A': npy.angle(S21, deg=True),
            'S12DB': 20*npy.log10(npy.abs(S12)),
            'S12A': npy.angle(S12, deg=True),
            'S22DB': 20*npy.log10(npy.abs(S22)),
            'S22A': npy.angle(S22, deg=True),
        }

        for k in sp_ri:
            self.assertTrue(k in expected_sp_ri)

            self.assertTrue( (expected_sp_ri[k] == sp_ri[k]).all(),
                    msg='Field %s does not match. Expected "%s", got "%s"'%(
                        k, str(expected_sp_ri[k]), str(sp_ri[k]))  )

        for k in sp_db:
            self.assertTrue(k in expected_sp_db)

            self.assertTrue( (expected_sp_db[k] == sp_db[k]).all(),
                    msg='Field %s does not match. Expected "%s", got "%s"'%(
                        k, str(expected_sp_db[k]), str(sp_db[k]))  )


    def test_HFSS_touchstone_files(self):
        """
        HFSS can export additional information in the Touchstone file
        such as gamma and z0 for each port. However, the way there are stored
        depend of the HFSS version...

        In versions before 2020 R2, data were stored as following:

        ! Gamma ! re1 im2 re2 im2 re3 im3 re4 im4
        !       re5 im5 re6 im6 re7 im7 re8 im8
        !       re9 im9  [etc]
        ! Port Impedancere1 im2 re2 im2 re3 im3 re4 im4
        !       re5 im5 re6 im6 re7 im7 re8 im8
        !       re9 im9  [etc]
            [NB: there is an extra ! before re1 for Gamma]
            [NB: re1 value is stuck to the 'e' of Impedance]

        Since version 2020 R2n the data are stored in a single line:

        ! Gamma re1 im2 re2 im2 re3 im3 re4 im4 re5 im5 re6 im6 re7 im7 re8 im8 [etc]
        ! Port Impedance re1 im2 re2 im2 re3 im3 re4 im4 re5 im5 re6 im6 re7 im7 re8 im8 [etc]
            [NB: re1 value is no more stuck to the 'e' of Impedance]

        This test checks that the shape of gamma and z0 matche the rank of the Network
        for Touchstone files of various port obtained from different HFSS version
        """
        HFSS_RELEASES= ['HFSS_2019R2', 'HFSS_2020R2']

        p = Path('.')
        for hfss_release in HFSS_RELEASES:
            for sNp_file in p.glob(hfss_release+'/*.s*'):
                touchst = Touchstone(sNp_file.as_posix())
                gamma, z0 = touchst.get_gamma_z0()

                assert(gamma.shape[-1] == touchst.rank)
                assert(z0.shape[-1] == touchst.rank)

suite = unittest.TestLoader().loadTestsFromTestCase(TouchstoneTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

