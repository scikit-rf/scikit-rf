import os
import unittest
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from skrf import Network
from skrf.io.touchstone import Touchstone, read_zipped_touchstones


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
        f_true = np.array([1.00000000e+09, 1.10000000e+09])
        s_true = np.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        comments_after_option_line = "freq	ReS11	ImS11	ReS21	ImS21	ReS12	ImS12	ReS22	ImS22"

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue(z0 == z0_true)
        self.assertTrue(touch.comments_after_option_line == comments_after_option_line)

    def test_double_option_line(self):
        filename = os.path.join(self.test_dir, 'double_option_line.s2p')
        touch = Touchstone(filename)

        self.assertTrue(touch.resistance == 10+10j)

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
        f_true = np.array([1.00000000e+09, 1.10000000e+09])
        s_true = np.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
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


        with pytest.warns(DeprecationWarning):
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
            'frequency': np.array([1.0e+09, 1.1e+09]),
            'S11R': np.array([1., 9.]),
            'S11I': np.array([ 2., 10.]),
            'S21R': np.array([ 3., 11.]),
            'S21I': np.array([ 4., 12.]),
            'S12R': np.array([ 5., 13.]),
            'S12I': np.array([ 6., 14.]),
            'S22R': np.array([ 7., 15.]),
            'S22I': np.array([ 8., 16.]),
        }

        S11 = np.array([1., 9.]) + 1j*np.array([ 2., 10.])
        S21 = np.array([ 3., 11.]) + 1j*np.array([ 4., 12.])
        S12 = np.array([ 5., 13.]) + 1j*np.array([ 6., 14.])
        S22 = np.array([ 7., 15.]) + 1j*np.array([ 8., 16.])
        expected_sp_db = {
            'frequency': np.array([1.0e+09, 1.1e+09]),
            'S11DB': 20*np.log10(np.abs(S11)),
            'S11A': np.angle(S11, deg=True),
            'S21DB': 20*np.log10(np.abs(S21)),
            'S21A': np.angle(S21, deg=True),
            'S12DB': 20*np.log10(np.abs(S12)),
            'S12A': np.angle(S12, deg=True),
            'S22DB': 20*np.log10(np.abs(S22)),
            'S22A': np.angle(S22, deg=True),
        }

        for k in sp_ri:
            self.assertTrue(k in expected_sp_ri)

            self.assertTrue( (expected_sp_ri[k] == sp_ri[k]).all(),
                    msg=f'Field {k} does not match. Expected "{expected_sp_ri[k]}", got "{sp_ri[k]}"')

        for k in sp_db:
            self.assertTrue(k in expected_sp_db)

            self.assertTrue( (expected_sp_db[k] == sp_db[k]).all(),
                    msg=f'Field {k} does not match. Expected "{expected_sp_db[k]}", got "{sp_db[k]}"')


        with pytest.warns(DeprecationWarning):
            for k, v in zip(touch.get_sparameter_names(), touch.sparameters.T):
                if k[0] != 'S':
                    # frequency doesn't match because of Hz vs GHz.
                    continue
                self.assertTrue(np.all(expected_sp_ri[k] == v))


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

        p = Path(self.test_dir)
        for hfss_release in HFSS_RELEASES:
            for sNp_file in p.glob(hfss_release+'/*.s*'):
                touchst = Touchstone(sNp_file.as_posix())
                gamma, z0 = touchst.get_gamma_z0()
                print(z0)

                assert(gamma.shape[-1] == touchst.rank)
                assert(z0.shape[-1] == touchst.rank)


    def test_touchstone_2(self):
        net = Touchstone(os.path.join(self.test_dir, "ts/ansys.ts"))

        assert net.port_names[0] == "U29_B6_1024G_EAS3QB_A_DBI.37.FD_0-1"
        assert net.port_names[1] == "U29_B6_1024G_EAS3QB_A_DBI.38.GND"
        assert net.port_names[2] == "U40_178BGA.E10.FD_0-1"

    def test_ansys_modal_data(self):
        net = Touchstone(os.path.join(self.test_dir, "ansys_modal_data.s2p"))
        z0 = np.array([
            [51. +1.j, 52. +2.j],
            [61.+11.j, 62.+12.j]
        ])
        assert np.allclose(net.z0, z0)

    @pytest.mark.skip
    def test_ansys_terminal_data(self):
        net = Touchstone(os.path.join(self.test_dir, "ansys_terminal_data.s4p"))

        z0 = np.array([
            [51. +1.j, 52. +2.j, 53. +3.j, 54. +4.j],
            [61.+11.j, 62.+12.j, 63.+13.j, 64.+14.j]
        ])
        assert np.allclose(net.z0, z0)

    def test_read_zipped_touchstones(self):
        file = ZipFile(os.path.join(self.test_dir, "ntwk_zip.zip"))
        ntwk1 = Network(os.path.join(self.test_dir, "ntwk1.s2p"))
        ntwk2 = Network(os.path.join(self.test_dir, "ntwk2.s2p"))
        ntwk3 = Network(os.path.join(self.test_dir, "ntwk3.s2p"))

        read1 = read_zipped_touchstones(file, "Folder1")
        read2 = read_zipped_touchstones(file, "Folder2")
        read3 = read_zipped_touchstones(file)

        assert read1 == {"ntwk1": ntwk1}
        assert read2 == {"ntwk1": ntwk1, "ntwk2": ntwk2}
        assert read3 == {"ntwk3": ntwk3}


suite = unittest.TestLoader().loadTestsFromTestCase(TouchstoneTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
