import os
import tempfile
import unittest

import numpy as np

import skrf as rf
from skrf.io.mdif import Mdif
from skrf.networkSet import NetworkSet


class MdifTestCase(unittest.TestCase):
    """
    Test the IO of GMDIF files
    """
    def setUp(self):
        """
        Sets up the test directory
        """
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/MDIF_CITI_MDL/'

        # constructor from filename
        self.oneport_example1 = Mdif(self.test_dir + 'test_1p_gmdif.mdf')
        self.oneport_example2 = Mdif(self.test_dir + 'test_1p_gmdif_2.mdf')
        self.twoport_example1 = Mdif(self.test_dir + 'test_2p_gmdif.mdf')
        self.twoport_example2 = Mdif(self.test_dir + 'test_2p_gmdif_2.mdf')
        self.twoport_example3 = Mdif(self.test_dir + 'test_2p_gmdif_3.mdf')
        self.twoport_example_z = Mdif(self.test_dir + 'test_2p_gmdif_z.mdf')
        self.twoport_example_yz = Mdif(self.test_dir + 'test_2p_gmdif_yz.mdf')

        self.fourport_example1 = Mdif(self.test_dir + 'test_4p_gmdif.mdf')

        self.examples = [self.oneport_example1, self.oneport_example2,
                         self.twoport_example1, self.twoport_example2,
                         self.twoport_example3,
                         self.fourport_example1]

        # constructor from file-object
        file = open(self.test_dir + 'test_1p_gmdif.mdf')
        self.oneport_example1_from_fo = Mdif(file)

    def test_equal(self):
        """ Test the comparison between two Mdif objects """
        self.assertTrue(self.oneport_example1, self.oneport_example1)
        self.assertTrue(self.oneport_example1, self.oneport_example1_from_fo)
        self.assertTrue(self.twoport_example_z, self.twoport_example_yz)

    def test_to_networkset(self):
        """ Test if MDIF are correctly converted into NetworkSet """
        for example in self.examples:
            self.assertIsInstance(example.to_networkset(), NetworkSet)

    def test_params(self):
        """ Test if the params are correctly parsed from the MDIF files """
        self.assertEqual(self.oneport_example1.params, ['Cm'])
        self.assertEqual(self.oneport_example2.params, ['mag', 'Phase'])
        self.assertEqual(self.twoport_example1.params, ['Cm'])
        self.assertEqual(self.twoport_example2.params, ['L1'])
        self.assertEqual(self.fourport_example1.params, ['Cm'])

    def test_to_to_networkset_params(self):
        """ Test if the params are correctly passed to the NetworkSet """
        self.assertEqual(self.oneport_example1.to_networkset().params, ['Cm'])
        self.assertEqual(self.oneport_example2.to_networkset().params, ['mag', 'Phase'])
        self.assertEqual(self.twoport_example1.to_networkset().params, ['Cm'])
        self.assertEqual(self.twoport_example2.to_networkset().params, ['L1'])
        self.assertEqual(self.fourport_example1.to_networkset().params, ['Cm'])

    def test_to_networkset_values(self):
        """ Test if we extract correctly the numerical values """
        # values described in real/imag
        ntwk = self.oneport_example1.to_networkset().sel({'Cm': 7e-16})[0]
        np.testing.assert_equal(ntwk.s[0,0], 0.999999951-0.000312274302j)
        np.testing.assert_equal(ntwk.f[0], 710000000)
        # values described in mag/deg
        ntwk = self.oneport_example2.to_networkset().sel({'mag': 0.25, 'Phase': 180})[0]
        np.testing.assert_equal(ntwk.s_mag[0,0], 0.1)
        np.testing.assert_equal(ntwk.s_deg[0,0], 180)
        np.testing.assert_equal(ntwk.f[0], 1e9)
        # values described in db/deg
        ntwk = self.twoport_example2.to_networkset().sel({'L1': 10})[0]
        np.testing.assert_almost_equal(ntwk.s_db[0,0,0], -0.099191746)
        np.testing.assert_almost_equal(ntwk.s_deg[0,0,0], 64.474118)
        np.testing.assert_almost_equal(ntwk.s_db[0,0,1], -40.635912)
        np.testing.assert_almost_equal(ntwk.s_deg[0,0,1], 154.35237)
        np.testing.assert_almost_equal(ntwk.s_db[0,1,0], -42.635912)
        np.testing.assert_almost_equal(ntwk.s_deg[0,1,0], 150.35237)
        np.testing.assert_equal(ntwk.f[0], 1e9)

    def test_comment_after_BEGIN(self):
        """Test reading a MDIF file which has comments after BEGIN ACDATA. """
        file = self.test_dir + 'test_comment_after_BEGIN.mdf'
        # Mdif Object Init
        mdif = Mdif(file)
        # to Networkset Init
        ns = NetworkSet.from_mdif(file)

    def test_awr_tab_continuation_lines(self):
        """
        Regression test for #1249: MDIF files exported by AWR Microwave Office
        wrap long %-header and data rows onto continuation lines that start
        with a tab. These must be folded back into their previous line before
        the parser sees them, or parsing fails with ValueError on the wrapped
        header tokens (e.g. 'N22X').
        """
        content = (
            "! AWR Design Environment (17603) Fri Mar  7 16:50:01 2025\n"
            "! nPorts: 3, nXvals: 6\n"
            "\n"
            "VAR x1 = 20\n"
            "VAR x2 = 30\n"
            "BEGIN ACDATA\n"
            "# GHz S DB R 50\n"
            "% F N11X N11Y N12X N12Y N13X N13Y N21X N21Y \n"
            "\tN22X N22Y N23X N23Y N31X N31Y N32X N32Y \n"
            "\tN33X N33Y \n"
            "0.1 -9.5574863 179.87028 -3.5265759 -0.08911217 -3.524635 -0.042428161 -3.5265759 -0.08911217 \n"
            "\t-9.5573313 179.84258 -3.5245584 -0.05627355 -3.524635 -0.042428161 -3.5245584 -0.05627355 \n"
            "\t-9.5534535 179.93604 \n"
            "1.1 -9.5640098 178.64805 -3.5296304 -0.95062039 -3.5257785 -0.45939033 -3.5296304 -0.95062039 \n"
            "\t-9.5651793 178.36214 -3.5264164 -0.60217335 -3.5257785 -0.45939033 -3.5264164 -0.60217335 \n"
            "\t-9.5582923 179.34614 \n"
            "END\n"
        )
        with tempfile.NamedTemporaryFile(mode='w', suffix='.mdf', delete=False) as tf:
            tf.write(content)
            path = tf.name
        try:
            ns = NetworkSet.from_mdif(path)
            self.assertEqual(len(ns), 1)
            self.assertEqual(ns[0].nports, 3)
            self.assertEqual(len(ns[0].frequency), 2)
            self.assertEqual(ns[0].params, {'x1': 20.0, 'x2': 30.0})
        finally:
            os.unlink(path)

    def test_read_and_write_back_noise(self):
        net = rf.Network("skrf/io/tests/ts/ex_18.s2p")
        nset1 = NetworkSet([net.copy() for _i in range(4)])

        #nset1 = NetworkSet.from_mdif("amplifier.mdf")
        with tempfile.TemporaryDirectory() as tempdir:
            nset1.write_mdif(os.path.join(tempdir, "out1.mdf"))
            nset2 = NetworkSet.from_mdif(os.path.join(tempdir, "out1.mdf"))
            nset2.write_mdif(os.path.join(tempdir, "out2.mdf"))
            nset3 = NetworkSet.from_mdif(os.path.join(tempdir, "out2.mdf"))
            nset3.write_mdif(os.path.join(tempdir, "out3.mdf"))
            nset4 = NetworkSet.from_mdif(os.path.join(tempdir, "out3.mdf"))
            assert nset1 == nset4

            for n1, n2 in zip(nset1, nset4):
                np.testing.assert_allclose(n1.noise, n2.noise)


suite = unittest.TestLoader().loadTestsFromTestCase(MdifTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
