import os
import unittest

import numpy as np

import skrf as rf


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
        self.oneport_example1 = rf.Mdif(self.test_dir + 'test_1p_gmdif.mdf')
        self.oneport_example2 = rf.Mdif(self.test_dir + 'test_1p_gmdif_2.mdf')
        self.twoport_example1 = rf.Mdif(self.test_dir + 'test_2p_gmdif.mdf')
        self.twoport_example2 = rf.Mdif(self.test_dir + 'test_2p_gmdif_2.mdf')
        self.twoport_example3 = rf.Mdif(self.test_dir + 'test_2p_gmdif_3.mdf')
        self.twoport_example_z = rf.Mdif(self.test_dir + 'test_2p_gmdif_z.mdf')
        self.twoport_example_yz = rf.Mdif(self.test_dir + 'test_2p_gmdif_yz.mdf')
        self.threeport_example = rf.Mdif(self.test_dir + 'test_3p_gmdif.mdf')
        self.fourport_example1 = rf.Mdif(self.test_dir + 'test_4p_gmdif.mdf')
        self.fiveport_example1 = rf.Mdif(self.test_dir + 'test_5p_gmdif.mdf')
        self.fiveport_example2 = rf.Mdif(self.test_dir + 'test_5p_gmdif_2.mdf')

        self.examples = [self.oneport_example1, self.oneport_example2,
                         self.twoport_example1, self.twoport_example2,
                         self.twoport_example3,
                         self.threeport_example,
                         self.fourport_example1,
                         self.fiveport_example1, self.fiveport_example2]

        # constructor from file-object
        file = open(self.test_dir + 'test_1p_gmdif.mdf')
        self.oneport_example1_from_fo = rf.Mdif(file)

    def test_equal(self):
        """ Test the comparison between two Mdif objects """
        self.assertTrue(self.oneport_example1, self.oneport_example1)
        self.assertTrue(self.oneport_example1, self.oneport_example1_from_fo)
        self.assertTrue(self.twoport_example_z, self.twoport_example_yz)

        self.assertTrue(self.fiveport_example1, self.fiveport_example2)

    def test_to_networkset(self):
        """ Test if MDIF are correctly converted into NetworkSet """
        for example in self.examples:
            self.assertIsInstance(example.to_networkset(), rf.NetworkSet)

    def test_params(self):
        """ Test if the params are correctly parsed from the MDIF files """
        self.assertEqual(self.oneport_example1.params, ['Cm'])
        self.assertEqual(self.oneport_example2.params, ['mag', 'Phase'])
        self.assertEqual(self.twoport_example1.params, ['Cm'])
        self.assertEqual(self.twoport_example2.params, ['L1'])
        self.assertEqual(self.threeport_example.params, ['x1', 'x2'])
        self.assertEqual(self.fourport_example1.params, ['Cm'])
        self.assertEqual(self.fiveport_example1.params, ['L1', 'R1'])
        self.assertEqual(self.fiveport_example2.params, ['L1', 'R1'])

    def test_to_to_networkset_params(self):
        """ Test if the params are correctly passed to the NetworkSet """
        self.assertEqual(self.oneport_example1.to_networkset().params, ['Cm'])
        self.assertEqual(self.oneport_example2.to_networkset().params, ['mag', 'Phase'])
        self.assertEqual(self.twoport_example1.to_networkset().params, ['Cm'])
        self.assertEqual(self.twoport_example2.to_networkset().params, ['L1'])
        self.assertEqual(self.threeport_example.to_networkset().params, ['x1', 'x2'])
        self.assertEqual(self.fourport_example1.to_networkset().params, ['Cm'])
        self.assertEqual(self.fiveport_example1.to_networkset().params, ['L1', 'R1'])
        self.assertEqual(self.fiveport_example2.to_networkset().params, ['L1', 'R1'])

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
        mdif = rf.Mdif(file)
        # to Networkset Init
        ns = rf.NetworkSet.from_mdif(file)

    def test_read_after_write_mdf(self):
        for mdf in self.examples:
            basename = os.path.basename(mdf.filename)
            new_name = '{}_copy.mdf'.format(basename.split('.')[0])

            nset = mdf.to_networkset()
            nset.write_mdif(new_name)
            cpy_nset = rf.NetworkSet.from_mdif(new_name)

            assert nset == cpy_nset

            os.remove(new_name)

    def test_read_and_write_back_noise(self):
        net = rf.Network("skrf/io/tests/ts/ex_18.s2p")
        nset1 = rf.NetworkSet([net.copy() for _i in range(4)])

        #nset1 = rf.NetworkSet.from_mdif("amplifier.mdf")
        nset1.write_mdif("out1.mdf")
        nset2 = rf.NetworkSet.from_mdif("out1.mdf")
        nset2.write_mdif("out2.mdf")
        nset3 = rf.NetworkSet.from_mdif("out2.mdf")
        nset3.write_mdif("out3.mdf")
        nset4 = rf.NetworkSet.from_mdif("out3.mdf")
        assert nset1 == nset4

        for n1, n2 in zip(nset1, nset4):
            np.testing.assert_allclose(n1.noise, n2.noise)

        os.remove("out1.mdf")
        os.remove("out2.mdf")
        os.remove("out3.mdf")


suite = unittest.TestLoader().loadTestsFromTestCase(MdifTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
