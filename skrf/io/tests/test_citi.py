import os
import unittest

import numpy as np

import skrf as rf


class CitiTestCase(unittest.TestCase):
    """
    Test the IO of CITI files.
    """
    def setUp(self):
        """
        Sets up the test directory
        """
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/MDIF_CITI_MDL/'

        # constructor from filename
        self.oneport_example1 = rf.Citi(self.test_dir + 'test_1p_citi.cti')
        self.oneport_example2 = rf.Citi(self.test_dir + 'test_1p_citi_2_ri.cti')
        self.twoport_example1 = rf.Citi(self.test_dir + 'test_2p_citi.cti')
        self.twoport_example2 = rf.Citi(self.test_dir + 'test_2p_citi_2.cti')
        self.twoport_example3 = rf.Citi(self.test_dir + 'test_2p_citi_2params.cti')
        self.twoport_example4 = rf.Citi(self.test_dir + 'test_2p_citi_2params_db.cti')
        self.twoport_example5 = rf.Citi(self.test_dir + 'test_2p_citi_3_ri.cti')
        self.fourport_example1 = rf.Citi(self.test_dir + 'test_4p_citi.cti')

        self.examples = [self.oneport_example1, self.oneport_example2,
                          self.twoport_example1, self.twoport_example2,
                          self.twoport_example3, self.twoport_example4,
                          self.twoport_example5, self.fourport_example1]

        # constructor from file-object
        file = open(self.test_dir + 'test_1p_citi.cti')
        self.oneport_example1_from_fo = rf.Citi(file)

    def test_to_networks(self):
        """ Test if CITI data are correctly converted into Networks """
        for ex in self.examples:
            for ntwk in ex.networks:
                self.assertIsInstance(ntwk, rf.Network)

    def test_to_networkset(self):
        """ Test if CITI data are correctly converted into NetworkSet """
        for example in self.examples:
            self.assertIsInstance(example.to_networkset(), rf.NetworkSet)

    def test_params(self):
        """ Test if the params are correctly parsed from the CITI files """
        self.assertEqual(self.oneport_example1.params, ['Cm'])
        self.assertEqual(self.twoport_example1.params, ['Cm'])
        self.assertEqual(self.fourport_example1.params, ['Cm'])

    def test_only_freq_in_var(self):
        """ File without any VAR except for freq should return non empty NetworkSet. """
        file = self.test_dir + 'test_2p_only_freq_VAR.cti'
        cti = rf.Citi(file)
        ns = cti.to_networkset()
        self.assertTrue(ns)  # not empty
        self.assertEqual(len(ns), 1)

    def test_values_1p_1(self):
        """ Test if the values are correctly parsed from the CITI files """
        deg = np.array([
         -0.0178919999, -0.0180179999, -0.0181439998, -0.0182699998,
         -0.0183959998, -0.0185219998, -0.0186479998, -0.0187739998,
         -0.0188999998, -0.0204479998, -0.0205919998, -0.0207359998,
         -0.0208799998, -0.0210239998, -0.0211679998, -0.0213119998,
         -0.0214559997, -0.0215999997, -0.0230039997, -0.0231659997,
         -0.0233279997, -0.0234899997, -0.0236519997, -0.0238139997,
         -0.0239759997, -0.0241379996, -0.0242999996, -0.0255599996,
         -0.0257399996, -0.0259199996, -0.0260999995, -0.0262799995,
         -0.0264599995, -0.0266399995, -0.0268199995, -0.0269999995])
        f = np.array([
                710000000, 715000000, 720000000, 725000000,
                730000000, 735000000, 740000000, 745000000,
                750000000,
        ])
        mag = np.ones(len(deg))
        s1p_1 = rf.magdeg_2_reim(mag, deg)
        ns_1p_1 = self.oneport_example1.to_networkset()
        self.assertEqual(len(ns_1p_1), 4)
        self.assertEqual(len(ns_1p_1[0]), 9)
        np.testing.assert_array_equal(ns_1p_1[0].f, f)

        np.testing.assert_array_almost_equal(ns_1p_1[0].s.squeeze(), s1p_1[:9])

    def test_values_1p_2(self):
        """ Test if the values are correctly parsed from the CITI files """
        s1p_2 = np.array([
            -1.31189E-3 - 1.47980E-3 * 1j,
            -3.67867E-3 - 0.67782E-3 * 1j,
            -3.43990E-3 + 0.58746E-3 * 1j,
            -2.70664E-4 - 9.76175E-4 * 1j,
            +0.65892E-4 - 9.61571E-4 * 1j])
        ns_1p_2 = self.oneport_example2.to_networkset()
        self.assertEqual(len(ns_1p_2), 1)
        self.assertEqual(len(ns_1p_2[0]), len(s1p_2))
        np.testing.assert_array_equal(ns_1p_2[0].f, np.array([1., 2., 3., 4., 5.]))
        np.testing.assert_array_almost_equal(ns_1p_2[0].s.squeeze(), s1p_2)

    def test_values_2p_1(self):
        """ Test if the values are correctly parsed from the CITI files """
        mag = np.array([[[0.999999951, 0.000312274295], [0.000312274295, 0.999999951]]])
        deg = np.array([[[-0.0178919994, 89.982108], [89.982108, -0.0178919994]]])
        s_2p_1 = rf.magdeg_2_reim(mag, deg)
        f = np.array([
                710000000, 715000000, 720000000, 725000000,
                730000000, 735000000, 740000000, 745000000,
                750000000,
        ])
        ns_2p_1 = self.twoport_example1.to_networkset()
        self.assertEqual(len(ns_2p_1), 4)
        self.assertEqual(len(ns_2p_1[0]), 9)
        np.testing.assert_array_equal(ns_2p_1[0].f, f)
        np.testing.assert_array_almost_equal(ns_2p_1[0].s_mag[0], mag.squeeze())
        np.testing.assert_array_almost_equal(ns_2p_1[0].s_deg[0], deg.squeeze())
        np.testing.assert_array_almost_equal(ns_2p_1[0].s[0], s_2p_1[0])

    def test_values_2p_2(self):
        """ Test if the values are correctly parsed from the CITI files """
        mag = np.array([
            [[0.1, 0.3],
             [0.5, 0.7]],
            [[0.2, 0.4],
             [0.6, 0.8]]
            ])
        deg = np.array([
            [[2, 4],
             [6, 8]],
            [[3, 5],
             [7, 9]]
            ])
        s2p_2 = rf.magdeg_2_reim(mag, deg)
        ns_2p_2 = self.twoport_example2.to_networkset()
        self.assertEqual(len(ns_2p_2), 1)
        self.assertEqual(len(ns_2p_2[0]), 2)
        np.testing.assert_array_equal(ns_2p_2[0].f, np.array([1e9, 2e9]))
        np.testing.assert_array_almost_equal(ns_2p_2[0].s.squeeze(), s2p_2)

    def test_values_4p(self):
        """ Test if the values are correctly parsed from the CITI files """
        ns_4p = self.fourport_example1.to_networkset()
        self.assertEqual(len(ns_4p), 3)
        self.assertEqual(len(ns_4p[0]), 51)
        self.assertEqual(ns_4p.coords['Cm'], [9e-16, 8e-16, 7e-16])

suite = unittest.TestLoader().loadTestsFromTestCase(CitiTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
