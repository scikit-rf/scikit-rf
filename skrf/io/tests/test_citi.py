import unittest
import os
import numpy as np
import skrf as rf
from pathlib import Path
from zipfile import ZipFile

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
        
        self.fourport_example1 = rf.Citi(self.test_dir + 'test_4p_citi.cti')

        self.examples = [self.oneport_example1, self.oneport_example2,
                          self.twoport_example1, self.twoport_example2, 
                          self.twoport_example3, self.twoport_example4, 
                          self.fourport_example1]

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


suite = unittest.TestLoader().loadTestsFromTestCase(CitiTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)
