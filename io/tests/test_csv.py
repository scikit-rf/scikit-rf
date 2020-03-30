
import unittest
import os
import numpy as npy

import skrf as rf

from nose.plugins.skip import SkipTest
from skrf.util import suppress_warning_decorator

class AgilentCSVTestCase(unittest.TestCase):
    ''' 
    AgilentCSVTestCase tests the IO of agilent style CSV files
    '''
    def setUp(self):
        ''' 
        Sets up the test directory and the initializes the members.
        This method gets the currect file path to this file, then gets the file
        name for pna_csv_reim.csv file, then reads it in.
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        filename = os.path.join(self.test_dir, 'pna_csv_reim.csv')
        self.acsv = rf.AgilentCSV(filename)

    def test_columns(self):
        ''' 
        This tests reading of columns from the test file.
        '''
        self.assertEqual(self.acsv.columns, ['Freq(Hz)', '"A,1"(REAL)',
                                             '"A,1"(IMAG)', '"R1,1"(REAL)',
                                             '"R1,1"(IMAG)'])
    @SkipTest # unicode error with carrage returns for p3 vs p2
    def test_comments(self):
        ''' 
        This tests reading of comment lines in the test file.
        '''
        self.assertEqual(self.acsv.comments.strip('\r'), 'this is a comment\nline\n')

    def test_data(self):
        '''
        This tests reading in of the data of the test file.
        '''
        self.assertTrue((self.acsv.data ==
                         npy.array([[750000000000, 1, 2, 3, 4],
                                    [1100000000000, 5, 6, 7,8],
                                   ])).all())

    def test_frequency(self):
        ''' 
        This tests the reading of frequency from the test file
        '''
        self.assertEqual(self.acsv.frequency, rf.F(750e9, 1100e9, 2, 'hz'))

    @suppress_warning_decorator("CSV format unrecognized")
    def test_networks(self):
        '''
        This only tests for execution, not accuracy
        '''
        a = self.acsv.networks

    def test_scalar_networks(self):
        '''
        This only tests for execution, not accuracy
        '''
        a = self.acsv.scalar_networks

