
import unittest
import os
import numpy as npy

import skrf as rf



class AgilentCSVTestCase(unittest.TestCase):
    '''
    '''
    def setUp(self):
        '''       
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        filename = os.path.join(self.test_dir, 'pna_csv_reim.csv')
        self.acsv = rf.AgilentCSV(filename)
    
    def test_columns(self):
        self.assertEqual(self.acsv.columns, ['Freq(Hz)', '"A,1"(REAL)', '"A,1"(IMAG)', '"R1,1"(REAL)', '"R1,1"(IMAG)'])

    def test_comments(self):
        self.assertEqual(self.acsv.comments,
            'this is a comment\r\nline\r\n')
    
    def test_data(self):
        self.assertTrue((self.acsv.data ==
                npy.array([\
                    [750000000000,1,2,3,4],\
                    [1100000000000,5,6, 7,8],\
                    ])).all()
                )
    
    def test_frequency(self):
        self.assertEqual(self.acsv.frequency,
            rf.F(750e9,1100e9,2,'hz'))

    def test_networks(self):
        '''
        ths only tess for execution, not accuracy
        '''
        a = self.acsv.networks

    def test_scalar_networks(self):
        '''
        ths only tess for execution, not accuracy
        '''
        a = self.acsv.scalar_networks
