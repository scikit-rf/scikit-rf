
import unittest
import os
import numpy as npy

import skrf as rf


class CSVTestCase(unittest.TestCase):
    '''
    '''
    def setUp(self):
        '''       
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        
    
    def test_read_pna_csv(self):
        filename = os.path.join(self.test_dir, 'pna_csv_reim.csv')
        header, comments, data = rf.io.csv.read_pna_csv(filename)
        self.assertEqual(header,
            'Freq(Hz),"A,1"(REAL),"A,1"(IMAG),"R1,1"(REAL),"R1,1"(IMAG)\r\n')
        self.assertEqual(comments,
            'this is a comment\r\nline\r\n')
        self.assertTrue((data ==
            npy.array([\
                [750000000000,1,2,3,4],\
                [1100000000000,5,6, 7,8],\
                ])).all()
            )
