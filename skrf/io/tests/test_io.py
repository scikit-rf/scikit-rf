
import unittest
import os
import numpy as npy

import skrf as rf


class IOTestCase(unittest.TestCase):
    '''
    '''
    def setUp(self):
        '''       
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'
        self.hfss_oneport_file = os.path.join(self.test_dir, 'hfss_oneport.s1p')
        self.hfss_twoport_file = os.path.join(self.test_dir, 'hfss_twoport.s2p')
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
    
    def test_write(self):
        filename = os.path.join(self.test_dir, 'pickled.p')
        rf.write(filename,self.ntwk1)
        rf.read(filename)  == self.ntwk1
        rf.write( filename,[self.ntwk1,self.ntwk2])
        rf.read(filename)  == [self.ntwk1,self.ntwk2]
        os.remove(os.path.join(self.test_dir, 'pickled.p'))
        
