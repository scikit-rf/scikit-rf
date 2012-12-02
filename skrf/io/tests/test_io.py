
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
        self.pickle_file = os.path.join(self.test_dir, 'pickled.p')
        self.hfss_oneport_file = os.path.join(self.test_dir, 'hfss_oneport.s1p')
        self.hfss_twoport_file = os.path.join(self.test_dir, 'hfss_twoport.s2p')
        self.ntwk1 = rf.Network(os.path.join(self.test_dir, 'ntwk1.s2p'))
        self.ntwk2 = rf.Network(os.path.join(self.test_dir, 'ntwk2.s2p'))
        self.ntwk3 = rf.Network(os.path.join(self.test_dir, 'ntwk3.s2p'))
        self.short = rf.Network(os.path.join(self.test_dir, 'short.s1p'))
        self.match = rf.Network(os.path.join(self.test_dir, 'match.s1p'))
        self.open = rf.Network(os.path.join(self.test_dir, 'open.s1p'))
        self.embeding_network= rf.Network(os.path.join(self.test_dir, 'embedingNetwork.s2p'))
    
    def test_readwrite_network(self):
        rf.write(self.pickle_file,self.ntwk1)
        self.assertEqual(rf.read(self.pickle_file), self.ntwk1)
        os.remove(self.pickle_file)
    
    def test_readwrite_list_of_network(self):
        rf.write(self.pickle_file,[self.ntwk1, self.ntwk2])
        self.assertEqual(rf.read(self.pickle_file), [self.ntwk1, self.ntwk2])
        os.remove(self.pickle_file)
    
    def test_readwrite_networkSet(self):
        '''
        test_readwrite_networkSet
        TODO: need __eq__ method for NetworkSet
        This doesnt test equality between  read/write, because there is no 
        __eq__ test for NetworkSet. it only tests for other errors
        '''
        rf.write(self.pickle_file,rf.NS([self.ntwk1, self.ntwk2]))
        rf.read(self.pickle_file)
        #self.assertEqual(rf.read(self.pickle_file), rf.NS([self.ntwk1, self.ntwk2])
        os.remove(self.pickle_file)    
    
    def test_readwrite_frequency(self):
        freq = rf.Frequency(1,10,10,'ghz')
        rf.write(self.pickle_file,freq)
        self.assertEqual(rf.read(self.pickle_file), freq)
        os.remove(self.pickle_file)

    def test_readwrite_calibration(self):
        ideals, measured = [], []
        std_list = [self.short, self.match,self.open]

        for ntwk in std_list:
            ideals.append(ntwk)
            measured.append(self.embeding_network ** ntwk)

        cal = rf.Calibration(\
                ideals = ideals,\
                measured = measured,\
                type = 'one port',\
                is_reciprocal = True,\
                )
        
        original = cal
        rf.write(self.pickle_file, original)
        unpickled = rf.read(self.pickle_file)
        # TODO: this test should be more extensive 
        self.assertEqual(original.ideals, unpickled.ideals)
        self.assertEqual(original.measured, unpickled.measured)
        
        os.remove(self.pickle_file)
