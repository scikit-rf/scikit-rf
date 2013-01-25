import unittest
import os
import cPickle as pickle
import skrf as rf


class OnePortStandardCalibration(unittest.TestCase):
    '''
    One-port calibration test.

    loads data for a reciprocal embeding network, and some ideal
    standards. ficticous measurements are made by cascading the ideals
    behind teh embeding network, and a calibration is performed.

    the calculated embedding network and de-embeded ideals are compared
    to originals as a metric of 'working'

    '''
    def setUp(self):
        test_dir = os.path.dirname(__file__)
        self.short = rf.Network(os.path.join(test_dir, 'short.s1p'))
        self.match = rf.Network(os.path.join(test_dir, 'match.s1p'))
        self.open = rf.Network(os.path.join(test_dir, 'open.s1p'))
        self.delay_short = rf.Network(os.path.join(test_dir,
                                                   'delay short.s1p'))
        self.embeding_network = rf.Network(os.path.join(test_dir,
                                               'embedingNetwork.s2p'))
        self.test_dir = test_dir

    def test_standard_calibration(self):
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
        # did we find correct embeding network?
        self.assertEqual(self.embeding_network, cal.error_ntwk)
        # are the de-embeded networks the same as their ideals?
        for ntwk in std_list:
            self.assertEqual(ntwk, cal.apply_cal(self.embeding_network**ntwk))
    
    def test_least_squares_calibration(self):
        ideals, measured = [],[]
        std_list = [self.short, self.match,self.open,self.delay_short]

        for ntwk in std_list:
            ideals.append(ntwk)
            measured.append(self.embeding_network**ntwk)

        cal = rf.Calibration(\
                ideals = ideals,\
                measured = measured,\
                type = 'one port',\
                is_reciprocal = True,\
                )
        # did we find correct embeding network?
        self.assertEqual(self.embeding_network, cal.error_ntwk)
        # are the de-embeded networks the same as their ideals?
        for ntwk in std_list:
            self.assertEqual(ntwk,  cal.apply_cal(self.embeding_network**ntwk))
    
    def test_pickling(self):
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
        
        f = open(os.path.join(self.test_dir, 'pickled_cal.cal'),'wb')
        pickle.dump(original,f)
        f.close()
        f = open(os.path.join(self.test_dir, 'pickled_cal.cal'))
        unpickled = pickle.load(f)
        a = unpickled.error_ntwk
        unpickled.run()
        
        
        # TODO: this test should be more extensive 
        self.assertEqual(original.ideals, unpickled.ideals)
        self.assertEqual(original.measured, unpickled.measured)
        
        f.close()
        os.remove(os.path.join(self.test_dir, 'pickled_cal.cal'))

suite = unittest.TestLoader().loadTestsFromTestCase(OnePortStandardCalibration)
unittest.TextTestRunner(verbosity=2).run(suite)
