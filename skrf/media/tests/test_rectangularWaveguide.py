import unittest
import os
import skrf as rf

class MediaTestCase(unittest.TestCase):
    '''
    
    '''
    def setUp(self):
        '''
        
        '''
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )
            
        
    @unittest.skip
    def test_line(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'rectangularWaveguideWR10,200mil.s2p')
        
        qucs_ntwk = rf.Network(fname)
        wg = rf.RectangularWaveguide(
            frequency = qucs_ntwk.frequency, 
            a = 100*rf.mil
            )
        skrf_ntwk = wg.thru(z0=50)**wg.line(200*rf.mil)**wg.thru(z0=50)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    

        
