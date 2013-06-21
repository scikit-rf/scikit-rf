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
            
        

    def test_line(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'coaxial.s1p')
        qucs_ntwk = rf.Network(fname)
        
        a_media = rf.media.Coaxial(
            frequency = qucs_ntwk.frequency, 
            Dint=0.9e-3, Dout=3.177e-3, epsilon_r=2.29, \
            tan_delta=4e-4, sigma=1./0.022e-6
            )
        skrf_ntwk = a_media.thru(z0=50)**a_media.line(1)\
                    **a_media.thru(z0=50)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
        
