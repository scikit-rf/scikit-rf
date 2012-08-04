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
        
        
    def test_impedance_mismatch(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'impedanceMismatch,50to25.s2p')
        qucs_ntwk = rf.Network(fname)
        
        a_media = rf.media.Freespace(
            frequency = qucs_ntwk.frequency, 
            )
        skrf_ntwk = a_media.thru(z0=50)**a_media.thru(z0=25)
        
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
    def test_write_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        rf.wr10.write_csv(fname)
        os.remove(fname)
    
    def test_from_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        rf.wr10.write_csv(fname)
        a_media = rf.Media.from_csv(fname)
        self.assertEqual(a_media, rf.wr10)
        os.remove(fname)
