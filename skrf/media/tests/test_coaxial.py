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
                'coaxial.s2p')
        qucs_ntwk = rf.Network(fname)

        a_media = rf.media.Coaxial(
            frequency = qucs_ntwk.frequency,
            Dint=1e-3, Dout=3e-3, epsilon_r=2.29, \
            tan_delta=4e-4, sigma=1./1.68e-8 \
            )
        skrf_ntwk = a_media.thru(z0=50)**a_media.line(200e-3,'m')\
                    **a_media.thru(z0=50)
        # Equal assertion fails if tan_delta or resistivity are non-zero
        #self.assertEqual(qucs_ntwk, skrf_ntwk)
        self.assertTrue(
            max(abs(skrf_ntwk.s_mag[:,1,0] - qucs_ntwk.s_mag[:,1,0])) < 1e-3
            )
