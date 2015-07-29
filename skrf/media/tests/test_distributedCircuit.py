import unittest
import os
from skrf.media import DistributedCircuit
from skrf.network import Network
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
                'distributedCircuit,line1mm.s2p')
        qucs_ntwk = Network(fname)

        a_media = DistributedCircuit(
            frequency = qucs_ntwk.frequency,
            R=1e5, G=1, I=1e-6, C=8e-12
            )
        skrf_ntwk = a_media.thru(z0=50)**a_media.line(1e-3,'m')\
                    **a_media.thru(z0=50)
        self.assertEqual(qucs_ntwk, skrf_ntwk)


