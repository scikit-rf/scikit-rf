import unittest
import os
import numpy as npy


from skrf.media import DefinedGammaZ0, Media
from skrf.network import Network
from skrf.frequency import Frequency
import skrf


class DefinedGammaZ0TestCase(unittest.TestCase):
    def setUp(self):
        self.files_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'qucs_prj'
            )
        self.dummy_media = DefinedGammaZ0(
            frequency = Frequency(1,100,21,'ghz'),
            gamma=1j,
            z0 = 50 ,
            )

    def test_impedance_mismatch(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'impedanceMismatch,50to25.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.thru(z0=50)**\
            self.dummy_media.thru(z0=25)

        self.assertEqual(qucs_ntwk, skrf_ntwk)

    def test_resistor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'resistor,1ohm.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.resistor(1)
        self.assertEqual(qucs_ntwk, skrf_ntwk)

    def test_capacitor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'capacitor,p01pF.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.capacitor(.01e-12)
        self.assertEqual(qucs_ntwk, skrf_ntwk)


    def test_inductor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'inductor,p1nH.s2p')
        qucs_ntwk = Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.inductor(.1e-9)
        self.assertEqual(qucs_ntwk, skrf_ntwk)


    def test_scalar_gamma_z0_media(self):
        '''
        test ability to create a Media from scalar quanties for gamma/z0
        and change frequency resolution
        '''
        a = DefinedGammaZ0 (Frequency(1,10,101),gamma=1j,z0 = 50)
        self.assertEqual(a.line(1),a.line(1))
        
        # we should be able to re-sample the media 
        a.npoints = 21
        self.assertEqual(len(a.gamma), len(a))
        self.assertEqual(len(a.z0), len(a))
        self.assertEqual(len(a.z0), len(a))


    def test_vector_gamma_z0_media(self):
        '''
        test ability to create a Media from vector quanties for gamma/z0
        '''
        freq = Frequency(1,10,101)
        a = DefinedGammaZ0(freq,
                           gamma = 1j*npy.ones(len(freq)) ,
                           z0 =  50*npy.ones(len(freq)),
                            )
    
        
        self.assertEqual(a.line(1),a.line(1))
        with self.assertRaises(NotImplementedError):
            a.npoints=4

    def test_write_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        self.dummy_media.write_csv(fname)
        os.remove(fname)

    
    def test_from_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        self.dummy_media.write_csv(fname)
        a_media = DefinedGammaZ0.from_csv(fname)
        self.assertEqual(a_media,self.dummy_media)
        os.remove(fname)
