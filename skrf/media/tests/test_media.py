import unittest
import os
import skrf as rf
import numpy as npy

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
        self.dummy_media = rf.media.Media(
            frequency = rf.Frequency(1,100,21,'ghz'), 
            propagation_constant=1j,
            characteristic_impedance = 50 ,
            )
        
    def test_impedance_mismatch(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'impedanceMismatch,50to25.s2p')
        qucs_ntwk = rf.Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.thru(z0=50)**\
            self.dummy_media.thru(z0=25)
        
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
    def test_resistor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'resistor,1ohm.s2p')
        qucs_ntwk = rf.Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.resistor(1)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
        
    def test_capacitor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'capacitor,p01pF.s2p')
        qucs_ntwk = rf.Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.capacitor(.01e-12)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
    
    def test_inductor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'inductor,p1nH.s2p')
        qucs_ntwk = rf.Network(fname)
        self.dummy_media.frequency = qucs_ntwk.frequency
        skrf_ntwk = self.dummy_media.inductor(.1e-9)
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
    
    def test_scalar_gamma_z0_media(self):
        '''
        test ability to create a Media from scalar quanties for gamma/z0
        '''
        a_media = rf.media.Media(rf.f_wr10, 
            propagation_constant = 1j , 
            characteristic_impedance = 50 , 
            )
        self.assertEqual(a_media.line(1),a_media.line(1))
    
    
    def test_vector_gamma_z0_media(self):
        '''
        test ability to create a Media from vector quanties for gamma/z0
        '''
        a_media = rf.media.Media(rf.f_wr10, 
            propagation_constant = 1j*npy.ones(len(rf.f_wr10)) , 
            characteristic_impedance =  50*npy.ones(len(rf.f_wr10)), 
            )
            
        self.assertEqual(a_media.line(1),a_media.line(1))
    
    
    def test_write_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        self.dummy_media.write_csv(fname)
        os.remove(fname)
    
    
    def test_from_csv(self):
        fname = os.path.join(self.files_dir,\
                'out.csv')
        self.dummy_media.write_csv(fname)
        a_media = rf.Media.from_csv(fname)
        self.assertEqual(a_media,self.dummy_media)
        os.remove(fname)
