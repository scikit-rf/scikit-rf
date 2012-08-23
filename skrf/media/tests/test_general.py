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
    
    
    def test_capacitor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'capacitor,p01pF.s2p')
        qucs_ntwk = rf.Network(fname)
        
        a_media = rf.media.Freespace(
            frequency = qucs_ntwk.frequency, 
            )
        skrf_ntwk = a_media.capacitor(.01e-12,nports=2)
        
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
    def test_inductor(self):
        '''
        '''
        fname = os.path.join(self.files_dir,\
                'inductor,p1nH.s2p')
        qucs_ntwk = rf.Network(fname)
        
        a_media = rf.media.Freespace(
            frequency = qucs_ntwk.frequency, 
            )
        skrf_ntwk = a_media.capacitor(.1e-9,nports=2)
        
        self.assertEqual(qucs_ntwk, skrf_ntwk)
    
    def test_scalar_gamma_z0_media(self):
        '''
        test ability to create a Media from scalar quanties for gamma/z0
        '''
        a_media = rf.media.Media(rf.f_wr10, 
            propagation_constant = 1 , 
            characteristic_impedance = 50 , 
            )
        skrf_ntwk = a_media.line(1)
    
    def test_vector_gamma_z0_media(self):
        '''
        test ability to create a Media from vector quanties for gamma/z0
        '''
        a_media = rf.media.Media(rf.f_wr10, 
            propagation_constant = npy.ones(len(rf.f_wr10)) , 
            characteristic_impedance =  50*npy.ones(len(rf.f_wr10)), 
            )
            
        skrf_ntwk = a_media.line(1)
        
    
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
