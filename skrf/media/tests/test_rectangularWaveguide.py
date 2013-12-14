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
        self.pwd = os.path.join(
            os.path.dirname(os.path.abspath(__file__)))
        
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
    

        
    def test_conductor_loss(self):
        '''
        This only compares the magnitude of the generated line, because
        the loss approximation doesnt account for reactance of field on 
        sidewalls.
        '''
        ntwk = rf.Network(os.path.join(self.pwd, 'wr1p5_1in_swg_Al_0rough.s2p'))
        wg = rf.RectangularWaveguide(
            ntwk.frequency, 
            15*rf.mil, 
            z0=50, 
            rho = 1/(3.8e7),
            )
        self.assertTrue(
            max(abs(wg.line(1*rf.inch).s_mag[:,1,0] - ntwk.s_mag[:,1,0]))<1e-3 )
    
    def test_roughness(self):
        '''
        This only compares the magnitude of the generated line, because
        the loss approximation doesnt account for reactance of field on 
        sidewalls.
        '''
        ntwk = rf.Network(os.path.join(self.pwd, 'wr1p5_1in_swg_Al_100nm_rough.s2p'))
        wg = rf.RectangularWaveguide(
            ntwk.frequency, 
            15*rf.mil, 
            z0=50, 
            rho = 1/(3.8e7),
            roughness = 100e-9,
            )
        self.assertTrue(
            max(abs(wg.line(1*rf.inch).s_mag[:,1,0] - ntwk.s_mag[:,1,0]))<1e-3 )
