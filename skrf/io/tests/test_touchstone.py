
import unittest
import os
import numpy as npy
from pathlib import Path
from zipfile import ZipFile

import skrf as rf
from skrf.io.touchstone import Touchstone


class TouchstoneTestCase(unittest.TestCase):
    '''
    TouchstoneTestCase tests the IO of Touchstone files
    '''
    def setUp(self):
        '''
        Sets up the test directory
        '''
        self.test_dir = os.path.dirname(os.path.abspath(__file__))+'/'

    def test_read_data(self):
        '''
        This test reads data from simple_touchstone.s2p and compares with known
        true values.
        '''
        filename = os.path.join(self.test_dir, 'simple_touchstone.s2p')
        touch = Touchstone(filename)
        f, s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance)
        f_true = npy.array([1.00000000e+09, 1.10000000e+09])
        s_true = npy.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue((z0 == z0_true))


    def test_read_from_fid(self):
        '''
        This tests reading touch stone data from a file object as compared with
        a string path and name of the file.
        '''
        with open(os.path.join(self.test_dir, 'simple_touchstone.s2p')) as fid:
            touch = Touchstone(fid)
        f, s = touch.get_sparameter_arrays()
        z0 = complex(touch.resistance)
        f_true = npy.array([1.00000000e+09, 1.10000000e+09])
        s_true = npy.array([[[1.+2.j, 5.+6.j], [3.+4.j, 7.+8.j]],
                            [[9.+10.j, 13.+14.j], [11.+12.j, 15.+16.j]]])
        z0_true = 50+50j

        self.assertTrue((f == f_true).all())
        self.assertTrue((s == s_true).all())
        self.assertTrue((z0 == z0_true))

    def test_get_sparameter_data(self):
        '''
        This tests the get_sparameter_data function.

        '''
        with open(os.path.join(self.test_dir, 'simple_touchstone.s2p')) as fid:
            touch = Touchstone(fid)

        spardict = touch.get_sparameter_data(format="ri")
        self.assertTrue("frequency" in spardict)
        self.assertTrue("S11R" in spardict)
        self.assertTrue("S11I" in spardict)
        self.assertTrue("S12R" in spardict)
        self.assertTrue("S12I" in spardict)
        self.assertTrue("S21R" in spardict)
        self.assertTrue("S21I" in spardict)
        self.assertTrue("S22R" in spardict)
        self.assertTrue("S22I" in spardict)
        self.assertTrue("S11DB" not in spardict)
        self.assertTrue("S11M" not in spardict)


    def test_HFSS_touchstone_files(self):
        """ 
        HFSS can export additional information in the Touchstone file
        such as gamma and z0 for each port. However, the way there are stored
        depend of the HFSS version... 
        
        In versions before 2020 R2, data were stored as following:
        
        ! Gamma ! re1 im2 re2 im2 re3 im3 re4 im4  
        !       re5 im5 re6 im6 re7 im7 re8 im8
        !       re9 im9  [etc]
        ! Port Impedancere1 im2 re2 im2 re3 im3 re4 im4  
        !       re5 im5 re6 im6 re7 im7 re8 im8
        !       re9 im9  [etc]
            [NB: there is an extra ! before re1 for Gamma]
            [NB: re1 value is stuck to the 'e' of Impedance]
        
        Since version 2020 R2n the data are stored in a single line:
        
        ! Gamma re1 im2 re2 im2 re3 im3 re4 im4 re5 im5 re6 im6 re7 im7 re8 im8 [etc]
        ! Port Impedance re1 im2 re2 im2 re3 im3 re4 im4 re5 im5 re6 im6 re7 im7 re8 im8 [etc]
            [NB: re1 value is no more stuck to the 'e' of Impedance]
        
        This test checks that the shape of gamma and z0 matche the rank of the Network 
        for Touchstone files of various port obtained from different HFSS version
        """
        HFSS_RELEASES= ['HFSS_2019R2', 'HFSS_2020R2']

        p = Path('.')
        for hfss_release in HFSS_RELEASES:
            for sNp_file in p.glob(hfss_release+'/*.s*'):
                touchst = Touchstone(sNp_file.as_posix())
                gamma, z0 = touchst.get_gamma_z0()
                
                assert(gamma.shape[-1] == touchst.rank)
                assert(z0.shape[-1] == touchst.rank)

suite = unittest.TestLoader().loadTestsFromTestCase(TouchstoneTestCase)
unittest.TextTestRunner(verbosity=2).run(suite)

