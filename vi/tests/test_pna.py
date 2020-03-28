
from numpy.testing import dec
import unittest
from nose.plugins.skip import SkipTest, Skip
import skrf 

try:
    from skrf.vi.vna import PNA
except:
    raise SkipTest('visa failed to import, skipping')

class PNATest(unittest.TestCase):
    def setUp(self):
        self.vi = PNA(timeout=10)
        self.vi.delete_all_meas()
        self.vi.create_meas('s11','s11')
    
    def tearDown(self):
        self.vi.close()
        
            
    def test_idn(self):
        a= self.vi.idn
    
    def test_continuous(self):
        self.vi.continuous = False
        a = self.vi.continuous
        self.assertEqual(a, False)
    
    def test_npoints(self):
        self.vi.npoints = 101
        self.assertEqual(self.vi.npoints, 101)
    
    
    def test_get_frequency(self):
        a = self.vi.get_frequency()
        b = a.f_scaled
    
    def test_set_yscale_auto(self):
        a = self.vi.set_yscale_auto()
        
    def test_if_bw(self):
        self.vi.if_bw = 100
        self.assertEqual(self.vi.if_bw, 100)
    
    def test_get_oneport(self):
        a = self.vi.get_oneport(1)
        b = a.s
        
    def test_get_twoport(self):
        a = self.vi.get_twoport()   
        b = a.s 
    
    def test_get_switchterms(self):
        a = self.vi.get_switch_terms()
    
    def test_get_meas_list(self):
        a = self.vi.get_meas_list()

    def test_get_active_meas(self):
        a = self.vi.get_active_meas()
        
    def test_get_create_meas(self):
        a = self.vi.create_meas('a1','a1')
    
    
