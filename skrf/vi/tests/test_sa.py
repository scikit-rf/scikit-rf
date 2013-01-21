
from numpy.testing import dec
import unittest
from nose.plugins.skip import SkipTest, Skip
import skrf 

try:
    from skrf.vi.sa import HP8500
except:
    raise SkipTest('visa failed to import, skipping')

class HP8500Test(unittest.TestCase):
    def setUp(self):
        self.vi = HP8500(timeout=3)
        self.vi.save_state(2)
    
    def tearDown(self):
        self.vi.recall_state(2)    
    
    def test_frequency(self):
        f = self.vi.frequency
        
    def test_get_ntwk(self):
        n = self.vi.get_ntwk()    
    
    def test_f_start(self):
        f = self.vi.f_start
    
    def test_f_stop(self):
        f = self.vi.f_stop
            
    def test_trace_a(self):
        a = self.vi.trace_a
    
    def test_trace_b(self):
        b = self.vi.trace_b    
    
    def test_single_sweep(self):
        self.vi.single_sweep()
    
    def test_goto_local(self):
        self.vi.goto_local()
    
    def test_cont_sweep(self):
        self.vi.cont_sweep()  
    
    def test_sweep(self):
        done = self.vi.sweep() 
        
    def test_save_state(self):
        self.vi.save_state()
    
    def test_recall_state(self):
        self.vi.recall_state()

