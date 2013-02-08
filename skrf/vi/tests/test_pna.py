
from numpy.testing import dec
import unittest
from nose.plugins.skip import SkipTest, Skip
import skrf 

try:
    from skrf.vi.sa import PNA
except:
    raise SkipTest('visa failed to import, skipping')

class PNATest(unittest.TestCase):
    def setUp(self):
        self.vi = PNA(timeout=3)
        
    
    def test_continuous(self):
        self.vi.continuous = False
        a = self.vi.continuous
        self.assertTrue(a, False)
    
    

