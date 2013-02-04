
from numpy.testing import dec
import unittest
from nose.plugins.skip import SkipTest, Skip
import skrf 

try:
    from skrf.vi.sa import PNAX
except:
    raise SkipTest('visa failed to import, skipping')

class PNAXTest(unittest.TestCase):
    def setUp(self):
        self.vi = PNAX(timeout=3)
        
    
    

