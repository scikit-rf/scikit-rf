
from numpy.testing import dec
import unittest
from nose.plugins.skip import SkipTest, Skip
import skrf 

try:
    from skrf.vi.vna import VectorStar
except:
    raise SkipTest('visa failed to import, skipping')

class VectorStarTestCase(unittest.TestCase):
    def setUp(self):
        self.vi = VectorStar(timeout=5)
        
    def test_idn(self):
        a = self.vi.idn
        
    def test_get_twoport(self):
        a = self.vi.get_twoport()
        
    def test_rtl(self):
        self.vi.rtl()

